#!/usr/bin/env python3
"""evaluate_nab.py — evaluate NAB detection vs labels and against 3-sigma baseline.
No sklearn. Only csv + json + plain numpy (manually).
"""
from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
NAB_DIR = ROOT / "runs" / "nab"
SCORED_DIR = ROOT / "runs" / "nab_scored"
LABELS = NAB_DIR / "labels.json"

def parse_ts(s: str) -> float:
    """parse timestamp to unix epoch seconds"""
    s = s.split(".")[0]
    return datetime.fromisoformat(s).timestamp()

with LABELS.open(encoding="utf-8") as f:
    labels = json.load(f)

# map our CSV filename -> key in labels.json
def to_label_key(filename: str) -> str:
    # filename: realKnownCause_ec2_request_latency_system_failure.csv
    # key:      realKnownCause/ec2_request_latency_system_failure.csv
    if "_" in filename:
        idx = filename.index("_")
        return filename[:idx] + "/" + filename[idx+1:]
    return filename

# baseline: simple z-score over baseline window (first 1/3)
def zscore_baseline(values):
    n = len(values)
    end = min(n // 3, 1000)
    mu = sum(values[:end]) / end
    var = sum((v-mu)**2 for v in values[:end]) / end
    sd = var ** 0.5 + 1e-9
    return [abs(v - mu) / sd for v in values]

def load_scored(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def in_anomaly_window(ts_epoch, windows):
    for start, end in windows:
        s = parse_ts(start); e = parse_ts(end)
        if s <= ts_epoch <= e:
            return True
    return False

def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for t,p in zip(y_true,y_pred) if t and p)
    fp = sum(1 for t,p in zip(y_true,y_pred) if (not t) and p)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t and (not p))
    prec = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn

def best_threshold_f1(y_true, scores):
    """sweep thresholds and return the one that maximizes F1"""
    sorted_s = sorted(set(scores))
    best = (0.0, 0.0, 0.0, 0.0)  # (f1, prec, rec, thr)
    # test ~50 thresholds
    step = max(1, len(sorted_s) // 50)
    for i in range(0, len(sorted_s), step):
        thr = sorted_s[i]
        y_pred = [s >= thr for s in scores]
        p, r, f1, *_ = precision_recall_f1(y_true, y_pred)
        if f1 > best[0]:
            best = (f1, p, r, thr)
    return best

print(f"\n{'series':55s} | {'n':>5s} | {'anom%':>6s} | {'F1 ours':>9s} | {'F1 3-sig':>9s} | win")
print("-" * 110)

results = []
for csvf in SCORED_DIR.glob("*.csv"):
    key = to_label_key(csvf.name)
    if key not in labels:
        continue
    windows = labels[key]
    if not windows:
        continue
    rows = load_scored(csvf)
    n = len(rows)
    timestamps = [parse_ts(r["timestamp"]) for r in rows]
    values = [float(r["value"]) for r in rows]
    score_smooth = [float(r["score_smooth"]) for r in rows]
    score_baseline = zscore_baseline(values)
    y_true = [in_anomaly_window(t, windows) for t in timestamps]
    n_anom = sum(y_true)
    pct_anom = 100.0 * n_anom / n

    f1_ours, p_o, r_o, thr_o = best_threshold_f1(y_true, score_smooth)
    f1_base, p_b, r_b, thr_b = best_threshold_f1(y_true, score_baseline)
    winner = "OURS" if f1_ours > f1_base else ("BASE" if f1_base > f1_ours else "==")
    name_short = csvf.name.replace(".csv", "")[:55]
    print(f"{name_short:55s} | {n:5d} | {pct_anom:5.1f}% | {f1_ours:.3f}    | {f1_base:.3f}    | {winner}")
    results.append((csvf.name, n, pct_anom, f1_ours, p_o, r_o, f1_base, p_b, r_b))

# summary
if results:
    print("\n=== Summary ===")
    avg_ours = sum(r[3] for r in results) / len(results)
    avg_base = sum(r[6] for r in results) / len(results)
    print(f"  avg F1 ours      : {avg_ours:.3f}")
    print(f"  avg F1 3-sigma   : {avg_base:.3f}")
    wins = sum(1 for r in results if r[3] > r[6])
    losses = sum(1 for r in results if r[6] > r[3])
    print(f"  wins ours        : {wins}/{len(results)}")
    print(f"  wins 3-sigma     : {losses}/{len(results)}")
