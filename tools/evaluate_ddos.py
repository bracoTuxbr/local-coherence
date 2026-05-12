#!/usr/bin/env python3
"""evaluate_ddos.py — evaluate DDoS detection on synthetic flows.
Key metric: time from attack start to detection.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORED = ROOT / "runs" / "synth_flows_scored.csv"
LABELS = ROOT / "runs" / "synth_flows_labels.txt"

def load_labels(path):
    d = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line: continue
            k, v = line.split("=", 1)
            d[k] = int(v)
    return d

labels = load_labels(LABELS)
attack_start = labels["attack_start_pt"]
attack_end = labels["attack_end_pt"]
print(f"Attack ground truth: points [{attack_start}, {attack_end}] = ~[{labels['attack_start_s']}, {labels['attack_end_s']}]s\n")

rows = list(csv.DictReader(SCORED.open(encoding="utf-8")))
N = len(rows)
values = [float(r["value"]) for r in rows]
score_smooth = [float(r["score_smooth"]) for r in rows]
stab = [int(r["stability"]) for r in rows]

# 3-sigma baseline over rolling window of 100 points before the current point
def baseline_3sigma_rolling(values, win=100):
    scores = [0.0] * len(values)
    for i in range(win, len(values)):
        window = values[i-win:i]
        mu = sum(window) / win
        var = sum((v-mu)**2 for v in window) / win
        sd = var ** 0.5 + 1e-9
        scores[i] = abs(values[i] - mu) / sd
    return scores

baseline_scores = baseline_3sigma_rolling(values, 100)

# for each threshold, measure time to first detection after attack
def first_detection_after(scores, start_pt, threshold):
    for i in range(start_pt, len(scores)):
        if scores[i] >= threshold:
            return i - start_pt  # points after attack start
    return -1

# false positive rate before the attack
def fp_rate_before(scores, end_baseline, threshold):
    if end_baseline <= 0: return 0
    fps = sum(1 for s in scores[:end_baseline] if s >= threshold)
    return 100.0 * fps / end_baseline

print(f"{'method':25s} | {'thr':>6s} | {'detect (ms)':>13s} | {'FP% pre-attack':>15s}")
print("-" * 80)

# multiple thresholds for our approach
print("\n--- Our stack (score_smooth) ---")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    det_pts = first_detection_after(score_smooth, attack_start, thr)
    det_ms = det_pts * 100 if det_pts >= 0 else -1
    fp = fp_rate_before(score_smooth, attack_start, thr)
    if det_pts >= 0:
        print(f"  ours threshold={thr:.1f}     | {thr:6.2f} | {det_ms:>10d} ms | {fp:>13.2f}%")
    else:
        print(f"  ours threshold={thr:.1f}     | {thr:6.2f} | {'(no detect)':>13s} | {fp:>13.2f}%")

print("\n--- 3-sigma rolling (window=100 pts = 10s) ---")
for thr in [2.0, 3.0, 4.0, 5.0, 6.0]:
    det_pts = first_detection_after(baseline_scores, attack_start, thr)
    det_ms = det_pts * 100 if det_pts >= 0 else -1
    fp = fp_rate_before(baseline_scores, attack_start, thr)
    if det_pts >= 0:
        print(f"  3-sig threshold={thr:.1f}    | {thr:6.2f} | {det_ms:>10d} ms | {fp:>13.2f}%")
    else:
        print(f"  3-sig threshold={thr:.1f}    | {thr:6.2f} | {'(no detect)':>13s} | {fp:>13.2f}%")

# show temporal profile around the attack
print(f"\n--- Score profile around the attack (each row = 1s = 10 pts) ---")
print(f"{'t (s)':>8s} | {'value':>10s} | {'stab':>5s} | {'ours':>7s} | {'3sig':>7s}")
for t_s in range(295, 320):
    pt = t_s * 10
    if pt >= N: break
    avg_v = sum(values[pt:pt+10]) / 10
    avg_st = sum(stab[pt:pt+10]) / 10
    avg_o = sum(score_smooth[pt:pt+10]) / 10
    avg_b = sum(baseline_scores[pt:pt+10]) / 10
    marker = " <-- ATTACK START" if t_s == labels['attack_start_s'] else ""
    print(f"{t_s:>8d} | {avg_v:>10.1f} | {avg_st:>5.1f} | {avg_o:>7.3f} | {avg_b:>7.3f}{marker}")
