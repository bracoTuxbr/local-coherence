"""plot_pr_curves_lc.py — TEN-VAD's official PR-curve protocol + LC + WebRTC.

Adapted from https://github.com/TEN-framework/ten-vad/blob/main/examples/plot_pr_curves.py
Same testset (30 wav + 30 scv), same metric (precision-recall), threshold sweep 0-1 step 0.01.

Additions:
  - LC VAD (with both canonical and simple_avg kernel)
  - WebRTC VAD (baseline reference)

Usage:
  python plot_pr_curves_lc.py [--testset DIR] [--out PNG]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as Wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torchaudio
from silero_vad import load_silero_vad
from ten_vad import TenVad
import webrtcvad

# Requires `lcruntime` Python binding (separate repo: bracoTuxbr/lcruntime-python).
HERE = Path(__file__).resolve().parent

import lcruntime as lc

SAMPLE_RATE = 16000


def convert_label_to_framewise(label_file: str, hop_size: int) -> np.ndarray:
    """Match TEN-VAD's official label parser exactly."""
    frame_duration = hop_size / SAMPLE_RATE
    with open(label_file, "r") as f:
        lines = f.readlines()
    content = lines[0].strip().split(",")[1:]
    start = np.array(content[::3], dtype=float)
    end = np.array(content[1:][::3], dtype=float)
    lab_manual = np.array(content[2:][::3], dtype=int)
    assert len(start) == len(end) == len(lab_manual)

    num = np.array(np.round(((end - start) / frame_duration)), dtype=np.int32)
    label_framewise = np.array([])
    for segment_idx in range(len(num)):
        cur_lab = int(lab_manual[segment_idx])
        n = num[segment_idx]
        seg = np.ones(n) if cur_lab == 1 else np.zeros(n)
        label_framewise = np.append(label_framewise, seg)
    frame_num = min(label_framewise.__len__(), int((end[-1] - start[0]) / frame_duration))
    return label_framewise[:frame_num]


def get_precision_recall(scores: np.ndarray, label: np.ndarray, thr: float):
    pred = np.where(scores >= thr, 1, 0)
    if label.sum() == 0:
        return 0, 0
    cm = confusion_matrix(label, pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall


# === Detectors ===
def silero_v5_scores(wav_path: str, model) -> tuple[np.ndarray, int]:
    """Silero V5 at 512-sample frames (TEN's protocol)."""
    WINDOW = 512
    model.reset_states()  # reset BEFORE processing this file
    sr, data = Wavfile.read(wav_path)
    assert sr == SAMPLE_RATE
    audio_f32 = data.astype(np.float32) / 32768.0
    wav = torch.from_numpy(audio_f32)
    probs = []
    with torch.no_grad():
        for i in range(0, len(wav), WINDOW):
            chunk = wav[i:i + WINDOW]
            if len(chunk) < WINDOW:
                break
            probs.append(model(chunk, sr).item())
    return np.array(probs, dtype=np.float32), WINDOW


def ten_vad_scores(wav_path: str, hop_size: int = 256) -> np.ndarray:
    """TEN-VAD at 256-sample frames (official)."""
    vad = TenVad(hop_size, 0.5)
    _, data = Wavfile.read(wav_path)
    num = data.shape[0] // hop_size
    probs = []
    for i in range(num):
        chunk = data[i * hop_size:(i + 1) * hop_size]
        p, _ = vad.process(chunk)
        probs.append(p)
    return np.array(probs, dtype=np.float32)


def lc_vad_scores(wav_path: str, hop_size: int = 256,
                  n_cells: int = 1024, steps_per_chunk: int = 8,
                  sig_delta: int = 1, kernel_id: int | None = None) -> np.ndarray:
    """LC VAD at hop_size-sample chunks. Default config = simple_avg kernel + steps=8 (champion)."""
    if kernel_id is None:
        kernel_id = lc.KERNEL_SIMPLE_AVG
    _, data = Wavfile.read(wav_path)
    if data.dtype != np.float32:
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        else:
            audio = data.astype(np.float32)
    else:
        audio = data
    num = len(audio) // hop_size
    chunks = audio[:num * hop_size].reshape(num, hop_size)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1) + 1e-12)
    rms_log = 20 * np.log10(rms + 1e-9)
    rmin, rmax = float(rms_log.min()), float(rms_log.max())
    if rmax - rmin < 1e-6:
        rmax = rmin + 1
    mag = ((rms_log - rmin) / (rmax - rmin) * 60000).astype(np.int32)
    mag = np.clip(mag, 0, 60000)
    tissue = lc.Tissue1D(n_cells)
    tissue.set_sig_delta(sig_delta)
    tissue.set_kernel(kernel_id)
    scores = np.zeros(num, dtype=np.float32)
    center = n_cells // 2
    for i in range(num):
        if mag[i] > 0:
            tissue.inject(center, 1, int(mag[i]))
        tissue.step(steps_per_chunk)
        field = tissue.get_field()
        if not isinstance(field, np.ndarray):
            field = np.frombuffer(field, dtype=np.uint16)
        scores[i] = float(field.max()) / 60000.0
    tissue.close()
    return scores


def webrtc_scores(wav_path: str, hop_size: int = 256, mode: int = 3) -> np.ndarray:
    """WebRTC requires 10/20/30ms frames; 16ms (256) not directly supported.
    We use 10ms internal frames (160 samples) and aggregate to hop_size."""
    vad = webrtcvad.Vad(mode)
    _, data = Wavfile.read(wav_path)
    num = data.shape[0] // hop_size
    internal_size = SAMPLE_RATE * 10 // 1000   # 160
    scores = np.zeros(num, dtype=np.float32)
    for i in range(num):
        chunk = data[i * hop_size:(i + 1) * hop_size]
        # 1 internal frame of 10ms fits in 16ms chunk (with 6ms remainder ignored)
        s = chunk[:internal_size]
        if len(s) == internal_size:
            scores[i] = 1.0 if vad.is_speech(s.tobytes(), SAMPLE_RATE) else 0.0
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=Path("data/ten-vad-repo/data/ten-vad-testset"))
    ap.add_argument("--out", type=Path, default=Path("lab/silero_protocol/PR_curves_ten_testset.png"))
    ap.add_argument("--out-data", type=Path, default=Path("lab/silero_protocol/pr_data.txt"))
    args = ap.parse_args()

    wavs = sorted(args.testset.glob("*.wav"))
    print(f"testset: {args.testset} ({len(wavs)} wavs)")

    silero_model = load_silero_vad(onnx=False)

    # Accumulators per detector (paired with their hop labels)
    ten_scores_all = np.array([])
    ten_labels_all = np.array([])
    silero_scores_all = np.array([])
    silero_labels_all = np.array([])
    lc_simple_scores_all = np.array([])
    lc_simple_labels_all = np.array([])
    lc_canon_scores_all = np.array([])
    lc_canon_labels_all = np.array([])
    webrtc_scores_all = np.array([])
    webrtc_labels_all = np.array([])

    for wp in wavs:
        scv_path = str(wp).replace(".wav", ".scv")

        # TEN/LC/WebRTC: hop 256
        lbl_256 = convert_label_to_framewise(scv_path, hop_size=256)
        ten_s = ten_vad_scores(str(wp), hop_size=256)
        lc_simple_s = lc_vad_scores(str(wp), hop_size=256, n_cells=1024,
                                     steps_per_chunk=8, sig_delta=1,
                                     kernel_id=lc.KERNEL_SIMPLE_AVG)
        lc_canon_s = lc_vad_scores(str(wp), hop_size=256, n_cells=1024,
                                    steps_per_chunk=4, sig_delta=4,
                                    kernel_id=lc.KERNEL_CANONICAL)
        wrtc_s = webrtc_scores(str(wp), hop_size=256, mode=3)
        # TEN's script skips frame 0 of ten_vad, aligns labels accordingly
        n_ten = min(len(lbl_256), len(ten_s))
        ten_scores_all = np.append(ten_scores_all, ten_s[1:n_ten])
        ten_labels_all = np.append(ten_labels_all, lbl_256[:n_ten - 1])
        n_lc = min(len(lbl_256), len(lc_simple_s))
        lc_simple_scores_all = np.append(lc_simple_scores_all, lc_simple_s[:n_lc])
        lc_simple_labels_all = np.append(lc_simple_labels_all, lbl_256[:n_lc])
        n_lc2 = min(len(lbl_256), len(lc_canon_s))
        lc_canon_scores_all = np.append(lc_canon_scores_all, lc_canon_s[:n_lc2])
        lc_canon_labels_all = np.append(lc_canon_labels_all, lbl_256[:n_lc2])
        n_w = min(len(lbl_256), len(wrtc_s))
        webrtc_scores_all = np.append(webrtc_scores_all, wrtc_s[:n_w])
        webrtc_labels_all = np.append(webrtc_labels_all, lbl_256[:n_w])

        # Silero: hop 512
        lbl_512 = convert_label_to_framewise(scv_path, hop_size=512)
        silero_s, _ = silero_v5_scores(str(wp), silero_model)
        n_s = min(len(lbl_512), len(silero_s))
        silero_scores_all = np.append(silero_scores_all, silero_s[:n_s])
        silero_labels_all = np.append(silero_labels_all, lbl_512[:n_s])

        print(f"  {wp.name}: ten={len(ten_s)}f silero={len(silero_s)}f lc={len(lc_simple_s)}f")

    print()
    print("Sweep thresholds and compute PR")

    thrs = np.arange(0, 1.01, 0.01)
    pr_ten = np.zeros((len(thrs), 2))
    pr_silero = np.zeros((len(thrs), 2))
    pr_lc_simple = np.zeros((len(thrs), 2))
    pr_lc_canon = np.zeros((len(thrs), 2))
    pr_webrtc = np.zeros((len(thrs), 2))

    for i, thr in enumerate(thrs):
        pr_ten[i] = get_precision_recall(ten_scores_all, ten_labels_all, thr)
        pr_silero[i] = get_precision_recall(silero_scores_all, silero_labels_all, thr)
        pr_lc_simple[i] = get_precision_recall(lc_simple_scores_all, lc_simple_labels_all, thr)
        pr_lc_canon[i] = get_precision_recall(lc_canon_scores_all, lc_canon_labels_all, thr)
        pr_webrtc[i] = get_precision_recall(webrtc_scores_all, webrtc_labels_all, thr)

    # Compute Average Precision (AP) using sklearn — handles partial recall coverage correctly
    from sklearn.metrics import average_precision_score, roc_auc_score
    auc_ten = float(average_precision_score(ten_labels_all, ten_scores_all))
    auc_silero = float(average_precision_score(silero_labels_all, silero_scores_all))
    auc_lc_simple = float(average_precision_score(lc_simple_labels_all, lc_simple_scores_all))
    auc_lc_canon = float(average_precision_score(lc_canon_labels_all, lc_canon_scores_all))
    auc_webrtc = float(average_precision_score(webrtc_labels_all, webrtc_scores_all))

    # Also report ROC-AUC for comparison
    roc_ten = float(roc_auc_score(ten_labels_all, ten_scores_all))
    roc_silero = float(roc_auc_score(silero_labels_all, silero_scores_all))
    roc_lc_simple = float(roc_auc_score(lc_simple_labels_all, lc_simple_scores_all))
    roc_lc_canon = float(roc_auc_score(lc_canon_labels_all, lc_canon_scores_all))
    roc_webrtc = float(roc_auc_score(webrtc_labels_all, webrtc_scores_all))

    print()
    print(f"=== Average Precision (sklearn) on TEN-VAD official testset ===")
    print(f"  {'detector':<22} {'AP':>7} {'ROC-AUC':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*10}")
    for name, ap, roc in [
        ("LC simple_avg s8 sd1", auc_lc_simple, roc_lc_simple),
        ("LC canonical s4 sd4", auc_lc_canon, roc_lc_canon),
        ("TEN-VAD", auc_ten, roc_ten),
        ("Silero V5", auc_silero, roc_silero),
        ("WebRTC mode 3", auc_webrtc, roc_webrtc),
    ]:
        print(f"  {name:<22} {ap:>7.4f} {roc:>10.4f}")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(pr_ten[:-1, 1], pr_ten[:-1, 0], color="red", linewidth=2,
            label=f"TEN VAD  (AP {auc_ten:.3f}, ROC {roc_ten:.3f})")
    ax.plot(pr_silero[:-1, 1], pr_silero[:-1, 0], color="blue", linewidth=2,
            label=f"Silero V5  (AP {auc_silero:.3f}, ROC {roc_silero:.3f})")
    ax.plot(pr_lc_simple[:-1, 1], pr_lc_simple[:-1, 0], color="#6C5CE7", linestyle="-", linewidth=2.5,
            label=f"LC simple_avg  (AP {auc_lc_simple:.3f}, ROC {roc_lc_simple:.3f})")
    ax.plot(pr_lc_canon[:-1, 1], pr_lc_canon[:-1, 0], color="#6C5CE7", linestyle="--", alpha=0.7,
            label=f"LC canonical  (AP {auc_lc_canon:.3f}, ROC {roc_lc_canon:.3f})")
    ax.plot(pr_webrtc[:-1, 1], pr_webrtc[:-1, 0], color="grey", linestyle=":", alpha=0.6,
            label=f"WebRTC m3  (AP {auc_webrtc:.3f}, ROC {roc_webrtc:.3f})")

    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision-Recall on TEN-VAD-TestSet (30 wavs, official protocol)",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nplot saved → {args.out}")

    # Also save zoomed (TEN's official xlim 0.65-1, ylim 0.7-1)
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.7, 1.02)
    zoom_out = args.out.with_name(args.out.stem + "_zoom" + args.out.suffix)
    fig.savefig(zoom_out, dpi=200, bbox_inches="tight")
    print(f"zoom plot saved → {zoom_out}")

    # Save raw PR data
    with open(args.out_data, "w") as f:
        f.write("threshold\tdetector\tprecision\trecall\n")
        for name, pr in [("TEN", pr_ten), ("Silero", pr_silero),
                          ("LC_simple_avg", pr_lc_simple), ("LC_canonical", pr_lc_canon),
                          ("WebRTC_m3", pr_webrtc)]:
            for i, thr in enumerate(thrs):
                f.write(f"{thr:.2f}\t{name}\t{pr[i,0]:.4f}\t{pr[i,1]:.4f}\n")
    print(f"raw data → {args.out_data}")


if __name__ == "__main__":
    main()
