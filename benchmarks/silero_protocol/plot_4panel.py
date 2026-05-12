"""plot_4panel.py — Replicate TEN-VAD README-style 4-panel plot with LC added.

Panels (top to bottom):
  1. Input audio waveform
  2. Silero VAD decisions (binary)
  3. TEN-VAD decisions (binary)
  4. LC VAD decisions (binary)

Optional: ground truth band at bottom.

Usage:
  python plot_4panel.py <wav_path> [--out OUT.png] [--segs PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Requires `lcruntime` Python binding (separate repo: bracoTuxbr/lcruntime-python).
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
from bench_silero_protocol import (
    SAMPLE_RATE, CHUNK_SAMPLES, CHUNK_MS,
    load_wav_resample_if_needed, chunkize,
    lc_score_chunks, silero_score_chunks, ten_vad_score_chunks,
    load_segs_to_chunk_labels,
)


def to_binary(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(np.int32)


def expand_to_audio_timeline(decisions: np.ndarray, audio_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand chunk-level binary decisions to audio-sample-timeline x,y arrays for step plot."""
    t = np.arange(len(decisions)) * (CHUNK_SAMPLES / SAMPLE_RATE)
    return t, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--segs", type=Path, default=None)
    ap.add_argument("--lc-thr", type=float, default=0.55)
    ap.add_argument("--silero-thr", type=float, default=0.5)
    ap.add_argument("--ten-thr", type=float, default=0.5)
    args = ap.parse_args()

    if not args.wav.exists():
        sys.exit(f"not found: {args.wav}")

    audio = load_wav_resample_if_needed(args.wav)
    if audio is None:
        sys.exit("failed to load audio")
    n_chunks = len(audio) // CHUNK_SAMPLES
    audio_dur = len(audio) / SAMPLE_RATE
    t_audio = np.arange(len(audio)) / SAMPLE_RATE

    print(f"audio: {args.wav.name}  duration={audio_dur:.2f}s  chunks={n_chunks}")

    silero_scores, _ = silero_score_chunks(audio)
    ten_scores, _ = ten_vad_score_chunks(audio)
    lc_scores, _ = lc_score_chunks(audio, n_cells=1024, steps_per_chunk=8,
                                    sig_delta=1, kernel="simple_avg")

    silero_bin = to_binary(silero_scores, args.silero_thr)
    ten_bin = to_binary(ten_scores, args.ten_thr)
    lc_bin = to_binary(lc_scores, args.lc_thr)

    has_segs = args.segs is not None and args.segs.exists()
    gt_bin = None
    if has_segs:
        gt_bin = load_segs_to_chunk_labels(args.segs, n_chunks)

    n_panels = 4 + (1 if has_segs else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.0 * n_panels),
                              sharex=True, gridspec_kw={"hspace": 0.35})

    axes[0].plot(t_audio, audio, color="#2D5BFF", linewidth=0.4)
    axes[0].set_title(f"Input audio: {args.wav.name}  (duration {audio_dur:.1f}s, mono 16 kHz)",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, alpha=0.2)

    def plot_decisions(ax, scores, decisions, title, color):
        t_chunk = np.arange(len(decisions)) * (CHUNK_SAMPLES / SAMPLE_RATE)
        ax.fill_between(t_chunk, 0, decisions, step="post", color=color, alpha=0.85)
        ax.plot(t_chunk, scores, color="black", linewidth=0.6, alpha=0.45)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("VAD")
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.2)

    plot_decisions(axes[1], silero_scores, silero_bin,
                    f"Silero VAD (threshold {args.silero_thr}, score in grey)", "#FF6B35")
    plot_decisions(axes[2], ten_scores, ten_bin,
                    f"TEN VAD (threshold {args.ten_thr}, score in grey)", "#00B894")
    plot_decisions(axes[3], lc_scores, lc_bin,
                    f"LC VAD (threshold {args.lc_thr}, score in grey)", "#6C5CE7")

    if has_segs and gt_bin is not None:
        t_chunk = np.arange(len(gt_bin)) * (CHUNK_SAMPLES / SAMPLE_RATE)
        axes[4].fill_between(t_chunk, 0, gt_bin, step="post", color="#222222", alpha=0.9)
        axes[4].set_title("Ground truth (manual labels)", fontsize=11, fontweight="bold")
        axes[4].set_ylabel("Speech")
        axes[4].set_ylim(-0.05, 1.10)
        axes[4].set_yticks([0, 1])
        axes[4].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    out = args.out or args.wav.with_suffix(".4panel.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"plot saved → {out}")


if __name__ == "__main__":
    main()
