"""gen_hero_image.py — generate the README hero image.

Two-panel composition:
  LEFT  : side-by-side code snippet (Python+PyTorch+Silero vs C99+LC).
  RIGHT : bar chart of throughput on identical hardware (Zen 4 EPYC VPS, single thread),
          numbers from benchmarks/silero_protocol/bench_vps_n200.json.

Output: paper/figures/hero_readme.png (PNG, 1600x720, 200 DPI for retina display).

Honest numbers source (VPS Zen 4 EPYC, single thread, 32 ms chunk):
  Silero V5    183.1 us/chunk  -> 5,461  chunks/sec
  TEN-VAD      209.4 us/chunk  -> 4,776  chunks/sec
  WebRTC m3      4.6 us/chunk  -> 217,391 chunks/sec   (low quality)
  LC sd4 s4      5.2 us/chunk  -> 192,308 chunks/sec
  Energy         0.2 us/chunk  -> 5,000,000 chunks/sec (no quality)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "hero_readme.png"


# --- Numbers from bench_vps_n200.json (Zen 4 EPYC, single thread) ---
DETECTORS = [
    ("Silero V5",    183.1, "#2E5BFF",     "PyTorch + torchaudio, ~700 MB"),
    ("TEN-VAD",      209.4, "#FF6B35",     "C lib, 731 KB"),
    ("LC Runtime",     5.2, "#6C5CE7",     "C99, ~5 MB compiled"),
]
THROUGHPUT = [1_000_000 / us for (_, us, _, _) in DETECTORS]  # chunks/sec


PYTHON_CODE = """\
# Silero VAD (PyTorch)
import torch
from silero_vad import load_silero_vad

model = load_silero_vad()    # ~2 MB JIT + PyTorch
for chunk in audio_chunks:
    p = model(chunk, sr=16000).item()
    if p > 0.5:
        speech_frame()
# 183 us/chunk on EPYC Zen 4"""


LC_CODE = """\
// LC Runtime (C99)
#include "lc/lc.h"

lc_tissue_t* t = lc_create_1d(1024);
lc_set_kernel(t, LC_KERNEL_CANONICAL);
for (each chunk) {
    lc_inject_1d(t, 512, 1, mag(chunk));
    lc_step(t, 4);
    if (max_field(t) > thr) speech_frame();
}
// 5.2 us/chunk on the same EPYC"""


def main():
    fig = plt.figure(figsize=(16, 8.5), dpi=150, facecolor="white")
    gs = GridSpec(2, 2, figure=fig,
                   width_ratios=[1.1, 1.0],
                   height_ratios=[1, 1],
                   wspace=0.18, hspace=0.32,
                   left=0.04, right=0.97, top=0.91, bottom=0.13)

    # --- top-left: traditional code ---
    ax_py = fig.add_subplot(gs[0, 0])
    ax_py.axis("off")
    ax_py.set_title("Traditional ML VAD: Python + PyTorch + Silero",
                     fontsize=13, fontweight="bold", loc="left",
                     color="#2E5BFF")
    ax_py.text(0.0, 0.95, PYTHON_CODE,
                ha="left", va="top",
                fontfamily="monospace", fontsize=11,
                bbox=dict(facecolor="#F5F7FA",
                          edgecolor="#D8DEE9", boxstyle="round,pad=0.6"))

    # --- bottom-left: LC code ---
    ax_lc = fig.add_subplot(gs[1, 0])
    ax_lc.axis("off")
    ax_lc.set_title("LC Runtime: C99, no GPU, no PyTorch",
                     fontsize=13, fontweight="bold", loc="left",
                     color="#6C5CE7")
    ax_lc.text(0.0, 0.95, LC_CODE,
                ha="left", va="top",
                fontfamily="monospace", fontsize=11,
                bbox=dict(facecolor="#F5F0FF",
                          edgecolor="#C8B6FF", boxstyle="round,pad=0.6"))

    # --- right: throughput bar chart ---
    ax_bar = fig.add_subplot(gs[:, 1])
    names = [d[0] for d in DETECTORS]
    colors = [d[2] for d in DETECTORS]
    subtitle = [d[3] for d in DETECTORS]
    ys = list(range(len(names)))
    bars = ax_bar.barh(ys, THROUGHPUT, color=colors, edgecolor="black",
                        linewidth=0.5, height=0.6)
    ax_bar.set_yticks(ys)
    ax_bar.set_yticklabels(names, fontsize=13, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("Throughput (chunks per second, log scale)",
                       fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="x", which="major", alpha=0.4)
    ax_bar.grid(True, axis="x", which="minor", alpha=0.15)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Annotate each bar
    for i, (bar, tp, sub) in enumerate(zip(bars, THROUGHPUT, subtitle)):
        ax_bar.text(tp * 1.3, i - 0.05, f"{tp:>7,.0f} ch/s",
                     va="center", ha="left",
                     fontsize=13, fontweight="bold")
        ax_bar.text(tp * 1.3, i + 0.20, sub,
                     va="center", ha="left",
                     fontsize=10, color="#555555", style="italic")

    ax_bar.set_xlim(left=1e3, right=2e7)
    ax_bar.set_title("VAD throughput @ 32 ms chunks, 1 thread, AMD EPYC Zen 4",
                      fontsize=12, fontweight="bold")

    # --- top banner ---
    fig.suptitle("Voice Activity Detection: same task, two paradigms",
                  fontsize=16, fontweight="bold", y=0.985, color="black")

    # --- bottom annotation ---
    fig.text(0.5, 0.035,
              "Same hardware, same 32 ms chunks. LC scores ROC-AUC 0.816 vs Silero 0.788 on a music+speech mix",
              ha="center", va="bottom", fontsize=10.5, color="#333333")
    fig.text(0.5, 0.012,
              "(ESC-50 noise + Instagram Reels) — see benchmarks/silero_protocol/RESULTS.md for the full audit.",
              ha="center", va="bottom", fontsize=10, color="#555555", style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
