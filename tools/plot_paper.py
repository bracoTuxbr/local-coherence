#!/usr/bin/env python3
"""Gera figuras do paper preprint.

Le CSVs em runs/ (UTF-16 LE com BOM, default do PowerShell 5.1).
Filtra linhas de cabecalho do print_cpu via heuristica (linha CSV de dados
comeca com digito ou '-').

Output: paper/figures/*.png em 300 DPI.
"""

from __future__ import annotations
import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def read_csv_utf16(path: Path):
    """Le CSV escrito por PowerShell 5.1 (UTF-16 LE com BOM).

    Ignora linhas que nao parecem dados (sem virgula ou comecam com letra).
    Retorna dict[str, list[float]] indexado por nome da coluna do header.
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8")

    header = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            continue
        # primeira linha com virgula apos linhas de info eh header
        if header is None:
            # detecta header: nao deve ter '=' (info linha) e nao deve comecar
            # com digito
            if "=" in line:
                continue
            if line[0].isdigit() or line[0] == "-":
                # comeca direto com dados? cabecalho ausente, abort
                raise RuntimeError(f"{path}: header not found")
            header = [h.strip() for h in line.split(",")]
            continue
        # linha de dados: tenta parsear como csv
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            continue
        # primeiro campo precisa ser numerico
        try:
            float(parts[0])
        except ValueError:
            continue
        rows.append(parts)

    if header is None:
        raise RuntimeError(f"{path}: no header row")

    out: dict[str, list] = {h: [] for h in header}
    for parts in rows:
        for h, p in zip(header, parts):
            try:
                out[h].append(float(p))
            except ValueError:
                out[h].append(p)
    return out


# ---------------------------------------------------------------- PLOTTING ---


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.figsize": (4.5, 3.0),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_pad_1d(data: dict, out_path: Path) -> None:
    gens = np.array(data["gen"])
    a1d = np.array(data["a1d"])
    pulse = 60000
    r_star = math.log(pulse) / math.log(4)

    expected = 2 * gens + 1

    fig, ax = plt.subplots()
    ax.plot(gens, expected, "k--", linewidth=1.0, label=r"prediction $|A(t)|=2t+1$")
    ax.plot(
        gens,
        a1d,
        "o",
        markersize=3.5,
        color="C0",
        label="observed",
    )
    ax.axvline(r_star, color="C3", linestyle=":", linewidth=1.0)
    ax.text(
        r_star + 0.7,
        ax.get_ylim()[1] * 0.40,
        r"$r^\ast{=}\frac{\log v_0}{\log 4}{\approx}7.94$",
        fontsize=8,
        color="C3",
    )
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$|A(t)|$ (active cells)")
    ax.set_title(r"PAD validation, 1D kernel $(l+2c+r)\!\gg\!2 \cdot 255/256$")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pad_2d(data: dict, out_path: Path) -> None:
    gens = np.array(data["gen"])
    a2d = np.array(data["a2d"])
    pulse = 60000
    r_star = math.log(pulse) / math.log(8) - 1  # cantos morrem 1 antes

    expected = 2 * gens * gens + 2 * gens + 1

    fig, ax = plt.subplots()
    # mostrar so primeiros 12 gens (dead regime apos)
    mask = gens <= 12
    ax.plot(
        gens[mask],
        expected[mask],
        "k--",
        linewidth=1.0,
        label=r"prediction $|A(t)|=2t^{2}+2t+1$",
    )
    ax.plot(
        gens[mask],
        a2d[mask],
        "s",
        markersize=3.5,
        color="C2",
        label="observed",
    )
    ax.axvline(r_star, color="C3", linestyle=":", linewidth=1.0)
    ax.text(
        r_star + 0.2,
        ax.get_ylim()[1] * 0.30,
        r"$r^\ast{-}1{=}4$" + "\n(corners die first)",
        fontsize=8,
        color="C3",
    )
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$|A(t)|$ (active cells)")
    ax.set_title(r"PAD validation, 2D 5-point Manhattan diamond")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_stabilization(data: dict, out_path: Path) -> None:
    gens = np.array(data["gen"])
    a1d = np.array(data["a1d"])
    r1d = np.array(data["r1d"])

    fig, axL = plt.subplots()
    axR = axL.twinx()

    l1 = axL.plot(gens, a1d, "-", color="C0", linewidth=1.2, label=r"$|A(t)|$")
    l2 = axR.plot(gens, r1d, "-", color="C3", linewidth=1.2, label=r"$r_{\rm eff}(t)$")

    # marca peak
    peak_a = int(a1d.max())
    peak_g = int(gens[a1d.argmax()])
    axL.axvline(peak_g, color="C0", linestyle=":", linewidth=0.8, alpha=0.5)
    axL.annotate(
        f"peak $|A|={peak_a}$ at $t={peak_g}$",
        xy=(peak_g, peak_a),
        xytext=(peak_g + 30, peak_a - 2),
        fontsize=8,
        color="C0",
    )

    axL.set_xlabel("generation $t$")
    axL.set_ylabel(r"$|A(t)|$ (active cells)", color="C0")
    axR.set_ylabel(r"$r_{\rm eff}(t)$ (effective radius)", color="C3")
    axL.tick_params(axis="y", labelcolor="C0")
    axR.tick_params(axis="y", labelcolor="C3")
    axR.grid(False)

    axL.set_title("Stabilization trace, 1D pulse $v_0{=}60000$, $n{=}256$K")

    lns = l1 + l2
    axL.legend(lns, [l.get_label() for l in lns], loc="lower right", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_roofline(data: dict, out_path: Path) -> None:
    ws_kb = np.array(data["ws_kb"])
    nsc = np.array(data["ns_cell_median"])
    gbs = np.array(data["bandwidth_gbs"])

    fig, axL = plt.subplots()
    axR = axL.twinx()

    l1 = axL.plot(ws_kb, nsc, "o-", color="C0", linewidth=1.0, markersize=4, label="ns/cell")
    l2 = axR.plot(ws_kb, gbs, "s-", color="C2", linewidth=1.0, markersize=4, label="GB/s")

    axL.set_xscale("log")
    axL.set_xlabel("working set (KB)")
    axL.set_ylabel("ns / cell", color="C0")
    axR.set_ylabel("sustained bandwidth (GB/s)", color="C2")
    axL.tick_params(axis="y", labelcolor="C0")
    axR.tick_params(axis="y", labelcolor="C2")
    axR.grid(False)

    # bandas L1/L2/L3/DRAM
    bands = [
        (4, 32, "L1", "#e8f0fe"),
        (32, 512, "L2", "#fef9e8"),
        (512, 4096, "L3", "#fee8e8"),
        (4096, 1e6, "DRAM", "#f0f0f0"),
    ]
    yl_lo, yl_hi = axL.get_ylim()
    for lo, hi, label, color in bands:
        axL.axvspan(lo, hi, color=color, alpha=0.5, zorder=0)
        x_mid = math.sqrt(lo * hi)
        if x_mid <= ws_kb.max() * 2:
            axL.text(
                x_mid,
                yl_lo + (yl_hi - yl_lo) * 0.05,
                label,
                fontsize=7,
                ha="center",
                va="bottom",
                alpha=0.7,
                fontweight="bold",
            )

    axL.set_xlim(ws_kb.min() * 0.7, ws_kb.max() * 1.4)
    axL.set_title("Roofline: cache hierarchy is absorbed in linear access")

    lns = l1 + l2
    axL.legend(lns, [l.get_label() for l in lns], loc="upper left", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_cpu_vs_time(data: dict, out_path: Path, n_total: int = 262144) -> None:
    """%CPU = |A(t)| / n_total ao longo do tempo (M4 task #57).

    Mostra curva descendente apos peak, demonstrando que freeze
    reduz CPU monotonicamente apos perturbacao se dissipar.
    """
    gens = np.array(data["gen"])
    active = np.array(data["a1d"])
    cpu_pct = 100.0 * active / n_total

    fig, ax = plt.subplots()
    ax.plot(gens, cpu_pct, "-", color="C0", linewidth=1.2)
    ax.fill_between(gens, 0, cpu_pct, alpha=0.25, color="C0")

    # marcadores de fases
    peak_idx = int(np.argmax(active))
    peak_g = int(gens[peak_idx])
    peak_v = float(cpu_pct[peak_idx])
    ax.axvline(peak_g, color="C3", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate(f"peak {peak_v:.3f}% at t={peak_g}",
                xy=(peak_g, peak_v),
                xytext=(peak_g + 30, peak_v * 0.7),
                fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", alpha=0.5))

    # detect convergence (active=0)
    zero_mask = active == 0
    if np.any(zero_mask) and peak_idx > 0:
        first_zero = int(gens[np.where(zero_mask)[0][peak_idx:][0]] if len(np.where(zero_mask)[0]) > peak_idx else gens[-1])
        ax.axvline(first_zero, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(first_zero - 5, ax.get_ylim()[1] * 0.5,
                f"converged\nat t={first_zero}",
                fontsize=8, color="green", ha="right")

    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"% CPU active = $|A(t)| / |\Lambda|$ (×100)")
    ax.set_title("M4 freeze: CPU% decays monotonically after dissipation")
    ax.set_ylim(bottom=0)
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------- MAIN -----


def main() -> None:
    setup_style()

    pad = read_csv_utf16(RUNS / "e18_pad.csv")
    stab = read_csv_utf16(RUNS / "e18_stab.csv")
    roof = read_csv_utf16(RUNS / "e17.csv")

    plot_pad_1d(pad, FIGS / "fig1_pad_1d.png")
    plot_pad_2d(pad, FIGS / "fig2_pad_2d.png")
    plot_stabilization(stab, FIGS / "fig3_stabilization.png")
    plot_roofline(roof, FIGS / "fig4_roofline.png")
    plot_cpu_vs_time(stab, FIGS / "fig5_cpu_vs_time.png", n_total=262144)

    print("Generated figures in", FIGS)
    for p in sorted(FIGS.glob("*.png")):
        sz = p.stat().st_size
        print(f"  {p.name}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
