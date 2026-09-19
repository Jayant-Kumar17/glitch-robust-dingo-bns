"""Locked journal style for the glitch-resilient DINGO-BNS figures.

Physical sizes are millimetre-accurate. Do not use bbox_inches='tight':
that changes the exported size and therefore the on-page font.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

MM = 1.0 / 25.4
SINGLE_W = 90.0 * MM   # 3.543 in
DOUBLE_W = 190.0 * MM  # 7.480 in

CLEAN = "#1F77B4"
POISON = "#C0392B"
GATED = "#2A9D8F"
ORACLE = "#7F7F7F"
NEUTRAL = "#444444"
GRID = "#E5E5E5"
HELD_OUT_EDGE = "#2A9D8F"

PRETTY = {
    "broadband_burst": "broadband burst",
    "double_blip": "double blip",
    "glitch_train": "glitch train",
    "narrowband_tone": "narrowband tone",
    "ringing": "ringing",
    "scattered_light": "scattered light",
    "sine_gaussian": "sine-Gaussian",
    "whistle": "whistle",
    "Blip": "Blip",
    "Koi_Fish": "Koi Fish",
    "Scattered_Light": "Scattered Light",
    "Tomte": "Tomte",
    "Whistle": "Whistle",
    "Noise_Only": "noise only",
}
HELD_OUT = frozenset({"ringing", "double_blip", "narrowband_tone"})


def mm(width_mm: float, height_mm: float) -> Tuple[float, float]:
    return (width_mm * MM, height_mm * MM)


def apply() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.pad": 2.2,
            "ytick.major.pad": 2.2,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "figure.dpi": 120,
            "axes.titlepad": 4.0,
            "axes.labelpad": 2.5,
        }
    )


def save(fig, outdir: Union[str, Path], name: str) -> None:
    """Write PDF (vector, Type 42) and 300 dpi PNG at the figure's physical size."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=300)
    w, h = fig.get_size_inches()
    print(f"wrote {name}  {w * 25.4:.1f} x {h * 25.4:.1f} mm")
    plt.close(fig)


def panel_letter(ax, letter: str, *, dx: float = -0.04, dy: float = 1.04) -> None:
    ax.text(
        dx,
        dy,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
        color="k",
        clip_on=False,
    )


def takeaway(ax, text: str, loc: str = "upper left", *, color: str = NEUTRAL) -> None:
    """Place a takeaway as an axes title (never on top of data)."""
    loc_title = "left"
    if "right" in loc:
        loc_title = "right"
    ax.set_title(text, fontsize=7, color=color, loc=loc_title, pad=3)


def fraction_axis(ax, which: str = "y") -> None:
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    if which in ("y", "both"):
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(ticks)
    if which in ("x", "both"):
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(ticks)


def restyle_axes(ax, *, grid: bool = False) -> None:
    ax.tick_params(direction="out", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.7)
    else:
        ax.grid(False)


def wilson_ci(k: float, n: float, z: float = 1.0) -> Tuple[float, float]:
    """68 % Wilson interval (z=1). Returns (lo, hi) on [0, 1]."""
    n = float(n)
    if n <= 0:
        return (0.0, 0.0)
    p = float(k) / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_from_rate(rate: float, n: float, z: float = 1.0) -> Tuple[float, float]:
    k = float(rate) * float(n)
    return wilson_ci(k, n, z=z)


def errbar_from_rate(rate: float, n: float) -> Tuple[float, float]:
    lo, hi = wilson_from_rate(rate, n)
    return (rate - lo, hi - rate)


def sci_rho(x: float, digits: int = 1) -> str:
    """Format a whitened SNR as $a \\times 10^{b}$."""
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        return r"—"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** exp
    if digits == 0:
        return rf"$10^{{{exp}}}$"
    return rf"${mant:.{digits}f}\times 10^{{{exp}}}$"


def as_bool(series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()
