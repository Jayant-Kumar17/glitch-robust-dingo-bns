#!/usr/bin/env python3
"""Vector figures for the gate-aware importance sampling paper.

Every number below is copied from an archived report in the repository and the
source is named next to it. Run from the paper directory:

    python scripts/make_figures.py          # writes figures/fig2..fig4 *.pdf

Nothing here is simulated or interpolated except the plotted Wilson intervals.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8986", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
W = 6.3  # inches, the text width of the manuscript


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def err(ks, ns):
    vals = [wilson(k, n) for k, n in zip(ks, ns)]
    p = [v[0] for v in vals]
    lo = [max(0.0, v[0] - v[1]) for v in vals]
    hi = [max(0.0, v[2] - v[0]) for v in vals]
    return p, [lo, hi]


def tag(ax, s):
    ax.set_title(f"{s}  " + ax.get_title(loc="left"), loc="left", color=INK)


# ====================================================================
# Fig. 2: where the frozen network fails (PHASE_C_REPORT.md, sections 2 and 3)
# ====================================================================
def fig_collapse():
    # Section 2, "Collapse versus injected rho_w": (bin label, cells, collapsed)
    bins = [("$\\leq$100", 108, 0), ("100–\n170", 27, 0), ("170–\n250", 30, 9),
            ("250–\n350", 9, 6), ("350–\n500", 12, 6), ("500–\n750", 9, 4)]
    # Section 3, loudness sweep: rho_w -> (cells, collapsed) per family
    sweep = {
        "broadband burst": {10: (15, 0), 27: (5, 0), 30: (10, 0), 72: (5, 0), 100: (10, 0), 193: (5, 2),
                            518: (5, 1), 1389: (5, 4), 3728: (5, 5), 10000: (5, 5)},
        "sine-Gaussian": {10: (14, 0), 27: (5, 0), 30: (10, 0), 72: (5, 0), 100: (10, 1), 193: (5, 2),
                          518: (5, 2), 1389: (5, 4), 3728: (5, 5), 10000: (4, 4)},
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})

    x = list(range(len(bins)))
    p, e = err([c for _, _, c in bins], [n for _, n, _ in bins])
    a.bar(x, p, width=0.62, color=BLUE, edgecolor="white", linewidth=1.0, zorder=2)
    a.errorbar(x, p, yerr=e, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    for xi, (_, n, c) in zip(x, bins):
        a.text(xi, min(1.0, wilson(c, n)[2]) + 0.02, f"{c}/{n}", ha="center", va="bottom", fontsize=6.6, color=MUTED)
    a.set_xticks(x, [b_[0] for b_ in bins], rotation=0, fontsize=6.8)
    a.set_ylim(0, 1.08)
    a.set_ylabel("collapsed fraction")
    a.set_xlabel("injected $\\rho_w$ (real O3 glitches)")
    a.set_title("Real glitches, 195 cells", loc="left", color=INK)
    tag(a, "(a)")

    for fam, col, mk, dx in (("broadband burst", ORANGE, "s", 0.96), ("sine-Gaussian", BLUE, "o", 1.04)):
        rr = sorted(sweep[fam])
        ks = [sweep[fam][r][1] for r in rr]
        ns = [sweep[fam][r][0] for r in rr]
        pp, ee = err(ks, ns)
        b.errorbar([r * dx for r in rr], pp, yerr=ee, color=col, marker=mk, ms=4.5, mfc=col, mec="white",
                   mew=0.6, elinewidth=0.7, capsize=1.5, label=fam, zorder=3)
    b.axvline(769, color=MUTED, ls=":", lw=1.0)
    b.text(769 * 1.12, 0.06, "pooled 50%\n$\\rho_w\\approx770$", fontsize=6.8, color=MUTED, va="bottom")
    b.set_xscale("log")
    b.set_ylim(0, 1.04)
    b.set_xlabel("injected $\\rho_w$ (synthetic sweep)")
    b.set_title("Loudness sweep, 138 cells", loc="left", color=INK)
    b.legend(frameon=False, loc="upper left")
    tag(b, "(b)")
    fig.savefig(OUT / "fig2_collapse.pdf")
    plt.close(fig)


# ====================================================================
# Fig. 3: gate tolerance (PHASE_B_REPORT.md section 1; Stage 2 verdict)
# ====================================================================
def fig_tolerance():
    hw = [0.05, 0.1, 0.2, 0.4, 0.8]
    # network alone, recovered out of 20 (PHASE_B_REPORT.md section 1)
    series = [
        ("band-limited, zero fill, near merger", [20, 20, 17, 7, 0], hw, BLUE, "o", "-"),
        ("band-limited, noise fill, near merger", [20, 20, 16, 7, 4], hw, AQUA, "^", "-"),
        ("raw strain, zero fill, near merger", [0, 0, 4, 7, 1], hw, ORANGE, "s", "-"),
        ("band-limited, zero fill, far from merger", [20, 20, 20], [0.1, 0.4, 0.8], GREY, "D", "--"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.3})
    for lab, ks, xs, col, mk, ls in series:
        pp, ee = err(ks, [20] * len(ks))
        a.errorbar(xs, pp, yerr=ee, color=col, marker=mk, ms=4.2, ls=ls, mec="white", mew=0.6,
                   elinewidth=0.7, capsize=1.5, label=lab, zorder=3)
    a.set_xscale("log")
    a.set_xticks(hw, ["0.05", "0.1", "0.2", "0.4", "0.8"])
    a.minorticks_off()
    a.set_ylim(0, 1.04)
    a.set_xlabel("gate half-width (s)")
    a.set_ylabel("fraction passing (network alone)")
    a.set_title("Network alone, 20 cells per point", loc="left", color=INK)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.5, ncol=1, handlelength=2.2)
    tag(a, "(a)")

    # Stage 2 (STAGE2_VERDICT.md): 10 cells per half-width, near merger, band-limited zero fill
    groups = ["$\\pm$0.2 s", "$\\pm$0.4 s", "$\\pm$0.8 s"]
    vals = {
        "network alone": ([8, 3, 0], GREY),
        "ordinary IS": ([8, 3, 0], ORANGE),
        "gated IS, $d_L$ passes": ([10, 9, 5], "#9cc0ec"),
        "gated IS, trusted ($n_{\\rm eff}\\geq100$)": ([9, 3, 0], BLUE),
    }
    nb = len(vals)
    wdt = 0.19
    for i, (lab, (ks, col)) in enumerate(vals.items()):
        xs = [g + (i - (nb - 1) / 2) * (wdt + 0.015) for g in range(len(groups))]
        a_ = b.bar(xs, [k / 10 for k in ks], width=wdt, color=col, edgecolor="white", linewidth=0.8, label=lab, zorder=2)
        for xx, k in zip(xs, ks):
            b.text(xx, k / 10 + 0.02, str(k), ha="center", va="bottom", fontsize=6.5, color=MUTED)
    b.set_xticks(range(len(groups)), groups)
    b.set_ylim(0, 1.1)
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_ylabel("fraction of 10 cells")
    b.set_xlabel("gate half-width")
    b.set_title("Near merger, 10 cells per width", loc="left", color=INK)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.5, ncol=2, handlelength=1.2)
    tag(b, "(b)")
    fig.savefig(OUT / "fig3_tolerance.pdf")
    plt.close(fig)


# ====================================================================
# Fig. 4: real glitches (STAGE3_REPORT.md; crosstab from the Stage 3 cells)
# ====================================================================
def fig_stage3():
    groups = ["damaged\n$\\pm$0.1 s", "damaged\n$\\pm$0.2 s", "broken\n$\\pm$0.1 s", "broken\n$\\pm$0.2 s"]
    untrusted = [8, 10, 10, 14]
    groups = [g + f"\n({u} untr.)" for g, u in zip(groups, untrusted)]
    n = [41, 41, 14, 14]
    vals = {
        "network alone": ([31, 25, 0, 0], GREY),
        "ordinary IS": ([29, 21, 0, 0], ORANGE),
        "gated IS (trusted and passes)": ([32, 30, 4, 0], BLUE),
    }
    untrusted = [8, 10, 10, 14]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    nb = len(vals)
    wdt = 0.24
    for i, (lab, (ks, col)) in enumerate(vals.items()):
        xs = [g + (i - (nb - 1) / 2) * (wdt + 0.02) for g in range(len(groups))]
        pp, ee = err(ks, n)
        a.bar(xs, pp, width=wdt, color=col, edgecolor="white", linewidth=0.8, label=lab, zorder=2)
        a.errorbar(xs, pp, yerr=ee, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=3)
    a.set_xticks(range(len(groups)), groups)
    a.set_ylim(0, 1.0)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.tick_params(axis="x", labelsize=6.8)
    a.set_ylabel("fraction passing")
    a.set_title("110 real-glitch cells", loc="left", color=INK)
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.3), fontsize=6.3, ncol=3, handlelength=1.0, columnspacing=0.6)
    tag(a, "(a)")

    # each method judged by its own n_eff >= 100 flag, damaged cells (REVISION_REPORT.md section 1)
    rows = [("gated IS\n82 analyses", 62, 2, 18), ("ordinary IS\n82 analyses", 50, 31, 1),
            ("ungated IS\n38 cells", 0, 3, 35)]
    ys = [2, 1, 0]
    for y, (lab, tp, tf, ref) in zip(ys, rows):
        tot = tp + tf + ref
        left = 0.0
        for val, col in ((tp, BLUE), (tf, "#e34948"), (ref, "#c9c8c3")):
            b.barh(y, val / tot, left=left, color=col, height=0.55, edgecolor="white", linewidth=1.0, zorder=2)
            if val / tot > 0.06:
                b.text(left + val / tot / 2, y, str(val), ha="center", va="center", fontsize=6.8,
                       color="white" if col != "#c9c8c3" else INK)
            left += val / tot
    b.set_yticks(ys, [r[0] for r in rows], fontsize=6.8)
    b.set_xlim(0, 1)
    b.set_ylim(-0.6, 2.6)
    b.set_xlabel("fraction (damaged cells)")
    b.set_title("Own-flag outcomes", loc="left", color=INK)
    b.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=BLUE, label="trusted, passes"), Patch(color="#e34948", label="trusted, fails"),
                      Patch(color="#c9c8c3", label="refused")], frameon=False, fontsize=6.3,
             loc="upper left", bbox_to_anchor=(-0.05, -0.27), ncol=1, handlelength=1.0)
    tag(b, "(b)")
    fig.savefig(OUT / "fig4_real_glitches.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_collapse()
    fig_tolerance()
    fig_stage3()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")))
