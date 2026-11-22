#!/usr/bin/env python3
"""Journal figures from archived results/ (filenames fig1, fig4–fig9, S1–S4).

Paper print order: 1 pipeline, 2 threshold (fig9), 3 real glitches (fig8),
6 heatmap (fig4), 7 synthetic (fig5), 8 ablation (fig6), 9 oracle (fig7).
Paper Figures 4–5 (filenames fig2, fig3) need strain / posterior samples;
see examples/paper_figures_2_3.py.

Usage::

    conda run -n adapt_env python paper/make_figures.py --repo . --out paper/figures

Paths are resolved from the repository root (``Path(__file__).resolve().parents[1]``),
so the script does not depend on the process working directory. ``--repo``
overrides that root when regenerating against a different checkout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adapt.paths import resolve_under_repo  # noqa: E402

from paper.figure_style import (  # noqa: E402
    CLEAN,
    GATED,
    HELD_OUT,
    NEUTRAL,
    ORACLE,
    POISON,
    PRETTY,
    apply,
    as_bool,
    errbar_from_rate,
    fraction_axis,
    mm,
    panel_letter,
    restyle_axes,
    save,
    sci_rho,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _bool(s) -> np.ndarray:
    return as_bool(s)


# ------------------------------------------------------------------ Fig 1
def fig1(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=mm(190, 58))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 58)
    ax.set_axis_off()

    boxes = [
        (3.0, "Contaminated\nstrain  $x(t)$", "#F2F2F2", "wave"),
        (34.0, "Whitened STFT\n(32 × 128)", "#F2F2F2", "spec"),
        (65.0, "CNN time-bin\ndetector", "#D7F0EB", "cnn"),
        (96.0, r"Tukey gates" + "\n" + r"$x_g = w(t)\,x(t)$", "#F2F2F2", "gate"),
        (127.0, "Matched-$\\delta$ rebuild\nkeep original ASD", "#D7F0EB", "rebuild"),
        (158.0, "DINGO-BNS\n(frozen)", "#F8D7DA", "dingo"),
    ]
    bw, bh, by = 28.5, 26.0, 24.0
    for x, title, fc, _kind in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, by), bw, bh, boxstyle="round,pad=0.6,rounding_size=1.2",
                fc=fc, ec=NEUTRAL, lw=0.7, zorder=2,
            )
        )
        ax.text(x + bw / 2, by + bh - 3.0, title, ha="center", va="top",
                fontsize=7, color="k", zorder=3, linespacing=1.15)
    for x, *_ in boxes[:-1]:
        ax.add_patch(
            FancyArrowPatch(
                (x + bw + 0.4, by + bh / 2), (x + 31.0 - 0.4, by + bh / 2),
                arrowstyle="-|>", mutation_scale=7, lw=0.8, color=NEUTRAL, zorder=4,
            )
        )

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 280)
    wave = 0.18 * np.sin(2 * np.pi * 9 * t)
    spike = 1.15 * np.exp(-((t - 0.42) / 0.018) ** 2)
    gated = wave * (1.0 - np.exp(-((t - 0.42) / 0.05) ** 4))

    def _mini(x0, y0, w, h):
        return fig.add_axes([x0 / 190.0, y0 / 58.0, w / 190.0, h / 58.0])

    a_w = _mini(5.2, 25.6, 24.2, 9.6)
    a_w.plot(t, wave + spike, color="k", lw=0.45)
    a_w.plot(t, spike, color=POISON, lw=0.6)
    a_w.set_axis_off()
    a_w.set_xlim(0, 1)

    a_s = _mini(36.2, 25.6, 24.2, 9.6)
    yy, xx = np.mgrid[0:32, 0:48]
    blob = np.exp(-((xx - 18) / 3.2) ** 2 - ((yy - 22) / 2.4) ** 2)
    a_s.imshow(blob + 0.04 * rng.random(blob.shape), cmap="magma", origin="lower", aspect="auto")
    a_s.set_axis_off()

    a_c = _mini(67.4, 26.2, 23.6, 7.0)
    a_c.set_xlim(0, 32)
    a_c.set_ylim(0, 1.2)
    fire = {10, 11, 12, 13}
    for i in range(32):
        a_c.add_patch(Rectangle((i, 0.15), 0.92, 0.7,
                                fc=POISON if i in fire else "#DDDDDD",
                                ec="white", lw=0.2))
    a_c.set_axis_off()
    ax.text(
        79.2, 39.4, "trained here\n(only learned)",
        ha="center", va="top", fontsize=6, color=GATED, linespacing=1.1,
    )

    a_g = _mini(98.2, 25.6, 24.2, 9.6)
    a_g.plot(t, gated, color="k", lw=0.45)
    a_g.axvspan(0.32, 0.52, color=GATED, alpha=0.28, lw=0)
    a_g.set_axis_off()
    a_g.set_xlim(0, 1)

    ax.text(141.2, 34.0, r"$\tilde X = X + \mathcal{F}[x_g]-\mathcal{F}[x]$",
            ha="center", va="center", fontsize=6.5, color=NEUTRAL)
    ax.plot([141.2], [29.4], marker="s", ms=3.5, color=GATED, zorder=5)
    ax.text(141.2, 27.4, "original ASD retained", ha="center", va="top",
            fontsize=6.5, color=GATED)

    lock_x, lock_y = 172.2, 30.6
    ax.add_patch(Arc((lock_x, lock_y + 2.6), 3.2, 3.6, theta1=10, theta2=170,
                     lw=0.9, color=NEUTRAL, zorder=5))
    ax.add_patch(FancyBboxPatch((lock_x - 2.3, lock_y - 2.2), 4.6, 3.6,
                                boxstyle="round,pad=0.15,rounding_size=0.4",
                                fc="#E8A0A4", ec=NEUTRAL, lw=0.6, zorder=5))
    ax.plot([lock_x], [lock_y - 0.2], "o", ms=2.2, color=NEUTRAL, zorder=6)
    ax.text(172.2, 27.4, "no weights changed", ha="center", va="top",
            fontsize=6.5, color=POISON)

    ax.annotate(
        "", xy=(172.2, 5.4), xytext=(172.2, 16.6),
        arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=0.8, mutation_scale=7),
    )
    ax.text(172.2, 3.6, r"posterior $q_\phi(\theta\,|\,\tilde X)$",
            ha="center", va="top", fontsize=7, color="k")

    ax.plot([3.0, 123.5], [8.4, 8.4], color="#BBBBBB", lw=0.7)
    ax.plot([3.0, 3.0], [7.6, 9.2], color="#BBBBBB", lw=0.7)
    ax.plot([123.5, 123.5], [7.6, 9.2], color="#BBBBBB", lw=0.7)
    ax.text(63.0, 6.2, "time domain", ha="center", va="top", fontsize=7, color="#666666")

    ax.plot([127.0, 186.5], [8.4, 8.4], color="#BBBBBB", lw=0.7)
    ax.plot([127.0, 127.0], [7.6, 9.2], color="#BBBBBB", lw=0.7)
    ax.plot([186.5, 186.5], [7.6, 9.2], color="#BBBBBB", lw=0.7)
    ax.text(156.8, 6.2, "frequency domain", ha="center", va="top", fontsize=7, color="#666666")

    ax.plot([65.0, 155.5], [54.2, 54.2], color=GATED, lw=0.8)
    ax.plot([65.0, 65.0], [53.4, 55.0], color=GATED, lw=0.8)
    ax.plot([155.5, 155.5], [53.4, 55.0], color=GATED, lw=0.8)
    ax.text(110.2, 55.6, "this work", ha="center", va="bottom", fontsize=7,
            color=GATED, fontweight="bold")

    ax.text(141.2, 20.4, r"X  naive FFT replace $\to$ 0 % recovery",
            ha="center", va="top", fontsize=6.5, color=POISON)
    ax.text(172.2, 20.4, "X  no retraining",
            ha="center", va="top", fontsize=6.5, color=POISON)

    save(fig, out, "fig1_pipeline")


# ------------------------------------------------------------------ Fig 4
def fig4(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    df = pd.read_csv(repo / "results/stress_test_excision_v1/results.csv")
    loud = pd.read_csv(repo / "results/loudness_audit_v1/loudness.csv")
    syn = loud[loud.source == "synthetic_archive"]
    rho_med = syn.groupby("family").rho_w.median().to_dict()

    held = ["double_blip", "narrowband_tone", "ringing"]
    rest = sorted(f for f in df.family.unique() if f not in held)
    fams = held + rest
    sevs = [3.0, 6.0, 10.0]
    cmap_red = LinearSegmentedColormap.from_list("wred", ["#FFFFFF", POISON])
    cmap_teal = LinearSegmentedColormap.from_list("wteal", ["#FFFFFF", GATED])

    fig = plt.figure(figsize=mm(190, 70))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.38],
        left=0.14, right=0.88, top=0.88, bottom=0.22, wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_rho = fig.add_subplot(gs[0, 3])
    panels = [
        ("Poisoned: collapse", "poison_collapsed", cmap_red),
        ("Gated: recovery", "gated_recovers", cmap_teal),
        ("Centred oracle: recovery", "oracle_recovers", cmap_teal),
    ]
    im = None
    for i, (ax, (title, col, cmap)) in enumerate(zip(axes, panels)):
        M = np.array(
            [[df[(df.family == f) & (df.severity == s)][col].mean() for s in sevs] for f in fams]
        )
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for r in range(len(fams)):
            for c in range(3):
                v = M[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "k")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["3", "6", "10"])
        ax.set_xlabel("in-band severity")
        ax.set_title(title, fontsize=9, pad=3)
        ax.tick_params(length=0)
        ax.set_yticks(range(len(fams)))
        if i == 0:
            labels = [PRETTY[f] + (r" $^\dagger$" if f in HELD_OUT else "") for f in fams]
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)

    ax_rho.set_xlim(0, 1)
    ax_rho.set_ylim(len(fams) - 0.5, -0.5)
    ax_rho.set_yticks([])
    ax_rho.set_xticks([])
    for sp in ax_rho.spines.values():
        sp.set_visible(False)
    ax_rho.set_title(r"median $\rho_w$", fontsize=8, pad=3)
    for r, f in enumerate(fams):
        ax_rho.text(0.05, r, sci_rho(rho_med.get(f, np.nan), 1),
                    ha="left", va="center", fontsize=7, color=NEUTRAL)
    cax = fig.add_axes([0.90, 0.22, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("fraction of cells", fontsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    fig.text(
        0.14, 0.03,
        r"$^\dagger$ held-out: never seen by the detector in training.  "
        r"Each cell: 5 seeds $\times$ 2 ASD policies $=$ 10 injections." + "\n"
        r"Oracle $<$ gated on long-duration families: gate extent, not localization (Fig. 9).",
        fontsize=7, color=NEUTRAL, va="bottom",
    )
    save(fig, out, "fig4_gw170817_heatmap")


# ------------------------------------------------------------------ Fig 5
def fig5(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(repo / "results/stress_test_synthetic_bns_v1/results.csv")
    fams = ["sine_gaussian", "broadband_burst", "scattered_light", "ringing"]
    x = np.arange(len(fams))
    w = 0.36
    collapse = [df[df.family == f].poison_collapsed.mean() for f in fams]
    rec = [df[df.family == f].gated_recovers.mean() for f in fams]
    fire = [df[df.family == f].detector_fired.mean() for f in fams]
    overall = float(df.gated_recovers.mean())

    fig = plt.figure(figsize=mm(90, 74))
    ax = fig.add_axes([0.16, 0.32, 0.80, 0.52])
    restyle_axes(ax)
    ax.bar(x - w / 2, collapse, w, color=POISON, label="poisoned: collapsed")
    ax.bar(x + w / 2, rec, w, color=GATED, label="gated: recovered")
    ax.plot(x, fire, "k_", ms=11, mew=1.4, label="detector fire rate")
    ax.axhline(overall, color=GATED, ls=":", lw=0.8, zorder=1)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    labels = [PRETTY[f] + (r" $^\dagger$" if f in HELD_OUT else "") for f in fams]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("fraction of cells")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7)
    fig.text(
        0.58, 0.97,
        r"20 events $\times$ 4 families $\times$ 2 severities $=$ 160 cells",
        ha="center", va="top", fontsize=7, color=NEUTRAL,
    )
    fig.text(
        0.58, 0.915,
        f"overall gated recovery {overall:.1%}",
        ha="center", va="top", fontsize=7, color=GATED,
    )
    save(fig, out, "fig5_synthetic_panel")


# ------------------------------------------------------------------ Fig 6
def fig6(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    S = _load_json(repo / "results/journal_method_hardening_v2/ablation_summary.json")
    cells = pd.read_csv(repo / "results/journal_method_hardening_v2/ablation_results.csv")
    clean = S["clean_reference_d_L"]
    arms = [
        "adapt_full",
        "dingo_repackage",
        "dingo_repackage_aligned",
        "inpaint_repackage",
        "inpaint_repackage_aligned",
        "fft_replace",
        "gate_welch",
        "glitch_orig_asd",
        "poison_welch",
    ]
    labels = [
        "matched-δ\noriginal ASD",
        "DINGO\nshift +2 s",
        "DINGO\nno shift",
        "inpaint\nshift +2 s",
        "inpaint\nno shift",
        "FFT\nreplace",
        "gate\nWelch ASD",
        "no gate\noriginal ASD",
        "no gate\nWelch ASD",
    ]
    rec = [S["by_arm"][k]["recovers_like_clean"] for k in arms]
    rng = np.random.default_rng(0)
    cols = [GATED, POISON, "#E09F3E", POISON, "#E09F3E", POISON, ORACLE, ORACLE, ORACLE]

    fig = plt.figure(figsize=mm(190, 88))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.05, 1.15],
        left=0.08, right=0.98, top=0.90, bottom=0.30, wspace=0.28,
    )
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    restyle_axes(a1)
    restyle_axes(a2)
    panel_letter(a1, "a", dx=-0.12, dy=1.04)
    panel_letter(a2, "b", dx=-0.12, dy=1.04)
    y = np.arange(len(arms))
    a1.barh(y, rec, color=cols, height=0.72, zorder=3)
    for i, v in enumerate(rec):
        a1.text(v + 0.02, i, f"{v:.0%}", va="center", ha="left", fontsize=7)
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=6.5)
    a1.set_xlim(0, 1.18)
    a1.set_xlabel("clean-like recovery")
    a1.set_title("Same gates, different rewrite", fontsize=9)
    a1.invert_yaxis()

    a2.axhspan(clean["lo"], clean["hi"], color=CLEAN, alpha=0.15, zorder=0)
    a2.axhline(clean["med"], color=CLEAN, lw=1.0, zorder=1)
    a2.axhline(10.0, color=POISON, ls="--", lw=0.6)
    a2.axhline(100.0, color=POISON, ls=":", lw=0.6)
    for i, arm in enumerate(arms):
        vals = cells[cells.arm == arm]["med"].to_numpy(float)
        j = rng.uniform(-0.15, 0.15, size=len(vals))
        a2.scatter(np.full(len(vals), i) + j, vals, s=10, color=cols[i], alpha=0.8, linewidths=0, zorder=3)
        a2.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color="k", lw=1.3, zorder=4)
    a2.set_xticks(np.arange(len(arms)))
    a2.set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    a2.set_ylabel(r"$d_L$ median [Mpc]")
    a2.set_ylim(0, 115)
    a2.set_title(r"Where $d_L$ lands", fontsize=9)
    save(fig, out, "fig6_ablation")


# ------------------------------------------------------------------ Fig 7
def fig7(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(repo / "results/stress_test_excision_v1/results.csv")
    fail = df[~_bool(df.gated_recovers)].copy()
    fams = sorted(df.family.unique(), key=lambda f: -int((_bool(fail.family == f)).sum() if False else (fail.family == f).sum()))
    # sort by gated-failure count
    fams = sorted(df.family.unique(), key=lambda f: -(fail.family == f).sum())
    n_fail = [(fail.family == f).sum() for f in fams]
    n_or = [int(_bool(fail.loc[fail.family == f, "oracle_recovers"]).sum()) for f in fams]

    fig = plt.figure(figsize=mm(190, 74))
    gs = fig.add_gridspec(1, 2, left=0.16, right=0.78, top=0.88, bottom=0.22, wspace=0.38)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    restyle_axes(a1)
    restyle_axes(a2)
    panel_letter(a1, "a", dx=-0.22, dy=1.04)
    panel_letter(a2, "b", dx=-0.12, dy=1.04)

    y = np.arange(len(fams))
    a1.barh(y, n_fail, color="#D0D0D0", label="gated failures", height=0.7)
    a1.barh(y, n_or, color=ORACLE, label="centred oracle: recovered", height=0.7)
    a1.set_yticks(y)
    a1.set_yticklabels([PRETTY[f] for f in fams])
    a1.invert_yaxis()
    a1.set_xlabel("cells")
    a1.set_title(f"{len(fail)} gated failures — detector fired in all {len(fail)}", fontsize=9)
    a1.legend(loc="lower right", fontsize=7)

    fams2 = list(df.family.unique())
    g = [df[df.family == f].gated_recovers.mean() for f in fams2]
    o = [df[df.family == f].oracle_recovers.mean() for f in fams2]
    a2.plot([0, 1], [0, 1], "--", color="#999999", lw=0.7, zorder=1)
    handles = []
    for f, xo, yg in zip(fams2, o, g):
        held = f in HELD_OUT
        h = a2.scatter(
            [xo], [yg], s=36, zorder=3,
            facecolors="none" if held else CLEAN,
            edgecolors=GATED if held else CLEAN, linewidths=1.1,
            label=PRETTY[f] + (r" $^\dagger$" if held else ""),
        )
        handles.append(h)
    a2.set_xlim(-0.03, 1.05)
    a2.set_ylim(-0.03, 1.05)
    a2.set_xlabel("centred oracle: recovered")
    a2.set_ylabel("gated: recovered")
    a2.legend(
        loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=7,
        title="family", title_fontsize=7,
    )
    fig.text(
        0.58, 0.02,
        "Above the diagonal: glitch outlasts the oracle 0.8 s window (data-driven support wins).  "
        r"$^\dagger$ held-out.",
        ha="center", va="bottom", fontsize=7, color=NEUTRAL,
    )
    save(fig, out, "fig7_oracle_gap")


# ------------------------------------------------------------------ Fig 8
def _snr_label(s: str) -> str:
    return {"[8,30)": "[8, 30)", "[30,100)": "[30, 100)",
            "[100,300)": "[100, 300)", "[300,inf)": r"$[300, \infty)$"}.get(s, s)


def fig8(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    S = _load_json(repo / "results/stress_real_glitches_v2/summary.json")
    bins = S["by_snr_bin"]
    labels_order = ["Blip", "Koi_Fish", "Scattered_Light", "Tomte", "Whistle"]
    by_lab = {d["family"]: d for d in S["by_label"]}
    noise = S["noise_control"]

    fig = plt.figure(figsize=mm(190, 84))
    gs = fig.add_gridspec(
        1, 2, left=0.07, right=0.99, top=0.88, bottom=0.42, wspace=0.24,
        width_ratios=[1.05, 1.15],
    )
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    restyle_axes(a1)
    restyle_axes(a2)
    panel_letter(a1, "a", dx=-0.08, dy=1.06)
    panel_letter(a2, "b", dx=-0.06, dy=1.06)
    a1.set_title("By Omicron-SNR bin", fontsize=9, pad=4)
    a2.set_title("By morphology", fontsize=9, pad=4)

    w = 0.22
    x = np.arange(len(bins))
    for i, (key, col, lab) in enumerate((
        ("poison_collapse", POISON, "poisoned: collapsed"),
        ("gated_recovery", GATED, "gated: recovered"),
        ("oracle_recovery", ORACLE, "centred oracle: recovered"),
    )):
        a1.bar(x + (i - 1) * w, [b[key] for b in bins], w, color=col, label=lab, zorder=3)
    a1.plot(x, [b["detector_fire_rate"] for b in bins], "k_", ms=10, mew=1.3,
            label="detector fire rate", zorder=4)
    a1.axvline(1.5, color="#BBBBBB", ls="--", lw=0.7, zorder=1)
    a1.set_xticks(x)
    xt = []
    for b in bins:
        xt.append(
            f"{_snr_label(b['snr_bin'])}\n{b['n_glitches']} gl., n={b['n']}\n"
            + rf"$\rho_w={b['median_injected_rho_w']:.0f}$"
        )
    a1.set_xticklabels(xt, fontsize=6.5)
    a1.set_xlabel("Omicron SNR")
    fraction_axis(a1)
    a1.set_ylabel("fraction of cells")

    names = labels_order + ["noise"]
    x2 = np.arange(len(names))
    rows = [by_lab[k] for k in labels_order] + [{
        "poison_collapse": noise["poison_collapse_rate"],
        "gated_recovery": noise["gated_recovery_rate"],
        "oracle_recovery": noise.get("by_mode", [{}])[0].get("oracle_recovery", 1.0),
        "detector_fire_rate": noise.get("by_mode", [{}])[0].get("detector_fire_rate", 1.0),
        "n": noise["n"],
    }]
    gap = np.array([0, 0, 0, 0, 0, 0.35])
    xx = x2 + gap
    for i, (key, col) in enumerate((
        ("poison_collapse", POISON),
        ("gated_recovery", GATED),
        ("oracle_recovery", ORACLE),
    )):
        a2.bar(xx + (i - 1) * w, [r[key] for r in rows], w, color=col, zorder=3,
               label=None)
    a2.plot(xx, [r["detector_fire_rate"] for r in rows], "k_", ms=10, mew=1.3, zorder=4)
    a2.set_xticks(xx)
    a2.set_xticklabels(
        [PRETTY.get(n, n) for n in labels_order] + ["noise only"],
        rotation=18, ha="right",
    )
    fraction_axis(a2)
    a1.legend(loc="upper center", bbox_to_anchor=(1.08, -0.62), ncol=4, fontsize=7)
    fig.text(
        0.53, 0.02,
        "Dashed line: collapse begins near SNR 100.  Every collapse is a Koi Fish (17/60 Koi-Fish cells).  Noise-only: 0/5 collapse.",
        ha="center", va="bottom", fontsize=7, color=NEUTRAL,
    )
    save(fig, out, "fig8_real_glitches")


# ------------------------------------------------------------------ Fig 9
def _pool_curve(rows: list) -> list:
    """Merge adjacent rho_w points within a factor 1.5, weighted by n."""
    pts = sorted(({**r} for r in rows), key=lambda r: r["rho_w_target"])
    out = []
    i = 0
    while i < len(pts):
        grp = [pts[i]]
        j = i + 1
        while j < len(pts) and pts[j]["rho_w_target"] / grp[0]["rho_w_target"] <= 1.5:
            grp.append(pts[j])
            j += 1
        n = sum(p["n"] for p in grp)
        rho = np.exp(sum(p["n"] * np.log(p["rho_w_target"]) for p in grp) / n)
        collapse = sum(p["n"] * p["collapse"] for p in grp) / n
        rec = sum(p["n"] * p["gated_recovery"] for p in grp) / n
        out.append({"rho_w_target": float(rho), "n": int(n), "collapse": collapse, "gated_recovery": rec})
        i = j
    return out


def fig9(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    S = _load_json(repo / "results/collapse_threshold_v1/summary.json")
    sg = _pool_curve(S["curves"]["sine_gaussian"])
    bb = _pool_curve(S["curves"]["broadband_burst"])
    rho50 = S["collapse_threshold"]["all"]["interpolated_rho_w_at_50pct"]

    fig = plt.figure(figsize=mm(90, 90))
    ax = fig.add_axes([0.16, 0.28, 0.80, 0.55])
    restyle_axes(ax, grid=True)
    ax.set_xscale("log")
    ax.set_xlim(8, 1.3e4)
    fraction_axis(ax)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel(r"whitened optimal SNR $\rho_w$ of injected glitch")
    ax.set_ylabel("fraction of cells")
    fig.text(
        0.16, 0.97,
        r"No collapse below $\rho_w \approx 100$; gated recovery 135/139 overall",
        ha="left", va="top", fontsize=7, color=NEUTRAL,
    )

    ax.axvspan(18, 142, color=CLEAN, alpha=0.12, zorder=0)
    ax.text(np.sqrt(18 * 142), 0.08, "real O3 glitches,\nOmicron SNR 9–30",
            ha="center", va="bottom", fontsize=6.5, color=CLEAN)
    ax.axvline(rho50, color=NEUTRAL, ls=":", lw=0.8)
    ax.text(rho50 * 0.88, 1.09, r"50 % collapse", fontsize=6.5, color=NEUTRAL,
            ha="right", va="bottom")

    def _draw(pts, ls, mk, collapse_lab, rec_lab):
        xs = [p["rho_w_target"] for p in pts]
        yc = [p["collapse"] for p in pts]
        yr = [p["gated_recovery"] for p in pts]
        yerr = np.array([errbar_from_rate(p["collapse"], p["n"]) for p in pts]).T
        ax.errorbar(xs, yc, yerr=yerr, fmt=mk, ls=ls, color=POISON, ms=5,
                    lw=1.1, capsize=2, capthick=0.6, elinewidth=0.7, label=collapse_lab, zorder=4)
        ax.plot(xs, yr, mk, ls=ls, color=GATED, ms=5, lw=1.1, label=rec_lab, zorder=4)

    _draw(sg, "-", "o", "collapse, sine-Gaussian", "gated recovery, sine-Gaussian")
    _draw(bb, "--", "s", "collapse, broadband burst", "gated recovery, broadband burst")

    ax.plot([4.3e3], [0.07], marker="v", color="k", ms=5, zorder=6)
    ax.text(4.3e3, 0.11, "Fig. 5 case", ha="center", va="bottom", fontsize=6.5, color="k")
    ax.plot([4.2e3, 1.0e4], [1.05, 1.05], color="#888888", lw=1.0, clip_on=False)
    ax.plot([4.2e3, 4.2e3], [1.035, 1.065], color="#888888", lw=0.8, clip_on=False)
    ax.plot([1.0e4, 1.0e4], [1.035, 1.065], color="#888888", lw=0.8, clip_on=False)
    ax.text(6.5e3, 1.07, "Figs 6-9 regime", ha="center", va="bottom", fontsize=6.5, color="#666666")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6.5, frameon=False)
    save(fig, out, "fig9_threshold")


# ------------------------------------------------------------------ S1
def figS1(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(repo / "results/stress_test_excision_v1/results.csv")
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=mm(90, 62))
    ax = fig.add_axes([0.22, 0.16, 0.74, 0.68])
    restyle_axes(ax)
    markers = {"narrowband_tone": "o", "whistle": "s", "ringing": "^"}
    for f, mk in markers.items():
        d = df[df.family == f]
        dur = []
        for p in d.params_json:
            q = json.loads(p)
            dur.append(q.get("duration", 3.0 * q.get("decay", np.nan)))
        dur = np.asarray(dur, float)
        ok = _bool(d.oracle_recovers)
        j1 = rng.uniform(-0.08, 0.08, ok.sum())
        j0 = rng.uniform(-0.08, 0.08, (~ok).sum())
        ax.scatter(dur[ok], np.ones(ok.sum()) + j1, marker=mk, s=18,
                   color=GATED, label=PRETTY[f], zorder=3)
        ax.scatter(dur[~ok], np.zeros((~ok).sum()) + j0, marker=mk, s=18,
                   color=POISON, zorder=3)
    ax.axvline(0.8, color=NEUTRAL, ls="--", lw=0.8)
    ax.text(0.82, 0.5, "oracle width\n(2 x 0.4 s)", fontsize=7, color=NEUTRAL, va="center", ha="left")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["oracle fails", "oracle recovers"])
    ax.set_xlabel(r"glitch duration [s]  (ringing: $3\tau$)")
    ax.set_xlim(0, 1.65)
    ax.set_ylim(-0.25, 1.25)
    ax.legend(loc="center left", fontsize=7, frameon=False, bbox_to_anchor=(0.02, 0.52))
    ax.set_title("Centred oracle fails when the glitch outlasts its window", fontsize=7, loc="left", color=NEUTRAL)
    save(fig, out, "figS1_oracle_extent")


# ------------------------------------------------------------------ S2
def figS2(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(repo / "results/clean_gate_control_v1/results.csv")
    S = _load_json(repo / "results/clean_gate_control_v1/summary.json")
    clean = S["clean_reference"]["luminosity_distance"]
    mean_abs = float(S.get("mean_abs_delta_dL_med_mpc", df["delta_dL_med"].abs().mean()))

    fig = plt.figure(figsize=mm(90, 64))
    ax = fig.add_axes([0.16, 0.28, 0.80, 0.60])
    restyle_axes(ax)
    ax.axhspan(clean["lo"], clean["hi"], color=CLEAN, alpha=0.15, zorder=0)
    ax.axhline(clean["med"], color=CLEAN, lw=1.1, zorder=1, label="clean median")
    h1 = df[df.modified_detectors.astype(str) == "H1"]
    both = df[df.modified_detectors.astype(str).str.contains("L1")]
    ax.errorbar(
        h1.t_rel, h1.luminosity_distance_med,
        yerr=np.vstack([
            h1.luminosity_distance_med - h1.luminosity_distance_lo,
            h1.luminosity_distance_hi - h1.luminosity_distance_med,
        ]),
        fmt="o", color=GATED, ms=4, lw=0.8, capsize=1.5, elinewidth=0.7,
        label="H1 gate",
    )
    if len(both):
        ax.errorbar(
            both.t_rel, both.luminosity_distance_med,
            yerr=np.vstack([
                both.luminosity_distance_med - both.luminosity_distance_lo,
                both.luminosity_distance_hi - both.luminosity_distance_med,
            ]),
            fmt="o", color=GATED, ms=5, mfc="white", mew=1.1, lw=0.8,
            capsize=1.5, elinewidth=0.7, label="H1+L1 gate",
        )
    ax.set_xlim(-2.05, -0.15)
    ax.set_ylim(5, 52)
    ax.set_xlabel(r"gate centre $t_{\rm rel}$ [s]")
    ax.set_ylabel(r"$d_L$ median [Mpc]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7)
    ax.set_title(
        rf"0/25 collapse; mean $|\Delta d_L|$ = {mean_abs:.1f} Mpc",
        fontsize=7, loc="left", color=NEUTRAL, pad=4,
    )
    save(fig, out, "figS2_clean_gate_cost")


# ------------------------------------------------------------------ S3
def figS3(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    loud = pd.read_csv(repo / "results/loudness_audit_v1/loudness.csv")
    syn = loud[loud.source == "synthetic_archive"].copy()
    real = loud[loud.source == "real_native"]
    off = loud[loud.source == "official_control"]
    rng = np.random.default_rng(1)
    collapsed = _bool(syn.poison_collapsed)

    fig = plt.figure(figsize=mm(90, 80))
    ax = fig.add_axes([0.18, 0.30, 0.78, 0.54])
    restyle_axes(ax, grid=True)
    ax.set_xscale("log")
    sev_col = {3.0: "#9BB7D4", 6.0: "#4C7FB2", 10.0: "#1F4E79"}
    for sev, col in sev_col.items():
        d = syn[syn.severity == sev]
        c = _bool(d.poison_collapsed)
        y = np.where(c, 1.0, 0.0) + rng.uniform(-0.10, 0.10, len(d))
        ax.scatter(d.rho_w, y, s=10, color=col, alpha=0.7, linewidths=0,
                   label=f"severity {int(sev)}", zorder=3)
    ax.plot(real.rho_w, np.full(len(real), -0.28), "|", color=GATED, ms=7, mew=1.1,
            label="real native", zorder=4)
    if len(off):
        ax.plot(off.rho_w, np.full(len(off), -0.28), "|", color="k", ms=9, mew=1.4,
                label="official control", zorder=5)
    # binned collapse fraction
    edges = np.logspace(np.log10(max(syn.rho_w.min() * 0.8, 80)), np.log10(syn.rho_w.max() * 1.05), 8)
    xc, yc = [], []
    rho = syn.rho_w.to_numpy(float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi)
        if m.sum() >= 8:
            xc.append(np.sqrt(lo * hi))
            yc.append(collapsed[m].mean())
    ax.plot(xc, yc, color=POISON, lw=1.3, zorder=4, label="binned collapse")
    ax.axvline(2700, color=POISON, ls=":", lw=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not collapsed", "collapsed"])
    ax.set_ylim(-0.45, 1.35)
    ax.set_xlabel(r"whitened optimal SNR $\rho_w$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5)
    fig.text(
        0.57, 0.97,
        "Synthetic grid is two orders of magnitude\nlouder than typical real glitches",
        ha="center", va="top", fontsize=7, color=NEUTRAL,
    )
    save(fig, out, "figS3_loudness")


# ------------------------------------------------------------------ S4
def figS4(repo: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    sweep = pd.read_csv(repo / "results/collapse_threshold_v1/results.csv")
    loud = pd.read_csv(repo / "results/loudness_audit_v1/loudness.csv")
    syn = loud[loud.source == "synthetic_archive"][["t_rel", "rho_w", "poison_collapsed"]].copy()
    sw = sweep[["t_rel", "rho_w", "poison_collapsed"]].copy()
    allc = pd.concat([sw, syn], ignore_index=True)
    S = _load_json(repo / "results/collapse_threshold_v1/summary.json")
    split = S["collapse_vs_trel_high_rho"]
    collapsed = _bool(allc.poison_collapsed)

    fig = plt.figure(figsize=mm(90, 72))
    ax = fig.add_axes([0.16, 0.28, 0.80, 0.56])
    restyle_axes(ax, grid=True)
    ax.set_yscale("log")
    ax.scatter(allc.loc[~collapsed, "t_rel"], allc.loc[~collapsed, "rho_w"],
               s=9, color=GATED, alpha=0.7, linewidths=0, label="not collapsed", zorder=3)
    ax.scatter(allc.loc[collapsed, "t_rel"], allc.loc[collapsed, "rho_w"],
               s=9, color=POISON, alpha=0.8, linewidths=0, label="collapsed", zorder=4)
    ax.axhline(3e3, color=NEUTRAL, ls="--", lw=0.8)
    ax.axvline(-1.0, color="#BBBBBB", ls=":", lw=0.6)
    ax.set_xlabel(r"$t_{\rm rel}$ [s]")
    ax.set_ylabel(r"$\rho_w$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    p = split["p_value"]
    fig.text(
        0.58, 0.97,
        rf"at $\rho_w > 3000$: {split['rate_t_rel_gt_-1']:.0%} vs "
        rf"{split['rate_t_rel_lt_-1']:.0%} either side of $-1$ s, $p = {p:.2f}$",
        ha="center", va="top", fontsize=7, color=NEUTRAL,
    )
    save(fig, out, "figS4_collapse_vs_trel")


def fig10(repo: Path, out: Path) -> None:
    """15-parameter JS and synthetic coverage from the multi-event validation."""
    import matplotlib.pyplot as plt

    S = _load_json(repo / "results/multi_param_validation_v1/summary.json")
    names = [
        "delta_chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2",
        "phi_12", "phi_jl", "theta_jn", "luminosity_distance", "geocent_time",
        "psi", "lambda_1", "lambda_2", "chirp_mass",
    ]
    pretty = [
        r"$\Delta\mathcal{M}$", r"$q$", r"$a_1$", r"$a_2$", r"$\theta_1$", r"$\theta_2$",
        r"$\phi_{12}$", r"$\phi_{JL}$", r"$\theta_{JN}$", r"$d_L$", r"$t_c$",
        r"$\psi$", r"$\Lambda_1$", r"$\Lambda_2$", r"$\mathcal{M}$",
    ]
    real = [S["real_koi_fish"]["per_parameter_js"][n]["median"] for n in names]
    synth = [S["synthetic"]["per_parameter_js"][n]["median"] for n in names]
    cov90_c = [S["synthetic_coverage_clean"][n]["coverage_90"] for n in names]
    cov90_g = [S["synthetic_coverage_gated"][n]["coverage_90"] for n in names]

    fig = plt.figure(figsize=mm(190, 78))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.28)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    restyle_axes(a1)
    restyle_axes(a2)
    panel_letter(a1, "a", dx=-0.10, dy=1.06)
    panel_letter(a2, "b", dx=-0.10, dy=1.06)
    x = np.arange(len(names))
    w = 0.38
    a1.bar(x - w / 2, real, w, color=GATED, label="17 real Koi Fish")
    a1.bar(x + w / 2, synth, w, color=CLEAN, label="19 synthetic")
    a1.set_xticks(x)
    a1.set_xticklabels(pretty, rotation=60, ha="right", fontsize=6.5)
    a1.set_ylabel("median 1D JS [nat]")
    a1.set_title("Clean versus adapt_full", fontsize=9)
    a1.legend(fontsize=6.5)
    a2.plot(x, cov90_c, "o", color=CLEAN, ms=4, label="clean 90%")
    a2.plot(x, cov90_g, "s", color=GATED, ms=4, label="gated 90%")
    a2.axhline(0.90, color=NEUTRAL, ls=":", lw=0.7)
    a2.set_xticks(x)
    a2.set_xticklabels(pretty, rotation=60, ha="right", fontsize=6.5)
    a2.set_ylim(0.5, 1.05)
    a2.set_ylabel("empirical coverage")
    a2.set_title("Injected-parameter coverage", fontsize=9)
    a2.legend(fontsize=6.5)
    save(fig, out, "fig10_multiparam")


FIGURES = {
    "fig1": fig1,
    "fig4": fig4,
    "fig5": fig5,
    "fig6": fig6,
    "fig7": fig7,
    "fig8": fig8,
    "fig9": fig9,
    "fig10": fig10,
    "figS1": figS1,
    "figS2": figS2,
    "figS3": figS3,
    "figS4": figS4,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=resolve_under_repo, default=REPO)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--only", nargs="*", default=None, choices=list(FIGURES))
    args = p.parse_args(argv)
    apply()
    repo = Path(args.repo).resolve()
    if args.out is None:
        out = repo / "paper" / "figures"
    else:
        out = resolve_under_repo(args.out, repo_root=repo)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FIGURES)
    for name in names:
        FIGURES[name](repo, out)


if __name__ == "__main__":
    main()
