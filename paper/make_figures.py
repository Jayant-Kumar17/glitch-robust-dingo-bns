"""Generate publication figures for the glitch-resilience paper from results/.

Usage:  python paper/make_figures.py [--repo .] [--out paper/figures]
Produces PDF (vector) + PNG (300 dpi) for Figs 1, 4, 5, 6, 7 and an extra
oracle-extent diagnostic (Fig S1). Figs 2 and 3 need posterior samples that are
not archived in the repo; see make_fig2_fig3_template.py.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default=None, help="output dir (default: <repo>/paper/figures)")
a = ap.parse_args()
R = a.repo
OUT = a.out if a.out is not None else os.path.join(R, "paper", "figures")
np.random.seed(0)  # deterministic jitter in Fig S1
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C_CLEAN, C_POISON, C_GATED, C_ORACLE = "#1f77b4", "#c0392b", "#2a9d8f", "#7f7f7f"
SINGLE, DOUBLE = 3.5, 7.0   # inches (≈90 mm / 190 mm)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig); print("wrote", name)

pretty = {"broadband_burst": "broadband burst", "double_blip": "double blip",
          "glitch_train": "glitch train", "narrowband_tone": "narrowband tone",
          "ringing": "ringing", "scattered_light": "scattered light",
          "sine_gaussian": "sine-Gaussian", "whistle": "whistle"}
heldout = {"ringing", "double_blip", "narrowband_tone"}

# ------------------------------------------------------------------ Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.1)); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    boxes = [
        (2, "Contaminated\nstrain  x(t)", "#eeeeee"),
        (18, "Whitened STFT\n(32 × 128 bins)", "#eeeeee"),
        (34, "CNN time-bin\ndetector\n(trained here)", "#d7f0eb"),
        (50, "Tukey gates\nx_g = w(t)·x(t)", "#eeeeee"),
        (66, "Matched-δ rebuild\nX + F[xg] − F[x]\nkeep original ASD", "#d7f0eb"),
        (84, "DINGO-BNS\n(frozen weights)", "#f9e1e1"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 9), 14, 13, boxstyle="round,pad=0.3",
                                    fc=col, ec="#444", lw=0.8))
        ax.text(x + 7, 15.5, txt, ha="center", va="center", fontsize=6.2)
    for x, _, _ in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x + 14.3, 15.5), (x + 15.8, 15.5),
                                     arrowstyle="-|>", mutation_scale=8, color="#444"))
    ax.text(91, 5.5, "posterior  q_φ(θ | X̃)", ha="center", fontsize=6.6, style="italic")
    ax.text(41, 26.5, "learned component (this work)", ha="center", fontsize=6.5, color="#2a9d8f")
    ax.text(91, 26.5, "weights untouched", ha="center", fontsize=6.5, color="#c0392b")
    ax.text(2, 3, "Time domain", fontsize=6.5, color="#666")
    ax.text(66, 3, "Frequency domain", fontsize=6.5, color="#666")
    ax.plot([2, 63], [1.5, 1.5], color="#bbb", lw=0.6); ax.plot([66, 98], [1.5, 1.5], color="#bbb", lw=0.6)
    save(fig, "fig1_pipeline")

# ------------------------------------------------------------------ Fig 4
def fig4():
    df = pd.read_csv(f"{R}/results/stress_test_excision_v1/results.csv")
    fams = sorted(df.family.unique(), key=lambda f: (f not in heldout, f))
    sevs = [3.0, 6.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    panels = [("Poisoned: collapse", "poison_collapsed", "Reds"),
              ("Gated: recovery", "gated_recovers", "Greens"),
              ("Centred-oracle: recovery", "oracle_recovers", "Greens")]
    for ax, (title, col, cmap) in zip(axes, panels):
        M = np.array([[df[(df.family == f) & (df.severity == s)][col].mean() for s in sevs] for f in fams])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(fams)):
            for j in range(len(sevs)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(s)}" for s in sevs])
        ax.set_xlabel("in-band severity (SNR)"); ax.set_title(title)
    axes[0].set_yticks(range(len(fams)))
    axes[0].set_yticklabels([pretty[f] + (" †" if f in heldout else "") for f in fams])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="fraction of cells")
    fig.text(0.01, -0.12, "† held-out family (not seen by the detector in training). "
             "Each cell: 5 seeds × 2 ASD policies = 10 injections.", fontsize=6.5)
    save(fig, "fig4_gw170817_heatmap")

# ------------------------------------------------------------------ Fig 5
def fig5():
    df = pd.read_csv(f"{R}/results/stress_test_synthetic_bns_v1/results.csv")
    fams = sorted(df.family.unique())
    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    ax.bar(x - w / 2, [df[df.family == f].poison_collapsed.mean() for f in fams], w,
           color=C_POISON, label="poisoned: collapsed")
    ax.bar(x + w / 2, [df[df.family == f].gated_recovers.mean() for f in fams], w,
           color=C_GATED, label="gated: recovered")
    if "detector_fired" in df:
        ax.plot(x, [df[df.family == f].detector_fired.mean() for f in fams], "k_",
                ms=14, mew=1.2, label="detector fire rate")
    ax.set_xticks(x); ax.set_xticklabels([pretty[f] for f in fams], rotation=20, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("fraction of cells")
    ax.axhline(df.gated_recovers.mean(), color=C_GATED, ls=":", lw=0.8)
    ax.text(0.5, df.gated_recovers.mean() + 0.02, f"overall {df.gated_recovers.mean():.1%}",
            ha="center", fontsize=6.5, color="k")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=3, fontsize=6.3)
    save(fig, "fig5_synthetic_panel")

# ------------------------------------------------------------------ Fig 6
def fig6():
    S = json.load(open(f"{R}/results/journal_method_hardening_v1/ablation_summary.json"))
    clean = S["clean_reference_d_L"]
    arms = ["poison_welch", "glitch_orig_asd", "gate_welch", "fft_replace", "adapt_full"]
    labels = ["no gate\nWelch ASD", "no gate\norig. ASD", "gate\nWelch ASD",
              "gate, FFT\nreplace", "gate, matched-δ\norig. ASD"]
    rec = [S["by_arm"][k]["recovers_like_clean"] for k in arms]
    med = [S["by_arm"][k]["median_med_d_L"] for k in arms]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
    cols = [C_POISON, C_POISON, C_ORACLE, C_ORACLE, C_GATED]
    b = a1.bar(range(5), rec, color=cols)
    for i, v in enumerate(rec):
        a1.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
    a1.set_xticks(range(5)); a1.set_xticklabels(labels, fontsize=5.6)
    a1.set_ylim(0, 1); a1.set_ylabel("clean-like recovery (N = 16)")
    a1.set_title("Which ingredients matter?")
    a2.axhspan(clean["lo"], clean["hi"], color=C_CLEAN, alpha=0.15, label="clean 90% CI")
    a2.axhline(clean["med"], color=C_CLEAN, lw=1, label="clean median")
    a2.plot(range(5), med, "o", color="k", ms=4)
    a2.set_xticks(range(5)); a2.set_xticklabels(["", "", "", "", ""])
    a2.set_ylabel("median $d_L$ across cells [Mpc]"); a2.set_title("Where does $d_L$ land?")
    a2.axhline(10, color="#999", ls="--", lw=0.7); a2.text(4.4, 11.5, "prior floor", fontsize=6, color="#666", ha="right")
    a2.legend(frameon=False, loc="upper left"); a2.set_ylim(0, 50)
    for i, l in enumerate(labels):
        a2.text(i, -3, l.split("\n")[0], ha="center", fontsize=5.8, rotation=25, va="top")
    save(fig, "fig6_ablation")

# ------------------------------------------------------------------ Fig 7 + S1
def fig7():
    df = pd.read_csv(f"{R}/results/stress_test_excision_v1/results.csv")
    fail = df[~df.gated_recovers]
    fams = sorted(df.family.unique(), key=lambda f: -len(fail[fail.family == f]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1, 1.1]})
    n = [len(fail[fail.family == f]) for f in fams]
    n_or = [fail[fail.family == f].oracle_recovers.sum() for f in fams]
    a1.barh(range(len(fams)), n, color="#bbb", label="gated failures")
    a1.barh(range(len(fams)), n_or, color=C_ORACLE, label="… rescued by centred oracle")
    a1.set_yticks(range(len(fams))); a1.set_yticklabels([pretty[f] for f in fams]); a1.invert_yaxis()
    a1.set_xlabel("cells"); a1.set_title(f"{len(fail)} gated failures; detector fired in 100%")
    a1.legend(frameon=False, fontsize=6.3, loc="lower right")
    # panel 2: gated vs oracle recovery per family — shows the extent effect
    fams2 = sorted(df.family.unique())
    g = [df[df.family == f].gated_recovers.mean() for f in fams2]
    o = [df[df.family == f].oracle_recovers.mean() for f in fams2]
    a2.scatter(o, g, c=[C_GATED if f in heldout else C_CLEAN for f in fams2], s=28, zorder=3)
    for f, xo, yg in zip(fams2, o, g):
        off={"sine_gaussian":(-4,-9),"glitch_train":(4,4),"double_blip":(4,-3),"narrowband_tone":(-30,5),"ringing":(4,-8),"broadband_burst":(4,-3),"whistle":(4,-3),"scattered_light":(4,-3)}
        a2.annotate(pretty[f], (xo, yg), textcoords="offset points", xytext=off[f], fontsize=6)
    a2.plot([0, 1], [0, 1], "--", color="#999", lw=0.7)
    a2.set_xlabel("centred fixed-width oracle recovery"); a2.set_ylabel("detector-driven gate recovery")
    a2.set_xlim(0, 1.05); a2.set_ylim(0, 1.05)
    a2.text(0.03, 0.95, "above line: data-driven\nsupport beats fixed ±0.4 s", fontsize=6.3, va="top")
    a2.set_title("Extent, not localization, limits recovery")
    save(fig, "fig7_oracle_gap")

    # S1: glitch duration vs oracle outcome for the three long-duration families
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for f, mk in [("narrowband_tone", "o"), ("whistle", "s"), ("ringing", "^")]:
        d = df[df.family == f]
        dur = []
        for p in d.params_json:
            q = json.loads(p)
            dur.append(q.get("duration", 3 * q.get("decay", np.nan)))  # 3τ ≈ visible ring-down
        dur = np.array(dur)
        ok = d.oracle_recovers.values
        ax.scatter(dur[ok], np.full(ok.sum(), 1) + np.random.uniform(-.08, .08, ok.sum()), marker=mk, s=16,
                   color=C_GATED, label=f"{pretty[f]} (oracle ok)" if f == "narrowband_tone" else None)
        ax.scatter(dur[~ok], np.full((~ok).sum(), 0) + np.random.uniform(-.08, .08, (~ok).sum()), marker=mk, s=16,
                   color=C_POISON, label=f"{pretty[f]} (oracle fails)" if f == "narrowband_tone" else None)
    ax.axvline(0.8, color="#999", ls="--", lw=0.8); ax.text(0.82, 0.5, "oracle width (2×0.4 s)", fontsize=6, color="#666")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["oracle fails", "oracle recovers"])
    ax.set_xlabel("glitch duration [s]  (ringing: 3τ)")
    ax.set_title("Centred oracle fails when the glitch outlasts its window")
    ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0,0.5))
    save(fig, "figS1_oracle_extent")

fig1(); fig4(); fig5(); fig6(); fig7()
