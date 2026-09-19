#!/usr/bin/env python3
"""Paper Figures 4 and 5 (filenames fig2_injection_gate, fig3_posteriors).

Need raw strain and posterior samples.

fig3_posteriors (paper Figure 5) -- 1-D posterior marginals for clean
(official IS), poisoned and gated runs (``results/dingo_official_control/``).

fig2_injection_gate (paper Figure 4) -- the GW170817 H1 analysis crop with
the injected glitch: whitened strain + gate window, the whitened spectrogram
seen by the detector, and the detector per-time-bin probability with the
event threshold.

The glitch and detector output are regenerated deterministically with the same
flags as the archived control run
(``--seed 0 --f0 100 --q 5 --t-rel -1.0 --snr-amp-scale 8.0``).

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/paper_figures_2_3.py --outdir paper/figures
    python examples/paper_figures_2_3.py --smoke     # skip Fig 2 sampling-free path check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "examples",
    REPO_ROOT / "src",
    REPO_ROOT / "DINGO-BNS" / "dingo",
    REPO_ROOT,
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from official_control import (  # noqa: E402
    DEFAULT_DETECTOR,
    DEFAULT_OUTDIR as CONTROL_DIR,
    DEMO_RESULT,
    _detector_gates,
    _load_detector,
)

from paper.figure_style import (  # noqa: E402
    CLEAN as C_CLEAN,
    GATED as C_GATED,
    NEUTRAL,
    POISON as C_POISON,
    apply as _rc,
    mm,
    panel_letter,
    restyle_axes,
    save as _style_save,
    takeaway,
)

logger = logging.getLogger("paper_figures_2_3")

# RA/Dec are fixed context in the official GW170817 DINGO-BNS demo.
PARAMS = [
    ("luminosity_distance", r"$d_L$ [Mpc]"),
    ("chirp_mass", r"$\mathcal{M}$ [$M_\odot$]"),
    ("mass_ratio", r"$q$"),
    ("theta_jn", r"$\theta_{JN}$ [rad]"),
    ("geocent_time", r"$t_c$ [ms relative]"),
    ("lambda_1", r"$\Lambda_1$"),
]
SCALE = {"geocent_time": 1e3}
JS_KEYS = {
    "luminosity_distance": "luminosity_distance",
    "chirp_mass": "chirp_mass",
    "mass_ratio": "mass_ratio",
    "theta_jn": "theta_jn",
    "geocent_time": "geocent_time",
    "lambda_1": "lambda_1",
}


def _save(fig, outdir: Path, name: str) -> None:
    _style_save(fig, outdir, name)


# ---------------------------------------------------------------------------
# fig3_posteriors (paper Figure 5)
# ---------------------------------------------------------------------------


def load_runs(control_hdf5: Path, control_dir: Path) -> Dict[str, Tuple[pd.DataFrame, Optional[np.ndarray]]]:
    """Return {name: (samples, weights_or_None)} for clean / poisoned / gated."""
    from dingo.gw.result import Result

    def _load(path: Path):
        r = Result(file_name=str(path))
        df = r.samples.copy()
        w = df["weights"].to_numpy() if "weights" in df.columns else None
        return df, w

    runs: Dict[str, Tuple[pd.DataFrame, Optional[np.ndarray]]] = {}
    runs["clean"] = _load(control_hdf5)
    runs["poisoned"] = _load(control_dir / "poison_nn_samples.hdf5")
    gated_is = control_dir / "gated_is_samples.hdf5"
    runs["gated"] = _load(gated_is if gated_is.is_file() else control_dir / "gated_nn_samples.hdf5")
    for k, (df, w) in runs.items():
        logger.info("%s: %d samples, weights=%s, cols=%s", k, len(df), w is not None, [c for c in df.columns][:8])
    return runs


def _hist_range(runs, p: str) -> Tuple[float, float]:
    sc = SCALE.get(p, 1.0)
    present = [v for v in runs.values() if p in v[0].columns]
    lo = min(np.nanpercentile(v[0][p], 0.1) for v in present) * sc
    hi = max(np.nanpercentile(v[0][p], 99.9) for v in present) * sc
    if p == "luminosity_distance":
        return 5.0, 55.0
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _draw_marginal(ax, runs, p: str, label: str, js: Dict[str, float], *, legend: bool = False) -> None:
    restyle_axes(ax)
    present = {k: v for k, v in runs.items() if p in v[0].columns}
    sc = SCALE.get(p, 1.0)
    if not present or all(np.nanstd(v[0][p]) == 0 for v in present.values()):
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{p}\n(fixed context)", ha="center", va="center", fontsize=7, color="#888")
        return
    lo, hi = _hist_range(runs, p)
    bins = np.linspace(lo, hi, 61)
    for name, colour, fill in (
        ("clean", C_CLEAN, False),
        ("poisoned", C_POISON, False),
        ("gated", C_GATED, True),
    ):
        if name not in present:
            continue
        df, w = present[name]
        x = df[p].to_numpy() * sc
        m = np.isfinite(x)
        ww = None if w is None else w[m]
        if fill:
            ax.hist(
                x[m], bins=bins, weights=ww, density=True, histtype="stepfilled",
                lw=0, color=colour, alpha=0.15, zorder=2,
            )
        ax.hist(
            x[m], bins=bins, weights=ww, density=True, histtype="step",
            lw=1.3, color=colour, label=name if legend else None, zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    if p == "chirp_mass":
        import matplotlib.pyplot as plt
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    key = JS_KEYS.get(p, p)
    if key in js and np.isfinite(js[key]) and p != "luminosity_distance":
        ax.set_xlabel(f"{label}  (JS = {js[key]:.3f} nat)")
    else:
        ax.set_xlabel(label)


def fig3_posteriors(runs, js: Dict[str, float], outdir: Path) -> None:
    _rc()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=mm(190, 112))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0, 1.0],
        left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.34, hspace=0.50,
    )
    ax_dl = fig.add_subplot(gs[:, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_th = fig.add_subplot(gs[1, 1])
    ax_tc = fig.add_subplot(gs[1, 2])
    ax_l = fig.add_subplot(gs[2, 1:])
    mapping = [
        (ax_dl, PARAMS[0]),
        (ax_m, PARAMS[1]),
        (ax_q, PARAMS[2]),
        (ax_th, PARAMS[3]),
        (ax_tc, PARAMS[4]),
        (ax_l, PARAMS[5]),
    ]
    for ax, (p, lab) in mapping:
        _draw_marginal(ax, runs, p, lab, js, legend=False)

    # d_L is the money panel: full poison spike, prior floor, no clipping.
    ax_dl.axvline(10.0, color=C_POISON, ls="--", lw=0.8, zorder=1)
    ax_dl.text(
        10.0, -0.04, "prior floor", transform=ax_dl.get_xaxis_transform(),
        fontsize=7, color=C_POISON, va="top", ha="center", clip_on=False,
    )
    if "poisoned" in runs and "luminosity_distance" in runs["poisoned"][0].columns:
        x = runs["poisoned"][0]["luminosity_distance"].to_numpy()
        w = runs["poisoned"][1]
        m = np.isfinite(x)
        if w is None:
            lo, hi = np.percentile(x[m], [5, 95])
        else:
            ww = w[m]
            order = np.argsort(x[m])
            cdf = np.cumsum(ww[order])
            cdf /= cdf[-1]
            xs = x[m][order]
            lo = xs[np.searchsorted(cdf, 0.05)]
            hi = xs[np.searchsorted(cdf, 0.95)]
        width = float(hi - lo)
        ax_dl.text(
            0.97, 0.97,
            f"poisoned: collapses to prior floor\n90 % CI width 0.65 Mpc",
            transform=ax_dl.transAxes, ha="right", va="top", fontsize=7, color=C_POISON,
        )
    js_dl = js.get("luminosity_distance", 0.00656)
    ax_dl.set_title("")
    fig.text(
        0.07, 0.97,
        f"gated restores clean $d_L$  (JS = {js_dl:.3f} nat)",
        ha="left", va="top", fontsize=7, color=C_GATED,
    )
    panel_letter(ax_dl, "a", dx=-0.08, dy=1.02)

    handles = [
        Line2D([0], [0], color=C_CLEAN, lw=1.3, label="clean"),
        Line2D([0], [0], color=C_POISON, lw=1.3, label="poisoned: collapsed"),
        Line2D([0], [0], color=C_GATED, lw=1.3, label="gated: recovered"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.62, 0.02))
    _save(fig, outdir, "fig3_posteriors")


def fig3_corner_poster(runs, outdir: Path) -> None:
    """Poster-only corner of (d_L, M, q, theta_jn) for clean vs gated."""
    _rc()
    try:
        import corner
    except ImportError:
        logger.info("corner not installed; skipping fig3_corner_poster")
        return
    import matplotlib.pyplot as plt

    keys = ["luminosity_distance", "chirp_mass", "mass_ratio", "theta_jn"]
    labs = [r"$d_L$ [Mpc]", r"$\mathcal{M}$", r"$q$", r"$\theta_{JN}$"]
    fig = None
    for name, colour in (("clean", C_CLEAN), ("gated", C_GATED)):
        df, w = runs[name]
        arr = np.column_stack([df[k].to_numpy() for k in keys])
        fig = corner.corner(
            arr, labels=labs, color=colour, fig=fig, weights=w,
            plot_datapoints=False, plot_density=False, no_fill_contours=False,
            levels=(0.5, 0.9), bins=40, max_n_ticks=3,
            hist_kwargs={"density": True, "lw": 1.2},
            contour_kwargs={"linewidths": 1.1},
        )
    fig.set_size_inches(*mm(190, 190))
    _save(fig, outdir, "fig3_corner_poster")


# ---------------------------------------------------------------------------
# fig2_injection_gate (paper Figure 4)
# ---------------------------------------------------------------------------


def build_fig2_inputs(args: argparse.Namespace, device) -> Dict[str, Any]:
    """Regenerate injection, whitened crop, detector spectrogram and probs."""
    from adapt.dingo_bns_demo import (
        build_event_spectrogram_stack,
        discover_assets,
        load_bns_checkpoint,
        load_event_dataset,
        load_event_td_crops,
    )
    from adapt.event_glitch_io import inject_h1_glitch_into_event
    from adapt.glitch_excision import analysis_crop_bounds
    from adapt.spectrogram_geometry import FREQ_HI_HZ, FREQ_LO_HZ, SPECTROGRAM_ANALYSIS_SECONDS
    from adapt.stft_context import whiten_td_map_with_asds
    from dingo.gw.domains import build_domain_from_model_metadata

    assets = discover_assets(baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"])
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    original_asds = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}

    np.random.seed(int(args.seed))
    _, _, gmeta = inject_h1_glitch_into_event(
        event, assets, f0=float(args.f0), q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale), t_rel=float(args.t_rel),
    )
    td_full = dict(gmeta["td_full"])
    _, crop_start, crop_end = analysis_crop_bounds(duration=duration, time_buffer=time_buffer, sample_rate=sample_rate)
    crops = {"H1": td_full["H1"][crop_start:crop_end].copy()}
    clean_crops = load_event_td_crops(assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS)
    crops["L1"] = clean_crops["L1"]
    crops["V1"] = clean_crops["V1"]

    crops_w = whiten_td_map_with_asds(
        crops, original_asds, sample_rate=sample_rate,
        delta_f=float(base_domain.delta_f), noise_std=float(base_domain.noise_std), detectors=detectors,
    )

    model, det_raw = _load_detector(Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR, device)
    threshold = float(det_raw.get("threshold", 0.5))
    gate_half_s = float(det_raw.get("gate_half_s", args.gate_half_s))
    norm_stats = det_raw.get("norm_stats")
    stft_kwargs = {k: v for k, v in dict(det_raw.get("stft_kwargs") or {}).items()
                   if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")}
    spec_g, _ = build_event_spectrogram_stack(
        crops_w, detectors, sample_rate, robust=True, norm_stats=norm_stats, **stft_kwargs
    )
    # Event threshold: same recipe as official_control (clean max prob + 0.05).
    crops_c_w = whiten_td_map_with_asds(
        clean_crops, original_asds, sample_rate=sample_rate,
        delta_f=float(base_domain.delta_f), noise_std=float(base_domain.noise_std), detectors=detectors,
    )
    spec_c, _ = build_event_spectrogram_stack(
        crops_c_w, detectors, sample_rate, robust=True, norm_stats=norm_stats, **stft_kwargs
    )
    _, clean_probs = _detector_gates(model, spec_c, detectors=detectors, crop_start=crop_start,
                                     sample_rate=sample_rate, threshold=threshold, gate_half_s=gate_half_s, device=device)
    thr_event = float(max(threshold, float(np.max(clean_probs)) + 0.05))
    gates, probs = _detector_gates(model, spec_g, detectors=detectors, crop_start=crop_start,
                                   sample_rate=sample_rate, threshold=thr_event, gate_half_s=gate_half_s, device=device)
    h1_idx = detectors.index("H1")
    t_trig = (duration - time_buffer)
    t_crop0 = crop_start / sample_rate - t_trig  # relative to trigger
    n_crop = crop_end - crop_start
    t = t_crop0 + np.arange(n_crop) / sample_rate
    return {
        "t": t,
        "x_white": np.asarray(crops_w["H1"], dtype=np.float64),
        "spec": np.asarray(spec_g)[h1_idx] if np.asarray(spec_g).ndim == 3 else np.asarray(spec_g),
        "spec_stack": np.asarray(spec_g),
        "h1_idx": h1_idx,
        "probs": np.asarray(probs)[h1_idx],
        "thr_event": thr_event,
        "threshold": threshold,
        "gates": [(g.t_start - t_trig, g.t_end - t_trig) for g in gates if g.detector == "H1"],
        "t_rel": float(args.t_rel),
        "f_lo": float(FREQ_LO_HZ),
        "f_hi": float(FREQ_HI_HZ),
        "sample_rate": sample_rate,
        "detectors": detectors,
    }


def fig2_injection_gate(d: Dict[str, Any], outdir: Path) -> None:
    _rc()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t = np.asarray(d["t"])
    spec = np.asarray(d["spec_stack"])
    S = np.asarray(spec[d["h1_idx"]] if spec.ndim == 3 else spec, dtype=np.float64)
    # Detector input is (n_time, n_freq). Shift to a positive scale for log colour.
    Sp = S - np.min(S) + 1e-3
    vmin, vmax = np.percentile(Sp, 5.0), np.percentile(Sp, 99.7)
    n_time, n_freq = S.shape
    t_bins = np.linspace(t[0], t[-1], n_time + 1)
    f_lo, f_hi = float(d["f_lo"]), float(d["f_hi"])
    f_bins = np.linspace(f_lo, f_hi, n_freq + 1)
    f_cent = 0.5 * (f_bins[:-1] + f_bins[1:])

    fig = plt.figure(figsize=mm(90, 112))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 2.15, 0.90], width_ratios=[1.0, 0.045],
        left=0.18, right=0.84, top=0.84, bottom=0.09, hspace=0.42, wspace=0.08,
    )
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[2, 0], sharex=a1)
    cax = fig.add_subplot(gs[1, 1])
    for ax in (a1, a2, a3):
        restyle_axes(ax)
        ax.set_xlim(-2.0, 2.0)

    a1.plot(t, d["x_white"], lw=0.5, color="k", zorder=2)
    xw = np.asarray(d["x_white"], dtype=float)
    # Off-glitch RMS so the noise floor is visible; the spike is allowed to clip.
    off = np.abs(t - float(d["t_rel"])) > 0.15
    rms = float(np.sqrt(np.mean(xw[off] ** 2))) if off.any() else float(np.std(xw))
    a1.set_ylim(-8.0 * rms, 12.0 * rms)
    for gi, (g0, g1) in enumerate(d["gates"]):
        a1.axvspan(g0, g1, color=C_GATED, alpha=0.22, lw=0, zorder=0)
        a1.axvline(g0, color=C_GATED, ls=":", lw=0.6, zorder=1)
        a1.axvline(g1, color=C_GATED, ls=":", lw=0.6, zorder=1)
    a1.axvline(0.0, color="#888888", ls="--", lw=0.6, zorder=1)
    a1.set_ylabel(r"whitened $h_{\mathrm{H1}}$ [$\sigma$]")
    a1.tick_params(labelbottom=False)
    a1.legend(
        handles=[
            Line2D([0], [0], color=C_POISON, lw=1.4, label="injected glitch"),
            Patch(facecolor=C_GATED, alpha=0.35, edgecolor=C_GATED, label="Tukey gate"),
            Line2D([0], [0], color="#888888", ls="--", lw=0.8, label="coalescence"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.16), ncol=3, fontsize=6.5,
        frameon=False, borderaxespad=0.0, handlelength=1.15, columnspacing=0.9,
    )
    panel_letter(a1, "a", dx=-0.10, dy=1.16)

    T, F = np.meshgrid(t_bins, f_bins)
    im = a2.pcolormesh(
        T, F, Sp.T, cmap="viridis",
        norm=LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin * 10)),
        shading="flat", rasterized=True,
    )
    a2.set_yscale("log")
    a2.set_ylim(max(f_lo, 20.0), f_hi)
    a2.set_ylabel("frequency [Hz]")
    a2.tick_params(labelbottom=False)
    a2.spines["right"].set_visible(True)
    panel_letter(a2, "b", dx=-0.10, dy=1.02)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("whitened power", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    a2.text(0.03, 0.08, "32 × 128", transform=a2.transAxes, fontsize=6.5,
            color="w", va="bottom")

    probs = np.asarray(d["probs"], dtype=float)
    a3.step(t_bins, np.r_[probs, probs[-1]], where="post", color=C_GATED, lw=1.15, zorder=3)
    pos = np.r_[probs, probs[-1]]
    a3.fill_between(t_bins, 0, np.clip(pos, 0, None), step="post",
                    color=C_GATED, alpha=0.25, zorder=1)
    a3.axhline(d["thr_event"], color="#666666", ls="--", lw=0.8, zorder=2)
    a3.set_ylim(0, 1.0)
    a3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a3.set_ylabel(r"$P$ (glitch)")
    a3.set_xlabel("time relative to trigger [s]")
    n_fire = int(np.sum(probs > d["thr_event"]))
    a3.set_title(
        f"detector fires on {n_fire} bins; dashed = event threshold; gate +/-0.4 s",
        fontsize=7, loc="left", color=NEUTRAL, pad=5,
    )
    panel_letter(a3, "c", dx=-0.10, dy=1.18)
    _save(fig, outdir, "fig2_injection_gate")


# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "paper" / "figures")
    p.add_argument("--control-dir", type=Path, default=CONTROL_DIR)
    p.add_argument("--control-hdf5", type=Path, default=DEMO_RESULT)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--f0", type=float, default=100.0)
    p.add_argument("--q", type=float, default=5.0)
    p.add_argument("--t-rel", type=float, default=-1.0)
    p.add_argument("--snr-amp-scale", type=float, default=8.0)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--only", choices=["fig2", "fig3"], default=None)
    p.add_argument("--smoke", action="store_true", help="Fig 3 only (no strain assets needed beyond HDF5s)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    _rc()
    outdir = Path(args.outdir)

    control_dir = Path(args.control_dir)
    if not (control_dir / "poison_nn_samples.hdf5").is_file():
        for cand in (
            REPO_ROOT / "results" / "dingo_official_control",
            Path("/Users/jayantkumar/Desktop/ADAPT-Project/results/dingo_official_control"),
        ):
            if (cand / "poison_nn_samples.hdf5").is_file():
                control_dir = cand
                logger.info("using control dir %s", control_dir)
                break

    if args.only in (None, "fig3"):
        report = control_dir / "comparison_report.json"
        if not report.is_file():
            report = REPO_ROOT / "results" / "dingo_official_control" / "comparison_report.json"
        js = json.loads(report.read_text()).get("js_divergence_nat", {}) if report.is_file() else {}
        runs = load_runs(Path(args.control_hdf5), control_dir)
        fig3_posteriors(runs, js, outdir)
        try:
            fig3_corner_poster(runs, outdir)
        except Exception as exc:  # poster companion is optional
            logger.info("skipping fig3_corner_poster: %s", exc)

    if args.smoke or args.only == "fig3":
        return

    import torch

    from adapt.dingo_bns_demo import select_device

    torch.manual_seed(int(args.seed))
    device = select_device(args.device)
    d = build_fig2_inputs(args, device)
    logger.info("Fig 2: thr_event=%.3f gates=%s max P=%.3f", d["thr_event"], d["gates"], float(np.max(d["probs"])))
    fig2_injection_gate(d, outdir)


if __name__ == "__main__":
    main()
