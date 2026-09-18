#!/usr/bin/env python3
"""Paper Figures 2 and 3 (need raw strain and posterior samples).

Figure 3 -- 1-D posterior marginals for clean (official IS), poisoned and
gated runs (``results/dingo_official_control/``).

Figure 2 -- the GW170817 H1 analysis crop with the injected glitch: whitened
strain + gate window, the whitened spectrogram seen by the detector, and the
detector per-time-bin probability with the event threshold.

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

logger = logging.getLogger("paper_figures_2_3")

C_CLEAN, C_POISON, C_GATED = "#1f77b4", "#c0392b", "#2a9d8f"
# RA/Dec are *fixed context* in the official GW170817 DINGO-BNS demo (the
# network is conditioned on the known sky position), so they carry no
# posterior information. The sky panels requested in the brief are therefore
# replaced by the two remaining core inferred parameters: geocentric time and
# the primary tidal deformability.
PARAMS = [
    ("luminosity_distance", r"$d_L$ [Mpc]"),
    ("chirp_mass", r"$\mathcal{M}$ [$M_\odot$]"),
    ("mass_ratio", r"$q$"),
    ("theta_jn", r"$\theta_{JN}$ [rad]"),
    ("geocent_time", r"$t_c - t_\mathrm{trig}$ [ms]"),
    ("lambda_1", r"$\Lambda_1$"),
]
SCALE = {"geocent_time": 1e3}


def _rc() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=300)
    logger.info("wrote %s/%s.{pdf,png}", outdir, name)


# ---------------------------------------------------------------------------
# Figure 3
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


def fig3_posteriors(runs, js: Dict[str, float], outdir: Path) -> None:
    _rc()  # dingo imports reset rcParams; re-apply
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for ax, (p, label) in zip(axes.ravel(), PARAMS):
        present = {k: v for k, v in runs.items() if p in v[0].columns}
        sc = SCALE.get(p, 1.0)
        if not present or all(np.nanstd(v[0][p]) == 0 for v in present.values()):
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{p}\n(fixed context)", ha="center", va="center", fontsize=7, color="#888")
            continue
        lo = min(np.nanpercentile(v[0][p], 0.1) for v in present.values()) * sc
        hi = max(np.nanpercentile(v[0][p], 99.9) for v in present.values()) * sc
        if p == "luminosity_distance":
            lo = min(lo, 5.0)
        bins = np.linspace(lo, hi, 61)
        ymax_ref = 0.0
        for name, colour in (("clean", C_CLEAN), ("poisoned", C_POISON), ("gated", C_GATED)):
            if name not in present:
                continue
            df, w = present[name]
            x = df[p].to_numpy() * sc
            m = np.isfinite(x)
            n, _, _ = ax.hist(x[m], bins=bins, weights=None if w is None else w[m], density=True,
                              histtype="step", lw=1.2, color=colour, label=name)
            if name in ("clean", "gated"):
                ymax_ref = max(ymax_ref, float(np.max(n)))
        if p == "luminosity_distance" and ymax_ref > 0:
            # The poisoned run piles up at the 10 Mpc prior floor; clip that
            # spike so the clean/gated overlap remains legible.
            ax.set_ylim(0, 2.2 * ymax_ref)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if p == "chirp_mass":
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            for lab in ax.get_xticklabels():
                lab.set_fontsize(6.3)
        if p in js and np.isfinite(js[p]):
            ax.text(0.98, 0.95, f"JS = {js[p]:.4f} nat", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#444")
        if p == "luminosity_distance":
            ax.axvline(10.0, color=C_POISON, ls=":", lw=0.8)
            ax.text(0.16, 0.90, "poisoned: saturates at\n10 Mpc prior floor (clipped)", transform=ax.transAxes,
                    fontsize=6.3, color=C_POISON, va="top")
    axes[0, 2].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    _save(fig, outdir, "fig3_posteriors")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2
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
    _rc()  # dingo imports reset rcParams; re-apply
    import matplotlib.pyplot as plt

    t = d["t"]
    spec = np.asarray(d["spec_stack"])
    # Detector input stack is (C, T, F) with the first n_det channels being the
    # per-detector robust-normalised log-power spectrograms (32 x 128).
    S = np.asarray(spec[d["h1_idx"]] if spec.ndim == 3 else spec, dtype=np.float64)  # (T, F)
    vmin, vmax = np.percentile(S, 1.0), np.percentile(S, 99.7)
    n_time = S.shape[0]
    t_bins = np.linspace(t[0], t[-1], n_time + 1)

    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(3.5, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.7, 0.7], "hspace": 0.12}
    )
    a1.plot(t, d["x_white"], lw=0.45, color="k")
    for (g0, g1) in d["gates"]:
        a1.axvspan(g0, g1, color=C_GATED, alpha=0.25, lw=0)
        a3.axvspan(g0, g1, color=C_GATED, alpha=0.25, lw=0)
    a1.axvline(d["t_rel"], color=C_POISON, ls=":", lw=0.8)
    a1.set_ylabel(r"whitened $h_{\rm H1}$ [$\sigma$]")
    a1.text(0.02, 0.92, "(a)", transform=a1.transAxes, fontsize=8, va="top")

    # Frequency grid of the detector input is linear between f_lo and f_hi.
    im = a2.imshow(S.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[t[0], t[-1], d["f_lo"], d["f_hi"]])
    a2.set_ylabel("frequency [Hz]")
    a2.text(0.02, 0.92, "(b)", transform=a2.transAxes, fontsize=8, va="top", color="w")
    cb = fig.colorbar(im, ax=a2, pad=0.02, fraction=0.05)
    cb.set_label("normalised log-power\n(detector input)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    a3.step(t_bins, np.r_[d["probs"], d["probs"][-1]], where="post", color=C_GATED, lw=1.1)
    a3.axhline(d["thr_event"], color="#666", ls="--", lw=0.8)
    a3.text(t[-1], d["thr_event"] + 0.03, f"threshold {d['thr_event']:.2f}", ha="right", va="bottom", fontsize=6.3, color="#666")
    a3.set_ylim(0, 1.05)
    a3.set_ylabel("P(glitch)")
    a3.set_xlabel("time relative to trigger [s]")
    a3.text(0.02, 0.92, "(c)", transform=a3.transAxes, fontsize=8, va="top")
    for ax in (a1, a2, a3):
        ax.set_xlim(t[0], t[-1])
    _save(fig, outdir, "fig2_injection_gate")
    plt.close(fig)


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

    if args.only in (None, "fig3"):
        report = Path(args.control_dir) / "comparison_report.json"
        js = json.loads(report.read_text()).get("js_divergence_nat", {}) if report.is_file() else {}
        runs = load_runs(Path(args.control_hdf5), Path(args.control_dir))
        fig3_posteriors(runs, js, outdir)

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
