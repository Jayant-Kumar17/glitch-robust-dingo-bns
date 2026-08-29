#!/usr/bin/env python3
"""Official DINGO-BNS clean control vs detect-and-gate on glitchy data.

Paper-faithful comparison (arXiv:2407.09602):
  - Control: official dingo_pipe importance-sampled GW170817 posterior (20k)
  - Treatment: same pretrained model on H1-glitch + detect-and-gate + orig ASD,
    sampled at N=20k then importance-sampled with the same IS settings as the demo
  - Poison: glitch + Welch (must still collapse)

Writes ``results/dingo_official_control/``:
  control_summary.json, gated_glitchy_summary.json, poison_summary.json,
  comparison_report.json, corner PDF.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
    python scripts/compare_official_vs_gated.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "src",
    REPO_ROOT / "DINGO-BNS" / "dingo",
    REPO_ROOT,
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEMO_RESULT = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "inference-dingo-pipe"
    / "outdir"
    / "result"
    / "GW170817_data0_1187008882-42_importance_sampling.hdf5"
)
DEMO_SAMPLING = DEMO_RESULT.parent / "GW170817_data0_1187008882-42_sampling.hdf5"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "dingo_official_control"
DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)

# Demo INI: all sampled + derived chirp_mass. Fixed sky/proxy are reported but not scored.
INFERENCE_PARAMS = [
    "delta_chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "psi",
    "lambda_1",
    "lambda_2",
    "chirp_mass",
]
CORE_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "theta_jn",
    "geocent_time",
]
JS_THRESHOLD = 0.02  # nat; paper-scale 5e-4 flagged separately
PAPER_JS = 5e-4
DL_MED_TOL_MPC = 5.0

logger = logging.getLogger("compare_official_vs_gated")


# ---------------------------------------------------------------------------
# Summaries / metrics
# ---------------------------------------------------------------------------


def _weighted_quantiles(
    x: np.ndarray, w: Optional[np.ndarray], qs: Sequence[float]
) -> List[float]:
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    x = x[mask]
    if w is None:
        return [float(np.quantile(x, q)) for q in qs]
    w = np.asarray(w, dtype=np.float64)[mask]
    w = np.maximum(w, 0.0)
    if not np.any(w > 0) or x.size == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(x)
    xw, ww = x[order], w[order]
    cdf = np.cumsum(ww)
    cdf /= cdf[-1]
    return [float(np.interp(q, cdf, xw)) for q in qs]


def param_summary(
    samples: pd.DataFrame,
    params: Sequence[str],
    *,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    w = weights
    if w is None and "weights" in samples.columns:
        w = samples["weights"].to_numpy()
    for p in params:
        if p not in samples.columns:
            continue
        lo, med, hi = _weighted_quantiles(samples[p].to_numpy(), w, [0.05, 0.5, 0.95])
        out[p] = {"lo": lo, "med": med, "hi": hi, "n": int(len(samples))}
    return out


def js_divergence_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    wx: Optional[np.ndarray] = None,
    wy: Optional[np.ndarray] = None,
    bins: int = 64,
) -> float:
    """1-D Jensen–Shannon divergence in nats between two weighted samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx = np.isfinite(x)
    my = np.isfinite(y)
    x, y = x[mx], y[my]
    if wx is not None:
        wx = np.asarray(wx, dtype=np.float64)[mx]
        wx = np.maximum(wx, 0.0)
        wx = wx / (wx.sum() + 1e-30)
    if wy is not None:
        wy = np.asarray(wy, dtype=np.float64)[my]
        wy = np.maximum(wy, 0.0)
        wy = wy / (wy.sum() + 1e-30)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges, weights=wx, density=True)
    py, _ = np.histogram(y, bins=edges, weights=wy, density=True)
    px = px + 1e-12
    py = py + 1e-12
    px = px / px.sum()
    py = py / py.sum()
    m = 0.5 * (px + py)
    kl_pm = float(np.sum(px * np.log(px / m)))
    kl_qm = float(np.sum(py * np.log(py / m)))
    return 0.5 * (kl_pm + kl_qm)


def ci_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


# ---------------------------------------------------------------------------
# Control dump
# ---------------------------------------------------------------------------


def dump_control_summary(path: Path) -> Dict[str, Any]:
    from dingo.gw.result import Result

    if not path.is_file():
        raise FileNotFoundError(
            f"Official IS result missing: {path}. Run dingo_pipe GW170817.ini first."
        )
    result = Result(file_name=str(path))
    summary = {
        "source": str(path),
        "n_samples": int(result.num_samples),
        "n_eff": float(result.n_eff) if result.n_eff is not None else None,
        "sample_efficiency": float(result.sample_efficiency)
        if result.sample_efficiency is not None
        else None,
        "log_evidence": float(result.log_evidence)
        if result.log_evidence is not None
        else None,
        "log_evidence_std": float(result.log_evidence_std)
        if getattr(result, "log_evidence_std", None) is not None
        else None,
        "params": param_summary(result.samples, INFERENCE_PARAMS),
        "has_weights": "weights" in result.samples.columns,
        "pipeline": "official_dingo_pipe_importance_sampling",
    }
    return summary


# ---------------------------------------------------------------------------
# Gated / poison packages + sampling + IS
# ---------------------------------------------------------------------------


def _oracle_h1_gates(*, t_rel: float, duration: float, time_buffer: float, half_s: float):
    from adapt.glitch_excision import GateWindow

    t_peak = (float(duration) - float(time_buffer)) + float(t_rel)
    return [
        GateWindow(
            detector="H1",
            t_start=t_peak - float(half_s),
            t_end=t_peak + float(half_s),
            score=1.0,
        )
    ]


def _load_detector(path: Path, device: torch.device):
    from adapt.models import GlitchDetectorSTFT

    raw = torch.load(path, map_location="cpu", weights_only=False)
    model = GlitchDetectorSTFT(**dict(raw.get("model_kwargs") or {}))
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, raw


def _detector_gates(
    model,
    spectrogram: np.ndarray,
    *,
    detectors: Sequence[str],
    crop_start: int,
    sample_rate: float,
    threshold: float,
    gate_half_s: float,
    device: torch.device,
):
    from adapt.glitch_excision import time_bin_mask_to_windows
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

    spec_t = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_probs(spec_t.to(device)).cpu().numpy()[0]
    gates = []
    t0 = float(crop_start) / float(sample_rate)
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    for di, det in enumerate(detectors):
        if di >= probs.shape[0] or det != "H1":
            continue
        bin_mask = probs[di] > float(threshold)
        if int(bin_mask.sum()) == 0:
            continue
        gates.extend(
            time_bin_mask_to_windows(
                bin_mask,
                n_samples=n_crop,
                sample_rate=sample_rate,
                t0=t0,
                detector=det,
                pad_s=float(gate_half_s),
                scores=probs[di],
            )
        )
    return gates, probs


def build_glitchy_packages(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return (assets, event, poison_data, gated_data, gate_meta)."""
    from adapt.glitch_excision import analysis_crop_bounds, rebuild_event_from_gated_td
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata
    from event_glitch_io import inject_h1_glitch_into_event
    from evaluate_gw170817_comparison import (
        discover_assets,
        load_event_dataset,
        load_event_td_crops,
    )
    from train_bns_spectrogram import load_bns_checkpoint

    pe = None
    for cand in (
        REPO_ROOT
        / "DINGO-BNS"
        / "dingo"
        / "binary-neutron-star-demo"
        / "GW170817"
        / "downloads"
        / "dingo-bns-model_GW170817.pt",
        REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt",
    ):
        if cand.is_file():
            pe = cand
            break
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=pe,
    )
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))

    poison, _, gmeta = inject_h1_glitch_into_event(
        event,
        assets,
        f0=float(args.f0),
        q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
    )
    td_full = dict(gmeta["td_full"])
    original_asds = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}

    gate_half_s = float(args.gate_half_s)
    gate_meta: Dict[str, Any] = {"gate_half_s": gate_half_s, "mode": "oracle"}
    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR

    if args.oracle_gate or not det_path.is_file():
        gates = _oracle_h1_gates(
            t_rel=float(args.t_rel),
            duration=duration,
            time_buffer=time_buffer,
            half_s=gate_half_s,
        )
        if not det_path.is_file():
            logger.warning("Detector missing at %s — oracle gate", det_path)
    else:
        model, det_raw = _load_detector(det_path, device)
        threshold = float(det_raw.get("threshold", 0.5))
        gate_half_s = float(det_raw.get("gate_half_s", gate_half_s))
        norm_stats = det_raw.get("norm_stats")
        stft_kwargs = dict(det_raw.get("stft_kwargs") or {})
        _, crop_start, crop_end = analysis_crop_bounds(
            duration=duration, time_buffer=time_buffer, sample_rate=sample_rate
        )
        crops_g = {"H1": td_full["H1"][crop_start:crop_end].copy()}
        clean_crops = load_event_td_crops(
            assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
        )
        crops_g["L1"] = clean_crops["L1"]
        crops_g["V1"] = clean_crops["V1"]
        crops_w = whiten_td_map_with_asds(
            crops_g,
            original_asds,
            sample_rate=sample_rate,
            delta_f=float(base_domain.delta_f),
            noise_std=float(base_domain.noise_std),
            detectors=detectors,
        )
        spec_g, _ = build_robust_spectrogram_from_td(
            crops_w,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=norm_stats,
            **{
                k: v
                for k, v in stft_kwargs.items()
                if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
            },
        )
        crops_c_w = whiten_td_map_with_asds(
            clean_crops,
            original_asds,
            sample_rate=sample_rate,
            delta_f=float(base_domain.delta_f),
            noise_std=float(base_domain.noise_std),
            detectors=detectors,
        )
        spec_c, _ = build_robust_spectrogram_from_td(
            crops_c_w,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=norm_stats,
            **{
                k: v
                for k, v in stft_kwargs.items()
                if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
            },
        )
        _, clean_probs = _detector_gates(
            model,
            spec_c,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=threshold,
            gate_half_s=gate_half_s,
            device=device,
        )
        thr_event = float(max(threshold, float(np.max(clean_probs)) + 0.05))
        gates, probs = _detector_gates(
            model,
            spec_g,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=thr_event,
            gate_half_s=gate_half_s,
            device=device,
        )
        gate_meta.update(
            {
                "mode": "learned_detector",
                "detector_ckpt": str(det_path),
                "threshold": threshold,
                "threshold_event": thr_event,
                "n_gates": len(gates),
                "probs_max_H1": float(np.max(probs[0])) if probs.shape[0] else 0.0,
            }
        )
        if not gates:
            logger.warning("Detector silent — falling back to oracle gate")
            gates = _oracle_h1_gates(
                t_rel=float(args.t_rel),
                duration=duration,
                time_buffer=time_buffer,
                half_s=gate_half_s,
            )
            gate_meta["fallback_oracle"] = True

    gate_meta["n_gates"] = len(gates)
    gate_meta["gates"] = [
        {"det": g.detector, "t_start": g.t_start, "t_end": g.t_end, "score": g.score}
        for g in gates
    ]

    excised = rebuild_event_from_gated_td(
        poison,
        td_by_det=td_full,
        gates=gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        original_asds=original_asds,
    )
    gate_meta["residual_power_frac"] = excised.meta.get("residual_power_frac")
    gate_meta["rebuild_mode"] = excised.meta.get("rebuild_mode")

    return {
        "assets": assets,
        "event": event,
        "settings": settings,
        "poison": poison,
        "gated": excised.data,
        "gate_meta": gate_meta,
        "gmeta": {
            k: v
            for k, v in gmeta.items()
            if k not in ("td_full", "td_clean_full")
        },
    }


def sample_to_result(
    baseline_ckpt: Path,
    event_data: Dict[str, Any],
    settings: Dict[str, Any],
    fixed_context: Dict[str, float],
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
):
    """NN sample then export a dingo Result (keeps log_prob for IS)."""
    from dingo.core.models import PosteriorModel
    from dingo.core.samplers import FixedInitSampler
    from dingo.gw.inference.gw_samplers import GWSamplerGNPE

    device_str = str(device)
    try:
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )
    except Exception:
        device_str = "cpu"
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )
    init = FixedInitSampler(fixed_context, log_prob=0.0)
    sampler = GWSamplerGNPE(
        model=pm,
        init_sampler=init,
        num_iterations=1,
        fixed_context_parameters=fixed_context,
    )
    ev = SimpleNamespace(data=event_data, settings=settings)
    sampler.context = ev.data
    sampler.event_metadata = ev.settings
    sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
    return sampler.to_result()


def run_importance_sampling(
    result,
    *,
    num_processes: int = 4,
) -> Any:
    """Match demo INI importance-sampling-settings.

    Note: ``np.std(pandas.Series)`` on a constant float32 ``chirp_mass_proxy``
    can return ~1e-4 (pandas/numpy quirk) and trip DINGO's heterodyning check
    even when every value is identical. Snap the column to a true scalar first.
    """
    if "chirp_mass_proxy" in result.samples.columns:
        proxy = float(np.mean(np.asarray(result.samples["chirp_mass_proxy"].to_numpy())))
        # Assign a plain ndarray column so np.std(Series) is exactly 0
        # (pandas float32 Series can make np.std report ~1e-4 spuriously).
        result.samples["chirp_mass_proxy"] = np.full(
            len(result.samples), proxy, dtype=np.float64
        )
        assert float(np.std(result.samples["chirp_mass_proxy"])) <= 1e-10

    synthetic_phase_kwargs = {
        "approximation_22_mode": True,
        "n_grid": 1000,
        "uniform_weight": 0,
        "compute_likelihood": True,
        "num_processes": int(num_processes),
    }
    likelihood_kwargs = {
        "decimate": True,
        "phase_heterodyning": True,
    }
    logger.info("Sampling synthetic phase (n_grid=1000)…")
    result.sample_synthetic_phase(synthetic_phase_kwargs, likelihood_kwargs)
    logger.info("Importance sampling with %d processes…", num_processes)
    result.importance_sample(num_processes=int(num_processes), **likelihood_kwargs)
    result.print_summary()
    return result


def summarize_result(result, *, label: str, extra: Optional[Dict[str, Any]] = None):
    out: Dict[str, Any] = {
        "label": label,
        "n_samples": int(result.num_samples),
        "n_eff": float(result.n_eff) if result.n_eff is not None else None,
        "sample_efficiency": float(result.sample_efficiency)
        if result.sample_efficiency is not None
        else None,
        "log_evidence": float(result.log_evidence)
        if result.log_evidence is not None
        else None,
        "params": param_summary(result.samples, INFERENCE_PARAMS),
        "has_weights": "weights" in result.samples.columns,
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Comparison + PDF
# ---------------------------------------------------------------------------


def compare_arms(
    control: Dict[str, Any],
    gated: Dict[str, Any],
    poison: Dict[str, Any],
    *,
    control_samples: pd.DataFrame,
    gated_samples: pd.DataFrame,
) -> Dict[str, Any]:
    cw = (
        control_samples["weights"].to_numpy()
        if "weights" in control_samples.columns
        else None
    )
    gw = (
        gated_samples["weights"].to_numpy()
        if "weights" in gated_samples.columns
        else None
    )
    js: Dict[str, float] = {}
    overlaps: Dict[str, bool] = {}
    for p in INFERENCE_PARAMS:
        if p not in control_samples.columns or p not in gated_samples.columns:
            continue
        js[p] = js_divergence_1d(
            control_samples[p].to_numpy(),
            gated_samples[p].to_numpy(),
            wx=cw,
            wy=gw,
        )
        if p in control["params"] and p in gated["params"]:
            overlaps[p] = ci_overlap(control["params"][p], gated["params"][p])

    core_js_ok = all(
        js.get(p, 1.0) < JS_THRESHOLD for p in CORE_PARAMS if p in js
    )
    core_overlap_ok = all(overlaps.get(p, False) for p in CORE_PARAMS if p in overlaps)
    dl_c = control["params"]["luminosity_distance"]
    dl_g = gated["params"]["luminosity_distance"]
    dl_med_ok = abs(dl_g["med"] - dl_c["med"]) <= DL_MED_TOL_MPC
    poison_dl = poison["params"].get("luminosity_distance", {})
    poison_collapsed = bool(poison_dl.get("hi", 100) < 15.0)

    gated_eff = gated.get("sample_efficiency")
    control_eff = control.get("sample_efficiency")
    eff_ok = True
    if gated_eff is not None and control_eff is not None:
        # same order of magnitude as ~10%, not ~0%
        eff_ok = float(gated_eff) > 0.02
    elif gated_eff is not None:
        eff_ok = float(gated_eff) > 0.02

    verdict_a = bool(
        core_js_ok and core_overlap_ok and dl_med_ok and poison_collapsed and eff_ok
    )
    paper_scale = {
        p: bool(js[p] < PAPER_JS) for p in CORE_PARAMS if p in js
    }

    return {
        "verdict": "A" if verdict_a else "B",
        "js_divergence_nat": js,
        "ci_overlap": overlaps,
        "core_js_ok": core_js_ok,
        "core_overlap_ok": core_overlap_ok,
        "dl_med_ok": dl_med_ok,
        "dl_med_delta_mpc": float(dl_g["med"] - dl_c["med"]),
        "poison_collapsed": poison_collapsed,
        "efficiency_ok": eff_ok,
        "js_threshold": JS_THRESHOLD,
        "paper_js_threshold": PAPER_JS,
        "paper_scale_js": paper_scale,
        "criteria": {
            "core_params": CORE_PARAMS,
            "dl_med_tol_mpc": DL_MED_TOL_MPC,
        },
    }


def write_corner_pdf(
    path: Path,
    control_df: pd.DataFrame,
    gated_df: pd.DataFrame,
    poison_df: pd.DataFrame,
    *,
    params: Sequence[str],
) -> None:
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    use = [p for p in params if p in control_df.columns and p in gated_df.columns]
    if len(use) < 2:
        logger.warning("Not enough params for corner PDF")
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    def _draw(df, w=None):
        data = df[use].to_numpy()
        if w is None and "weights" in df.columns:
            w = df["weights"].to_numpy()
        return data, w

    with PdfPages(path) as pdf:
        fig = None
        colors = ["#1f77b4", "#d62728", "#2ca02c"]
        labels = ["clean IS (official)", "gated IS (glitchy)", "poison NN"]
        dfs = [control_df, gated_df, poison_df]
        handles = []
        for color, label, df in zip(colors, labels, dfs):
            if df is None or len(df) == 0:
                continue
            data, w = _draw(df)
            # corner does not take weights in older versions; resample if needed
            if w is not None and np.any(w > 0):
                w = np.maximum(w, 0.0)
                w = w / w.sum()
                idx = np.random.choice(len(data), size=min(5000, len(data)), p=w)
                data = data[idx]
            ranges = []
            for j in range(data.shape[1]):
                col = data[:, j]
                lo, hi = float(np.min(col)), float(np.max(col))
                if hi <= lo:
                    hi = lo + 1e-6
                pad = 0.05 * (hi - lo)
                ranges.append((lo - pad, hi + pad))
            fig = corner.corner(
                data,
                labels=use,
                color=color,
                smooth=1.0,
                plot_datapoints=False,
                plot_density=False,
                plot_contours=True,
                levels=(0.5, 0.9),
                bins=30,
                no_fill_contours=True,
                fig=fig,
                range=ranges,
            )
            handles.append(plt.Line2D([], [], color=color, label=label, linewidth=3))
        if fig is not None:
            fig.legend(handles=handles, loc="upper right", fontsize=10)
            fig.suptitle(
                "Official clean IS vs gated glitchy IS vs poison",
                fontsize=13,
                y=1.02,
            )
            pdf.savefig(fig, dpi=160, bbox_inches="tight")
            plt.close(fig)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Ablation loop (verdict B only)
# ---------------------------------------------------------------------------


def run_gate_width_ablation(
    args: argparse.Namespace,
    *,
    device: torch.device,
    control_summary: Dict[str, Any],
    control_df: pd.DataFrame,
    outdir: Path,
) -> Dict[str, Any]:
    """Empirical gate half-width sweep; NN-only 2k samples for speed."""
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from event_glitch_io import inject_h1_glitch_into_event
    from evaluate_gw170817_comparison import discover_assets, load_event_dataset

    pe = (
        REPO_ROOT
        / "DINGO-BNS"
        / "dingo"
        / "binary-neutron-star-demo"
        / "GW170817"
        / "downloads"
        / "dingo-bns-model_GW170817.pt"
    )
    assets = discover_assets(baseline_ckpt=None, custom_ckpt=pe if pe.is_file() else None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = assets["fixed_context"]
    poison, _, gmeta = inject_h1_glitch_into_event(
        event,
        assets,
        f0=float(args.f0),
        q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
    )
    td_full = dict(gmeta["td_full"])
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    sample_rate = float(gmeta["sample_rate"])
    f_max = float(gmeta["f_max"])
    roll_off = float(gmeta["roll_off"])
    asds = {d: np.asarray(event.data["asds"][d]).copy() for d in event.data["asds"]}
    n = int(args.ablation_samples)
    bs = min(512, n)
    rows = []
    cw = control_df["weights"].to_numpy() if "weights" in control_df.columns else None
    for half in (0.2, 0.3, 0.4, 0.6):
        gates = _oracle_h1_gates(
            t_rel=float(args.t_rel),
            duration=duration,
            time_buffer=time_buffer,
            half_s=half,
        )
        excised = rebuild_event_from_gated_td(
            poison,
            td_by_det=td_full,
            gates=gates,
            sample_rate=sample_rate,
            roll_off=roll_off,
            f_max=f_max,
            original_asds=asds,
        )
        res = sample_to_result(
            Path(assets["baseline_ckpt"]),
            excised.data,
            settings,
            fixed,
            device=device,
            num_samples=n,
            batch_size=bs,
        )
        js_dl = js_divergence_1d(
            control_df["luminosity_distance"].to_numpy(),
            res.samples["luminosity_distance"].to_numpy(),
            wx=cw,
        )
        sm = param_summary(res.samples, ["luminosity_distance", "chirp_mass", "mass_ratio"])
        row = {
            "gate_half_s": half,
            "js_d_L": js_dl,
            "d_L": sm["luminosity_distance"],
            "residual_power_frac": excised.meta.get("residual_power_frac"),
        }
        rows.append(row)
        logger.info("Ablation gate_half=%.2f js_dL=%.4f dL=%s", half, js_dl, sm["luminosity_distance"])
    best = min(rows, key=lambda r: r["js_d_L"])
    out = {"rows": rows, "best_gate_half_s": best["gate_half_s"], "best_js_d_L": best["js_d_L"]}
    with open(outdir / "ablation_gate_width.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from evaluate_gw170817_comparison import select_device

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    logger.info("Device: %s", device)

    # ---- Step 1: official control ----
    logger.info("===== Dumping official dingo_pipe IS control =====")
    control = dump_control_summary(Path(args.control_hdf5))
    with open(outdir / "control_summary.json", "w") as f:
        json.dump(control, f, indent=2)
    logger.info(
        "Control d_L=%s ε=%.2f%% n_eff=%.1f",
        control["params"]["luminosity_distance"],
        100.0 * float(control["sample_efficiency"] or 0),
        float(control["n_eff"] or 0),
    )

    from dingo.gw.result import Result

    control_result = Result(file_name=str(Path(args.control_hdf5)))
    control_df = control_result.samples.copy()

    # ---- Step 2–3: gated + poison ----
    from dingo.gw.result import Result as GwResult

    if args.resume_from_nn:
        logger.info("===== Resume: loading existing NN HDF5s from %s =====", outdir)
        poison_result = GwResult(file_name=str(outdir / "poison_nn_samples.hdf5"))
        poison_summary = summarize_result(
            poison_result, label="poison_glitch_welch_nn", extra={"is_done": False}
        )
        with open(outdir / "poison_summary.json", "w") as f:
            json.dump(poison_summary, f, indent=2)
        poison_df = poison_result.samples.copy()

        gated_result = GwResult(file_name=str(outdir / "gated_nn_samples.hdf5"))
        gate_meta = {}
        gmeta_small: Dict[str, Any] = {}
        if (outdir / "gated_nn_summary.json").is_file():
            prev = json.loads((outdir / "gated_nn_summary.json").read_text())
            gate_meta = prev.get("gate_meta") or {}
        pkg = {"gate_meta": gate_meta, "gmeta": gmeta_small}
    else:
        logger.info("===== Building glitchy packages (poison + gated) =====")
        pkg = build_glitchy_packages(args, device=device)
        assets = pkg["assets"]
        settings = pkg["settings"]
        fixed = assets["fixed_context"]
        ckpt = Path(assets["baseline_ckpt"])

        logger.info(
            "===== Poison NN sampling N=%d (collapse check) =====", args.num_samples
        )
        poison_result = sample_to_result(
            ckpt,
            pkg["poison"],
            settings,
            fixed,
            device=device,
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
        )
        poison_nn_path = outdir / "poison_nn_samples.hdf5"
        poison_result.to_file(file_name=str(poison_nn_path))
        poison_summary = summarize_result(
            poison_result, label="poison_glitch_welch_nn", extra={"is_done": False}
        )
        if args.importance_sample_poison:
            poison_result = run_importance_sampling(
                poison_result, num_processes=int(args.is_cpus)
            )
            poison_result.to_file(file_name=str(outdir / "poison_is_samples.hdf5"))
            poison_summary = summarize_result(
                poison_result, label="poison_glitch_welch_is", extra={"is_done": True}
            )
        with open(outdir / "poison_summary.json", "w") as f:
            json.dump(poison_summary, f, indent=2)
        poison_df = poison_result.samples.copy()

        logger.info(
            "===== Gated NN sampling N=%d (detect-and-gate + orig ASD) =====",
            args.num_samples,
        )
        gated_result = sample_to_result(
            ckpt,
            pkg["gated"],
            settings,
            fixed,
            device=device,
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
        )
        gated_result.to_file(file_name=str(outdir / "gated_nn_samples.hdf5"))
        gated_nn_summary = summarize_result(
            gated_result,
            label="gated_nn",
            extra={"gate_meta": pkg["gate_meta"], "is_done": False},
        )
        with open(outdir / "gated_nn_summary.json", "w") as f:
            json.dump(gated_nn_summary, f, indent=2)

    if args.skip_importance_sampling:
        logger.warning("Skipping IS (--skip-importance-sampling)")
        gated_is_summary = summarize_result(
            gated_result,
            label="gated_nn",
            extra={"gate_meta": pkg.get("gate_meta"), "is_done": False},
        )
        gated_df = gated_result.samples.copy()
    else:
        logger.info("===== Gated importance sampling (demo settings) =====")
        gated_result = run_importance_sampling(
            gated_result, num_processes=int(args.is_cpus)
        )
        gated_result.to_file(file_name=str(outdir / "gated_is_samples.hdf5"))
        gated_is_summary = summarize_result(
            gated_result,
            label="gated_is",
            extra={"gate_meta": pkg.get("gate_meta"), "is_done": True},
        )
        gated_df = gated_result.samples.copy()
    with open(outdir / "gated_glitchy_summary.json", "w") as f:
        json.dump(gated_is_summary, f, indent=2)

    # ---- Compare ----
    logger.info("===== Comparing clean IS vs gated =====")
    comparison = compare_arms(
        control,
        gated_is_summary,
        poison_summary,
        control_samples=control_df,
        gated_samples=gated_df,
    )
    comparison["gate_meta"] = pkg["gate_meta"]
    comparison["glitch_meta"] = {
        k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
        for k, v in pkg["gmeta"].items()
        if not isinstance(v, (np.ndarray, dict))
    }
    comparison["control_efficiency"] = control.get("sample_efficiency")
    comparison["gated_efficiency"] = gated_is_summary.get("sample_efficiency")

    write_corner_pdf(
        outdir / "clean_vs_gated_vs_poison_corner.pdf",
        control_df,
        gated_df,
        poison_df,
        params=["chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn"],
    )

    ablation = None
    if comparison["verdict"] == "B" and not args.skip_ablation:
        logger.info("===== Verdict B → empirical gate-width ablation =====")
        ablation = run_gate_width_ablation(
            args,
            device=device,
            control_summary=control,
            control_df=control_df,
            outdir=outdir,
        )
        comparison["ablation_gate_width"] = ablation
        # If a better half-width exists, note it; do not silently change default.
        comparison["ablation_recommendation"] = {
            "best_gate_half_s": ablation["best_gate_half_s"],
            "best_js_d_L": ablation["best_js_d_L"],
            "note": "Re-run full IS with this half-width if JS improves vs 0.4s NN baseline",
        }
    elif comparison["verdict"] == "A":
        comparison["ablation"] = "skipped_verdict_A"

    with open(outdir / "comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info("VERDICT %s", comparison["verdict"])
    logger.info(
        "d_L control=%s gated=%s poison=%s",
        control["params"]["luminosity_distance"],
        gated_is_summary["params"]["luminosity_distance"],
        poison_summary["params"]["luminosity_distance"],
    )
    logger.info(
        "JS(d_L)=%.4f ε_control=%.2f%% ε_gated=%s",
        comparison["js_divergence_nat"].get("luminosity_distance", float("nan")),
        100.0 * float(control.get("sample_efficiency") or 0),
        None
        if gated_is_summary.get("sample_efficiency") is None
        else f"{100.0 * float(gated_is_summary['sample_efficiency']):.2f}%",
    )
    print(json.dumps({"verdict": comparison["verdict"], "outdir": str(outdir)}, indent=2))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--control-hdf5", type=Path, default=DEMO_RESULT)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--is-cpus", type=int, default=4)
    p.add_argument("--skip-importance-sampling", action="store_true")
    p.add_argument("--importance-sample-poison", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--ablation-samples", type=int, default=2000)
    p.add_argument("--oracle-gate", action="store_true")
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=100.0)
    p.add_argument("--q", type=float, default=5.0)
    p.add_argument("--t-rel", type=float, default=-1.0)
    p.add_argument("--snr-amp-scale", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume-from-nn",
        action="store_true",
        help="Skip NN sampling; load poison_nn_samples.hdf5 and gated_nn_samples.hdf5 from outdir",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
