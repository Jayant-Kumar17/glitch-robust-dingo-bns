#!/usr/bin/env python3
"""GW170817 glitch-robustness: poison baseline vs detect-and-gate excision.

Default comparison (honest path):
  - baseline: H1 glitch FD + on-source Welch ASD → frozen DINGO (expected d_L~10)
  - custom: detect-and-gate on the glitchy package, rebuild H1 FD from gated TD,
    restore original ASDs → frozen DINGO (expected d_L in the clean ~20–45 Mpc band)

The old STFT embedding-repair path is available behind ``--legacy-embedding-repair``.

Usage::

    conda activate adapt_env
    export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=DINGO-BNS/dingo:src
    python scripts/evaluate_glitch_robustness.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal as sp_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "results"
DEFAULT_BEST = REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt"
LEGACY_BEST = REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt"
DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)
GLITCH_COMPARE_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "theta_jn",
]

logger = logging.getLogger("evaluate_glitch_robustness")


def _ensure_paths() -> None:
    bns = REPO_ROOT / "DINGO-BNS" / "dingo"
    src = REPO_ROOT / "src"
    scripts = REPO_ROOT / "scripts"
    if bns.is_dir() and str(bns) not in sys.path:
        sys.path.insert(0, str(bns))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def sine_gaussian_glitch(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    f0: float = 100.0,
    q: float = 5.0,
    amplitude: float,
) -> np.ndarray:
    """Sine-Gaussian blip: ``A * exp(-(t-t0)^2/(2σ^2)) * cos(2π f0 (t-t0))``."""
    from adapt.stft_context import sine_gaussian_glitch as _sg

    return _sg(
        n_samples,
        sample_rate,
        t_peak=t_peak,
        f0=f0,
        q=q,
        amplitude=amplitude,
    )


def load_full_event_td(
    assets: Dict[str, Any],
    settings: Dict[str, Any],
    det: str = "H1",
) -> Tuple[np.ndarray, float, float]:
    """Load full analysis-segment TD for ``det`` (length ``T`` at ``f_s``)."""
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    f_s = float(settings.get("f_s", 4096.0))
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    start = trigger - (duration - time_buffer)
    end = trigger + time_buffer
    path = assets["strain_gwf"][det]
    channel = assets["channels"][det]
    logger.info("Loading full TD %s [%s, %s] from %s", det, start, end, path.name)
    ts = TimeSeries.read(str(path), channel=channel)
    ts = ts.crop(start, end)
    if float(ts.sample_rate.value) != f_s:
        ts = ts.resample(f_s)
    td = np.asarray(ts.value, dtype=np.float64)
    n_expected = int(round(duration * f_s))
    if td.size > n_expected:
        td = td[:n_expected]
    elif td.size < n_expected:
        td = np.pad(td, (0, n_expected - td.size))
    return td, f_s, time_buffer


def td_to_fd_strain(
    td: np.ndarray,
    sample_rate: float,
    *,
    roll_off: float = 0.4,
    f_max: float,
) -> np.ndarray:
    """Tukey-windowed FFT matching demo conditioning (trigger at t=0 via prior crop)."""
    x = np.asarray(td, dtype=np.float64).ravel()
    alpha = 2.0 * float(roll_off) * float(sample_rate) / max(len(x), 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    window = sp_signal.windows.tukey(len(x), alpha=alpha)
    xw = x * window
    fd = np.fft.rfft(xw) / float(sample_rate)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate))
    # Truncate / pad to f_max grid used by the event file
    n_keep = int(np.floor(float(f_max) / (freqs[1] if len(freqs) > 1 else 1.0))) + 1
    n_keep = min(n_keep, len(fd))
    return np.asarray(fd[:n_keep], dtype=np.complex128)


def welch_asd(
    td: np.ndarray,
    sample_rate: float,
    *,
    n_freq: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Welch ASD on the analysis segment; smear transients into a 1-D spectrum."""
    nperseg = int(min(len(td), max(256, int(round(sample_rate * 4.0)))))
    freqs, psd = sp_signal.welch(
        np.asarray(td, dtype=np.float64),
        fs=float(sample_rate),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    # Interpolate onto the FD frequency grid matching the packaged strain length.
    target_f = np.linspace(0.0, float(f_max), int(n_freq), dtype=np.float64)
    psd_i = np.interp(target_f, freqs, psd, left=psd[0], right=psd[-1])
    asd = np.sqrt(np.maximum(psd_i, 0.0))
    asd[target_f < float(f_min)] = 1.0
    asd = np.maximum(asd, 1e-30)
    return asd.astype(np.float64)


def inject_h1_glitch_into_event(
    event,
    assets: Dict[str, Any],
    *,
    f0: float = 100.0,
    q: float = 5.0,
    snr_amp_scale: float = 8.0,
    t_rel: float = -1.0,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Return glitchy event ``data`` dict, TD map for STFT, and glitch metadata."""
    settings = dict(event.settings)
    td_h1, f_s, time_buffer = load_full_event_td(assets, settings, "H1")
    duration = float(settings.get("T", 128.0))
    # Segment starts at trigger - (T - time_buffer); glitch at trigger + t_rel.
    t_peak = (duration - time_buffer) + float(t_rel)
    # Match training: amplitude from in-band RMS (23–f_max Hz), not broadband
    # GWF RMS (dominated by sub-20 Hz seismic, ~1e4× larger).
    from adapt.stft_context import inband_rms

    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    rms_broadband = float(np.std(td_h1)) or 1e-22
    rms = inband_rms(td_h1, f_s, f_min=f_min, f_max=f_max)
    amp = snr_amp_scale * rms
    glitch = sine_gaussian_glitch(
        len(td_h1), f_s, t_peak=t_peak, f0=f0, q=q, amplitude=amp
    )
    td_h1_g = td_h1 + glitch

    data = copy.deepcopy(event.data)
    roll_off = float(settings.get("roll_off", 0.4))
    n_freq = len(data["waveform"]["H1"])

    # Add FFT(glitch) onto the packaged H1 FD strain (preserves demo conditioning).
    glitch_fd = td_to_fd_strain(glitch, f_s, roll_off=roll_off, f_max=f_max)
    if len(glitch_fd) < n_freq:
        glitch_fd = np.pad(glitch_fd, (0, n_freq - len(glitch_fd)))
    else:
        glitch_fd = glitch_fd[:n_freq]
    data["waveform"]["H1"] = np.asarray(data["waveform"]["H1"], dtype=np.complex128) + glitch_fd
    data["asds"]["H1"] = welch_asd(
        td_h1_g, f_s, n_freq=n_freq, f_min=f_min, f_max=f_max
    )

    # STFT TD crops: centered 4 s around trigger (includes t=-1.0 s).
    from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS

    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * f_s))
    # Within full segment, trigger is at index (duration - time_buffer) * f_s
    trig_idx = int(round((duration - time_buffer) * f_s))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    if end > len(td_h1_g):
        end = len(td_h1_g)
        start = end - n_crop
    td_stft = {
        "H1": td_h1_g[start:end].copy(),
    }
    # L1/V1: load clean crops via comparison helper for STFT stack
    from evaluate_gw170817_comparison import load_event_td_crops

    clean = load_event_td_crops(
        assets, sample_rate=f_s, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
    )
    td_stft["L1"] = clean["L1"]
    td_stft["V1"] = clean["V1"]

    meta = {
        "det": "H1",
        "f0": f0,
        "q": q,
        "t_rel": t_rel,
        "amplitude": amp,
        "rms": rms,
        "rms_inband": rms,
        "rms_broadband": rms_broadband,
        "rms_basis": "inband",
        "t_peak_in_segment": t_peak,
        # Full analysis-segment TD used for honest detect-and-gate rebuild.
        # Not JSON-serializable as-is; callers should pop before dumping reports.
        "td_full": {"H1": td_h1_g},
        "td_clean_full": {"H1": td_h1},
        "sample_rate": float(f_s),
        "duration": float(duration),
        "time_buffer": float(time_buffer),
        "roll_off": float(roll_off),
        "f_min": float(f_min),
        "f_max": float(f_max),
        "asd_policy": "welch",
    }
    logger.info(
        "Injected H1 sine-Gaussian: f0=%.1f Hz Q=%.1f amp=%.3e "
        "(%.1fx in-band RMS; broadband RMS=%.3e) t_rel=%.2fs",
        f0,
        q,
        amp,
        snr_amp_scale,
        rms_broadband,
        t_rel,
    )
    return data, td_stft, meta


def glitch_meta_for_json(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large TD arrays before writing reports."""
    skip = {"td_full", "td_clean_full"}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in skip:
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def apply_detect_and_gate_excision(
    glitchy_data: Dict[str, Any],
    *,
    td_full: Dict[str, np.ndarray],
    gates,
    sample_rate: float,
    roll_off: float,
    f_max: float,
    original_asds: Dict[str, np.ndarray],
    keep_original_asd: bool = True,
):
    """Honest path: replace gated-IFO FD from gated TD; restore original ASDs."""
    from adapt.glitch_excision import rebuild_event_from_gated_td

    return rebuild_event_from_gated_td(
        glitchy_data,
        td_by_det=td_full,
        gates=gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        t0=0.0,
        original_asds=original_asds if keep_original_asd else None,
    )


def resolve_custom_ckpt(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for candidate in (DEFAULT_BEST, LEGACY_BEST):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Custom checkpoint not found; tried {DEFAULT_BEST} and {LEGACY_BEST}"
    )


def _dl_ci_df(df: pd.DataFrame) -> Tuple[float, float, float]:
    x = np.asarray(df["luminosity_distance"], dtype=np.float64)
    x = x[np.isfinite(x)]
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return float(med), float(lo), float(hi)


def _load_glitch_detector(path: Path, device: torch.device):
    from adapt.models import GlitchDetectorSTFT

    raw = torch.load(path, map_location="cpu", weights_only=False)
    kw = dict(raw.get("model_kwargs") or {})
    model = GlitchDetectorSTFT(**kw)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, raw


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


def _detector_gates_from_crop(
    model,
    spectrogram: np.ndarray,
    *,
    detectors: Sequence[str],
    crop_start: int,
    sample_rate: float,
    threshold: float,
    gate_half_s: float,
    device: torch.device,
    ifo_whitelist: Optional[Sequence[str]] = None,
):
    from adapt.glitch_excision import time_bin_mask_to_windows
    from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS

    spec_t = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_probs(spec_t.to(device)).cpu().numpy()[0]
    gates = []
    t0 = float(crop_start) / float(sample_rate)
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    allow = set(ifo_whitelist) if ifo_whitelist is not None else None
    for di, det in enumerate(detectors):
        if di >= probs.shape[0]:
            break
        if allow is not None and det not in allow:
            continue
        bin_mask = probs[di] > float(threshold)
        n_hit = int(bin_mask.sum())
        if n_hit == 0:
            continue
        if n_hit == 1 and float(np.max(probs[di])) < float(threshold) + 0.15:
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


def run_legacy_embedding_repair(args: argparse.Namespace) -> None:
    """Historical ablation: frozen/custom STFT embedding repair on Welch glitch."""
    from adapt.glitch_augmentation import make_fixed_eval_glitch, stft_whitening_asds
    from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS
    from evaluate_gw170817_comparison import (
        build_event_spectrogram_stack,
        discover_assets,
        load_custom_wrapper,
        load_event_dataset,
        package_event_strain,
        run_baseline_sampling,
        run_custom_sampling,
        save_samples_pt,
        select_device,
        standardize_context,
    )
    from dingo.gw.domains import build_domain_from_model_metadata

    device = select_device(args.device)
    logger.info("Device: %s (legacy embedding-repair ablation)", device)

    custom_ckpt = resolve_custom_ckpt(
        Path(args.custom_ckpt) if args.custom_ckpt else None
    )
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=custom_ckpt,
    )
    assets["custom_ckpt"] = custom_ckpt
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]
    _ = make_fixed_eval_glitch(
        det="H1",
        family="sine_gaussian",
        severity=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
        asd_policy="welch",
        f0=float(args.f0),
        q=float(args.q),
    )
    glitchy_data, td_stft, glitch_meta = inject_h1_glitch_into_event(
        event,
        assets,
        f0=float(args.f0),
        q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
    )
    glitch_meta["shared_conditioner"] = "adapt.glitch_augmentation.make_fixed_eval_glitch"
    glitch_event = SimpleNamespace(data=glitchy_data, settings=event.settings)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("===== Baseline on glitchy H1 (%d samples) =====", args.num_samples)
    baseline_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        glitch_event,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    logger.info("===== Custom STFT on glitchy H1 (%d samples) =====", args.num_samples)
    wrapper, metadata = load_custom_wrapper(
        assets["baseline_ckpt"], assets["custom_ckpt"], device
    )
    strain = package_event_strain(glitchy_data, metadata, fixed)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        event.settings.get("f_s")
        or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    stft_kw = metadata.get("_stft_kwargs") or {}
    stft_asds = stft_whitening_asds(
        {det: np.asarray(event.data["asds"][det]) for det in detectors},
        {det: np.asarray(glitchy_data["asds"][det]) for det in detectors},
    )
    spectrogram, log_energy = build_event_spectrogram_stack(
        td_stft,
        detectors,
        sample_rate,
        asds=stft_asds,
        delta_f=float(base_domain.delta_f),
        noise_std=float(base_domain.noise_std),
        robust=bool(metadata.get("_glitch_robust")),
        norm_stats=metadata.get("_norm_stats"),
        n_time=stft_kw.get("n_time"),
        n_freq=stft_kw.get("n_freq"),
        n_fft=stft_kw.get("n_fft"),
        win_length=stft_kw.get("win_length"),
        hop_length=stft_kw.get("hop_length"),
    )
    context_z = standardize_context(
        fixed, metadata["train_settings"]["data"]["standardization"]
    )
    custom_df = run_custom_sampling(
        wrapper,
        metadata,
        strain,
        spectrogram,
        log_energy,
        context_z,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    source_paths = {
        "event_hdf5": assets["event_hdf5"],
        "baseline_ckpt": assets["baseline_ckpt"],
        "custom_ckpt": assets["custom_ckpt"],
        "glitch": glitch_meta_for_json(glitch_meta),
        "mode": "legacy_embedding_repair",
    }
    save_samples_pt(
        outdir / "gw170817_glitch_baseline_samples.pt",
        baseline_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={"model": "baseline_glitchy", "glitch": glitch_meta_for_json(glitch_meta)},
    )
    save_samples_pt(
        outdir / "gw170817_glitch_custom_samples.pt",
        custom_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={
            "model": "legacy_embedding_repair",
            "glitch": glitch_meta_for_json(glitch_meta),
            "spectrogram_shape": tuple(spectrogram.shape),
        },
    )
    b_med, b_lo, b_hi = _dl_ci_df(baseline_df)
    c_med, c_lo, c_hi = _dl_ci_df(custom_df)
    logger.info(
        "d_L 90%% CI | baseline=[%.3f, %.3f] (med=%.3f) | custom=[%.3f, %.3f] (med=%.3f)",
        b_lo,
        b_hi,
        b_med,
        c_lo,
        c_hi,
        c_med,
    )
    write_glitch_robustness_pdf(
        outdir / "GW170817_glitch_robustness_comparison.pdf",
        baseline_df,
        custom_df,
        glitch_meta=glitch_meta,
        custom_ckpt=assets["custom_ckpt"],
        custom_label="legacy embedding repair",
    )
    logger.info("Legacy embedding-repair evaluation complete → %s", outdir)


def run_detect_and_gate(args: argparse.Namespace) -> None:
    """Default: poison baseline vs gated-TD rebuild + original ASD + frozen DINGO."""
    from adapt.glitch_excision import analysis_crop_bounds
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )
    from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata
    from evaluate_gw170817_comparison import (
        discover_assets,
        load_event_dataset,
        load_event_td_crops,
        run_baseline_sampling,
        save_samples_pt,
        select_device,
    )
    from train_bns_spectrogram import load_bns_checkpoint

    device = select_device(args.device)
    logger.info("Device: %s (detect-and-gate + frozen DINGO)", device)

    # discover_assets still wants a custom_ckpt path for bookkeeping; prefer PE ckpts.
    pe_probe = None
    for cand in (
        Path(args.custom_ckpt) if args.custom_ckpt else None,
        DEFAULT_BEST,
        LEGACY_BEST,
        DEFAULT_DETECTOR,
    ):
        if cand is not None and Path(cand).is_file():
            pe_probe = Path(cand)
            break
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=pe_probe,
    )
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]
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
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)

    glitchy_data, td_stft, glitch_meta = inject_h1_glitch_into_event(
        event,
        assets,
        f0=float(args.f0),
        q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
    )
    td_full = dict(glitch_meta["td_full"])
    original_asds = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Baseline poison: glitch FD + Welch ----
    glitch_event = SimpleNamespace(data=glitchy_data, settings=settings)
    logger.info("===== Baseline poison (glitch+Welch) (%d samples) =====", args.num_samples)
    baseline_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        glitch_event,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # ---- Detect / oracle gates on the same full TD ----
    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    gate_half_s = float(args.gate_half_s)
    gate_meta: Dict[str, Any] = {"gate_half_s": gate_half_s}
    if det_path.is_file() and not args.oracle_gate:
        model, det_raw = _load_glitch_detector(det_path, device)
        threshold = float(
            args.threshold
            if args.threshold is not None
            else det_raw.get("threshold", 0.5)
        )
        norm_stats = det_raw.get("norm_stats")
        stft_kwargs = dict(det_raw.get("stft_kwargs") or {})
        gate_half_s = float(det_raw.get("gate_half_s", gate_half_s))
        gate_meta.update(
            {
                "detector_ckpt": str(det_path),
                "threshold": threshold,
                "gate_half_s": gate_half_s,
                "mode": "learned_detector",
            }
        )
        trig_idx, crop_start, crop_end = analysis_crop_bounds(
            duration=duration, time_buffer=time_buffer, sample_rate=sample_rate
        )
        crops_g = {
            "H1": td_full["H1"][crop_start:crop_end].copy(),
        }
        clean_crops = load_event_td_crops(
            assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
        )
        crops_g["L1"] = clean_crops["L1"]
        crops_g["V1"] = clean_crops["V1"]
        crops_w = whiten_td_map_with_asds(
            crops_g,
            original_asds,
            sample_rate=sample_rate,
            delta_f=delta_f,
            noise_std=noise_std,
            detectors=detectors,
        )
        spec_g, loge_g = build_robust_spectrogram_from_td(
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
            delta_f=delta_f,
            noise_std=noise_std,
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
        _, clean_probs = _detector_gates_from_crop(
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
        _, probs = _detector_gates_from_crop(
            model,
            spec_g,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=thr_event,
            gate_half_s=gate_half_s,
            device=device,
        )
        h1_max = float(np.max(probs[0])) if probs.shape[0] > 0 else 0.0
        whitelist = ["H1"] if h1_max >= thr_event else None
        gates, probs = _detector_gates_from_crop(
            model,
            spec_g,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=thr_event,
            gate_half_s=gate_half_s,
            device=device,
            ifo_whitelist=whitelist,
        )
        gate_meta.update(
            {
                "threshold_event": thr_event,
                "n_gates": len(gates),
                "probs_max": {
                    det: float(np.max(probs[i]))
                    for i, det in enumerate(detectors)
                    if i < probs.shape[0]
                },
                "log_energy": {
                    d: float(v)
                    for d, v in zip(("H1", "L1", "V1"), np.asarray(loge_g).ravel()[:3])
                },
                "ifo_whitelist": whitelist,
            }
        )
        if not gates:
            logger.warning("Detector silent on glitch — falling back to oracle H1 gate")
            gates = _oracle_h1_gates(
                t_rel=float(args.t_rel),
                duration=duration,
                time_buffer=time_buffer,
                half_s=gate_half_s,
            )
            gate_meta["fallback_oracle"] = True
    else:
        if not det_path.is_file():
            logger.warning("No detector at %s — using oracle H1 gate", det_path)
        gates = _oracle_h1_gates(
            t_rel=float(args.t_rel),
            duration=duration,
            time_buffer=time_buffer,
            half_s=gate_half_s,
        )
        gate_meta.update(
            {
                "detector_ckpt": str(det_path) if det_path.is_file() else None,
                "mode": "oracle",
                "n_gates": len(gates),
            }
        )

    excised = apply_detect_and_gate_excision(
        glitchy_data,
        td_full=td_full,
        gates=gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        original_asds=original_asds,
        keep_original_asd=True,
    )
    gate_meta["residual_power_frac"] = excised.meta.get("residual_power_frac")
    gate_meta["modified_detectors"] = list(excised.modified_detectors)
    logger.info(
        "Honest excision gates=%d residual_power=%s asd=%s",
        len(gates),
        gate_meta.get("residual_power_frac"),
        excised.meta.get("asd_policy"),
    )

    logger.info(
        "===== Detect-and-gate + original ASD + frozen DINGO (%d samples) =====",
        args.num_samples,
    )
    custom_event = SimpleNamespace(data=excised.data, settings=settings)
    custom_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        custom_event,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    source_paths = {
        "event_hdf5": assets["event_hdf5"],
        "baseline_ckpt": assets["baseline_ckpt"],
        "glitch": glitch_meta_for_json(glitch_meta),
        "gate": gate_meta,
        "mode": "detect_and_gate",
    }
    save_samples_pt(
        outdir / "gw170817_glitch_baseline_samples.pt",
        baseline_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={"model": "baseline_glitchy_welch", "glitch": glitch_meta_for_json(glitch_meta)},
    )
    save_samples_pt(
        outdir / "gw170817_glitch_custom_samples.pt",
        custom_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={
            "model": "detect_and_gate_frozen_dingo",
            "glitch": glitch_meta_for_json(glitch_meta),
            "gate": gate_meta,
        },
    )
    b_med, b_lo, b_hi = _dl_ci_df(baseline_df)
    c_med, c_lo, c_hi = _dl_ci_df(custom_df)
    logger.info(
        "d_L 90%% CI | poison=[%.3f, %.3f] (med=%.3f) | gated+origASD=[%.3f, %.3f] (med=%.3f)",
        b_lo,
        b_hi,
        b_med,
        c_lo,
        c_hi,
        c_med,
    )
    if c_med > 20.0 and (c_hi - c_lo) > 10.0 and b_hi < 15.0:
        logger.info("SUCCESS: detect-and-gate recovered d_L while poison stayed collapsed.")
    elif c_hi < 15.0:
        logger.warning("Detect-and-gate still collapsed near d_L~10 — check residual FD / ASD.")

    write_glitch_robustness_pdf(
        outdir / "GW170817_glitch_robustness_comparison.pdf",
        baseline_df,
        custom_df,
        glitch_meta=glitch_meta,
        custom_ckpt=Path(gate_meta.get("detector_ckpt") or "detect_and_gate"),
        custom_label="detect-and-gate + orig ASD",
        baseline_label="poison (glitch+Welch)",
    )
    logger.info("Detect-and-gate evaluation complete → %s", outdir)


def run(args: argparse.Namespace) -> None:
    _ensure_paths()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.legacy_embedding_repair:
        run_legacy_embedding_repair(args)
    else:
        run_detect_and_gate(args)


def write_glitch_robustness_pdf(
    path: Path,
    baseline_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    *,
    glitch_meta: Dict[str, Any],
    custom_ckpt: Path,
    custom_label: str = "custom (2-D STFT)",
    baseline_label: str = "baseline (1-D PSD)",
) -> None:
    """Single PDF: cover + median/CI table + overlaid corners."""
    import corner
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from evaluate_gw170817_comparison import (
        _draw_table,
        _has_dynamic_range,
        _quantile_ci,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    table_params = [
        p
        for p in GLITCH_COMPARE_PARAMS
        if p in baseline_df.columns and p in custom_df.columns
    ]
    corner_params = [
        p
        for p in table_params
        if _has_dynamic_range(baseline_df[p].to_numpy())
        and _has_dynamic_range(custom_df[p].to_numpy())
    ]

    amp_ratio = glitch_meta["amplitude"] / max(glitch_meta["rms"], 1e-30)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        lines = [
            "GW170817 glitch robustness comparison",
            "",
            f"Samples per model: {len(baseline_df)}",
            f"Custom: {custom_ckpt}",
            "",
            "Injected glitch (H1 only):",
            "  type: sine-Gaussian (blip-like)",
            f"  t_rel: {glitch_meta['t_rel']} s (relative to trigger)",
            f"  f0: {glitch_meta['f0']} Hz",
            f"  Q: {glitch_meta['q']}",
            f"  amplitude: {glitch_meta['amplitude']:.3e}  ({amp_ratio:.1f}x RMS)",
            "",
            f"{baseline_label}.",
            f"{custom_label}.",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=11,
        )
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7))
        rows = []
        for p in table_params:
            b_med, b_lo, b_hi = _quantile_ci(baseline_df[p].to_numpy())
            c_med, c_lo, c_hi = _quantile_ci(custom_df[p].to_numpy())
            rows.append(
                [
                    p,
                    f"{b_med:.6g}",
                    f"[{b_lo:.6g}, {b_hi:.6g}]",
                    f"{c_med:.6g}",
                    f"[{c_lo:.6g}, {c_hi:.6g}]",
                ]
            )
        _draw_table(
            ax,
            (
                "Glitchy GW170817 — medians & 90% CI\n"
                f"{baseline_label} N={len(baseline_df)} | "
                f"{custom_label} N={len(custom_df)}"
            ),
            rows,
            [
                "parameter",
                "baseline median",
                "baseline 90% CI",
                "custom median",
                "custom 90% CI",
            ],
        )
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if len(corner_params) >= 2:
            serif_old = mpl.rcParams["font.family"]
            mpl.rcParams["font.family"] = "serif"
            colors = ["#1f77b4", "#d62728"]
            labels = [baseline_label, custom_label]
            fig = None
            handles = []
            for color, label, df in zip(colors, labels, [baseline_df, custom_df]):
                data = df[corner_params].to_numpy()
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
                    labels=corner_params,
                    color=color,
                    smooth=1.0,
                    smooth1d=1.0,
                    plot_datapoints=False,
                    plot_density=False,
                    plot_contours=True,
                    levels=(0.5, 0.9),
                    bins=30,
                    no_fill_contours=True,
                    fig=fig,
                    range=ranges,
                )
                handles.append(
                    plt.Line2D([], [], color=color, label=label, linewidth=3)
                )
            assert fig is not None
            fig.legend(handles=handles, loc="upper right", fontsize=11)
            fig.suptitle(
                "GW170817 glitch robustness: baseline vs custom",
                fontsize=14,
                y=1.02,
            )
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            plt.close(fig)
            mpl.rcParams["font.family"] = serif_old

    logger.info("Wrote %s", path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GW170817 glitch robustness evaluation")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument(
        "--custom-ckpt",
        type=Path,
        default=None,
        help="Only used with --legacy-embedding-repair (default: best glitch_robust)",
    )
    p.add_argument(
        "--detector-ckpt",
        type=Path,
        default=None,
        help=f"Glitch detector for detect-and-gate (default: {DEFAULT_DETECTOR})",
    )
    p.add_argument(
        "--legacy-embedding-repair",
        action="store_true",
        help="Ablation: old STFT embedding-repair PE head instead of detect-and-gate",
    )
    p.add_argument(
        "--oracle-gate",
        action="store_true",
        help="Skip learned detector; use ±gate-half-s around known H1 glitch peak",
    )
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--f0", type=float, default=100.0, help="Sine-Gaussian f0 (Hz)")
    p.add_argument("--q", type=float, default=5.0, help="Sine-Gaussian Q")
    p.add_argument(
        "--snr-amp-scale",
        type=float,
        default=8.0,
        help="Glitch amplitude as multiple of H1 RMS",
    )
    p.add_argument(
        "--t-rel",
        type=float,
        default=-1.0,
        help="Glitch peak relative to trigger (s)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
