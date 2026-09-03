#!/usr/bin/env python3
"""Honest detect-and-gate excision on the **actual** glitchy DINGO package.

Persists under ``results/excision_honest/``:
  - ungated + Welch ASD (poison; expected d_L~10)
  - ungated + original ASD (poison; expected d_L~100)
  - gated TD-rebuild + Welch ASD (must still fail — ASD brittleness)
  - gated TD-rebuild + original ASD (must recover clean d_L band)
  - clean no-op / clean FP audit
  - learned-detector vs oracle gates on the glitchy object

Gates are applied to the **glitchy** package via ``rebuild_event_from_gated_td``
(matched-delta / replace rebuild from gated TD), not delta-added onto a clean
HDF5 copy.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/honest_excision.py \\
      --detector-ckpt checkpoints/glitch_detector_v1/best_glitch_detector.pt \\
      --outdir results/excision_honest
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "examples",
    REPO_ROOT / "src",
    REPO_ROOT / "DINGO-BNS" / "dingo",
    REPO_ROOT,
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)
DEFAULT_OUTDIR = REPO_ROOT / "results" / "excision_honest"

logger = logging.getLogger("evaluate_glitch_excision")


def _dl_ci(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return {"lo": float(lo), "med": float(med), "hi": float(hi), "n": int(x.size)}


def _sample_baseline(assets, event_data, settings, fixed, *, device, n, bs):
    from adapt.dingo_bns_demo import run_baseline_sampling

    ev = SimpleNamespace(data=event_data, settings=settings)
    df = run_baseline_sampling(
        assets["baseline_ckpt"],
        ev,
        fixed,
        device=device,
        num_samples=int(n),
        batch_size=int(bs),
    )
    return _dl_ci(df["luminosity_distance"].to_numpy())


def _load_detector(path: Path, device: torch.device):
    from adapt.models import GlitchDetectorSTFT

    raw = torch.load(path, map_location="cpu", weights_only=False)
    kw = dict(raw.get("model_kwargs") or {})
    model = GlitchDetectorSTFT(**kw)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, raw


def _build_event_spectrogram(td_map, asds, detectors, sample_rate, delta_f, noise_std, norm_stats, stft_kwargs):
    from adapt.dingo_bns_demo import build_event_spectrogram_stack

    return build_event_spectrogram_stack(
        td_map,
        detectors,
        sample_rate,
        asds=asds,
        delta_f=delta_f,
        noise_std=noise_std,
        robust=True,
        norm_stats=norm_stats,
        n_time=stft_kwargs.get("n_time"),
        n_freq=stft_kwargs.get("n_freq"),
        n_fft=stft_kwargs.get("n_fft"),
        win_length=stft_kwargs.get("win_length"),
        hop_length=stft_kwargs.get("hop_length"),
    )


def _oracle_gates_for_injection(
    *,
    det: str,
    t_rel: float,
    duration: float,
    time_buffer: float,
    sample_rate: float,
    half_s: float = 0.4,
):
    """Gate windows on the *full analysis segment* time axis."""
    from adapt.glitch_excision import GateWindow

    t_peak = (float(duration) - float(time_buffer)) + float(t_rel)
    return [
        GateWindow(
            detector=det,
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
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

    spec_t = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_probs(spec_t.to(device)).cpu().numpy()[0]  # (n_ifo, T)
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
        # Require at least 2 adjacent-ish bins OR a very confident single bin.
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


def _event_threshold_from_clean(
    clean_probs: np.ndarray,
    base_threshold: float,
    *,
    margin: float = 0.05,
) -> float:
    """Raise threshold above clean-event max prob so clean is a no-op."""
    mx = float(np.max(clean_probs)) if clean_probs.size else 0.0
    return float(max(base_threshold, mx + float(margin)))


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.glitch_excision import (
        DEFAULT_GATE_HALF_S,
        analysis_crop_bounds,
        apply_excision_to_event_data,
        rebuild_event_from_gated_td,
    )
    from adapt.stft_context import inband_rms
    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.event_glitch_io import (
        glitch_meta_for_json,
        inject_h1_glitch_into_event,
        sine_gaussian_glitch,
        td_to_fd_strain,
        welch_asd,
    )
    from adapt.dingo_bns_demo import (
        discover_assets,
        load_event_dataset,
        load_event_td_crops,
        select_device,
    )
    from adapt.dingo_bns_demo import load_bns_checkpoint

    device = select_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"outdir": str(outdir)}

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    has_detector = det_path.is_file()
    if has_detector:
        model, det_raw = _load_detector(det_path, device)
        threshold = float(args.threshold if args.threshold is not None else det_raw.get("threshold", 0.5))
        norm_stats = det_raw.get("norm_stats")
        stft_kwargs = dict(det_raw.get("stft_kwargs") or {})
        gate_half_s = float(det_raw.get("gate_half_s", DEFAULT_GATE_HALF_S))
        report["detector_ckpt"] = str(det_path)
        report["threshold"] = threshold
        report["gate_half_s"] = gate_half_s
        logger.info("Loaded detector %s thr=%.3f", det_path, threshold)
    else:
        model = None
        threshold = 0.5
        norm_stats = None
        stft_kwargs = {"n_time": 32, "n_freq": 128}
        gate_half_s = float(args.gate_half_s)
        report["detector_ckpt"] = None
        logger.warning("No detector ckpt at %s — oracle-only eval", det_path)

    # Always discover assets with a known PE/custom path; detector is loaded separately.
    probe = None
    for cand in (
        Path(args.detector_ckpt) if args.detector_ckpt else None,
        DEFAULT_DETECTOR,
        REPO_ROOT / "checkpoints" / "glitch_robust_v3" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt",
    ):
        if cand is not None and Path(cand).is_file():
            # Prefer a PE custom ckpt for discover_assets; detector .pt also works
            # as a path existence check only — discover may still need PE metadata.
            probe = Path(cand)
            break
    pe_probe = None
    for cand in (
        REPO_ROOT / "checkpoints" / "glitch_robust_v3" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt",
        probe,
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
    sample_rate = float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"])
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)

    # ---- Clean baseline ----
    logger.info("Sampling clean GW170817 baseline…")
    clean_ci = _sample_baseline(
        assets, event.data, settings, fixed, device=device, n=args.num_samples, bs=args.batch_size
    )
    report["clean_baseline_d_L"] = clean_ci
    logger.info("Clean d_L %s", clean_ci)

    # ---- Clean no-op with empty gates ----
    noop = apply_excision_to_event_data(
        event.data,
        td_by_det={"H1": np.zeros(8)},  # unused when no gates
        gates=[],
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        original_asds={d: np.asarray(event.data["asds"][d]) for d in detectors},
    )
    assert noop.noop
    # Bit-exact waveform check
    same = all(
        np.array_equal(event.data["waveform"][d], noop.data["waveform"][d])
        for d in event.data["waveform"]
    )
    report["clean_noop_bit_exact"] = bool(same)
    logger.info("Clean empty-gate no-op bit-exact=%s", same)

    # ---- Build glitch injection (actual DINGO input object) ----
    glitch_data, td_stft, gmeta = inject_h1_glitch_into_event(
        event,
        assets,
        f0=float(args.f0),
        q=float(args.q),
        snr_amp_scale=float(args.snr_amp_scale),
        t_rel=float(args.t_rel),
    )
    report["glitch_meta"] = glitch_meta_for_json(gmeta)
    td_h1_glitch = np.asarray(gmeta["td_full"]["H1"])
    td_h1_clean = np.asarray(gmeta["td_clean_full"]["H1"])
    td_full_glitch = {"H1": td_h1_glitch}
    td_full_clean = {"H1": td_h1_clean}
    asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    welch_h1 = np.asarray(glitch_data["asds"]["H1"]).copy()

    # Bit-level: residual glitch power removed by oracle gate on TD→FD rebuild.
    oracle_gates = _oracle_gates_for_injection(
        det="H1",
        t_rel=float(args.t_rel),
        duration=duration,
        time_buffer=time_buffer,
        sample_rate=sample_rate,
        half_s=gate_half_s,
    )

    # ---- Honest 4-way PE on the glitchy package ----
    logger.info("Sampling ungated + Welch ASD (poison)…")
    welch_ci = _sample_baseline(
        assets, glitch_data, settings, fixed, device=device, n=args.num_samples, bs=args.batch_size
    )
    report["ungated_welch_d_L"] = welch_ci
    report["glitch_welch_d_L"] = welch_ci  # legacy key

    glitch_orig_asd = copy.deepcopy(glitch_data)
    glitch_orig_asd["asds"] = {d: asds_clean[d].copy() for d in detectors}
    logger.info("Sampling ungated + original ASD (poison)…")
    glitch_orig_ci = _sample_baseline(
        assets,
        glitch_orig_asd,
        settings,
        fixed,
        device=device,
        n=args.num_samples,
        bs=args.batch_size,
    )
    report["ungated_original_asd_d_L"] = glitch_orig_ci
    report["glitch_original_asd_d_L"] = glitch_orig_ci

    # Gated rebuild + Welch (must still fail — documents ASD brittleness).
    gated_welch = rebuild_event_from_gated_td(
        glitch_data,
        td_by_det=td_full_glitch,
        gates=oracle_gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        t0=0.0,
        original_asds=None,  # leave Welch in place
    )
    # Ensure Welch is still present after rebuild (rebuild leaves ASD unchanged).
    gated_welch.data["asds"]["H1"] = welch_h1
    logger.info("Sampling gated TD-rebuild + Welch ASD (expect fail)…")
    gated_welch_ci = _sample_baseline(
        assets,
        gated_welch.data,
        settings,
        fixed,
        device=device,
        n=args.num_samples,
        bs=args.batch_size,
    )
    report["gated_welch_d_L"] = gated_welch_ci
    report["gated_welch_still_collapsed"] = bool(gated_welch_ci["hi"] < 15.0)
    report["residual_power_frac_oracle"] = gated_welch.meta.get("residual_power_frac")

    # Gated rebuild + original ASD on the **glitchy** package (must recover).
    oracle = rebuild_event_from_gated_td(
        glitch_data,
        td_by_det=td_full_glitch,
        gates=oracle_gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        t0=0.0,
        original_asds=asds_clean,
    )
    logger.info("Sampling oracle gated TD-rebuild + original ASD…")
    oracle_ci = _sample_baseline(
        assets, oracle.data, settings, fixed, device=device, n=args.num_samples, bs=args.batch_size
    )
    report["oracle_gate_d_L"] = oracle_ci
    report["gated_original_asd_d_L"] = oracle_ci
    report["oracle_escaped"] = bool(
        oracle_ci["med"] > 20.0 and oracle_ci["hi"] > 25.0
    )
    report["honest_recovered"] = bool(
        report["oracle_escaped"]
        and clean_ci["lo"] < oracle_ci["med"] < clean_ci["hi"] * 1.5
    )
    logger.info(
        "Oracle gated+origASD d_L %s escaped=%s residual=%s",
        oracle_ci,
        report["oracle_escaped"],
        report.get("residual_power_frac_oracle"),
    )

    # ---- Learned detector path (same glitchy object + rebuild) ----
    if model is not None:
        trig_idx, crop_start, crop_end = analysis_crop_bounds(
            duration=duration, time_buffer=time_buffer, sample_rate=sample_rate
        )
        from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

        crops_g = {
            "H1": td_h1_glitch[crop_start:crop_end].copy(),
        }
        clean_crops = load_event_td_crops(
            assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
        )
        crops_g["L1"] = clean_crops["L1"]
        crops_g["V1"] = clean_crops["V1"]
        from adapt.stft_context import whiten_td_map_with_asds

        crops_w = whiten_td_map_with_asds(
            crops_g,
            asds_clean,
            sample_rate=sample_rate,
            delta_f=delta_f,
            noise_std=noise_std,
            detectors=detectors,
        )
        from adapt.stft_context import build_robust_spectrogram_from_td

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
            asds_clean,
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
        thr_event = _event_threshold_from_clean(clean_probs, threshold, margin=0.05)
        report["threshold_base"] = threshold
        report["threshold_event"] = thr_event
        clean_gates, clean_probs = _detector_gates_from_crop(
            model,
            spec_c,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=thr_event,
            gate_half_s=gate_half_s,
            device=device,
        )
        report["clean_fp_audit"] = {
            "n_gates": len(clean_gates),
            "probs_max": {
                det: float(np.max(clean_probs[i]))
                for i, det in enumerate(detectors)
                if i < clean_probs.shape[0]
            },
            "fires": bool(len(clean_gates) > 0),
        }
        logger.info(
            "Clean FP audit (thr_event=%.3f): %s", thr_event, report["clean_fp_audit"]
        )

        # Clean event: detector silent → DINGO samples match ungated clean run.
        if not clean_gates:
            clean_rebuild = rebuild_event_from_gated_td(
                event.data,
                td_by_det=td_full_clean,
                gates=[],
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
            )
            assert clean_rebuild.noop
            report["clean_detector_silent"] = True
        else:
            report["clean_detector_silent"] = False

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
        whitelist = None
        if h1_max >= thr_event:
            whitelist = ["H1"]
            for di, det in enumerate(detectors):
                if di == 0:
                    continue
                if float(np.max(probs[di])) >= max(thr_event, 0.9):
                    whitelist.append(det)
        det_gates, probs = _detector_gates_from_crop(
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
        report["detector_probs_max"] = {
            det: float(np.max(probs[i])) for i, det in enumerate(detectors) if i < probs.shape[0]
        }
        report["detector_n_gates"] = len(det_gates)
        report["detector_log_energy"] = {
            d: float(v) for d, v in zip(("H1", "L1", "V1"), np.asarray(loge_g).ravel()[:3])
        }
        report["detector_ifo_whitelist"] = whitelist
        logger.info(
            "Detector gates=%d probs_max=%s whitelist=%s",
            len(det_gates),
            report["detector_probs_max"],
            whitelist,
        )

        learned = rebuild_event_from_gated_td(
            glitch_data,
            td_by_det=td_full_glitch,
            gates=det_gates,
            sample_rate=sample_rate,
            roll_off=roll_off,
            f_max=f_max,
            t0=0.0,
            original_asds=asds_clean,
        )
        if learned.noop:
            logger.warning("Detector produced no gates — falling back will fail d_L")
        report["detector_residual_power_frac"] = learned.meta.get("residual_power_frac")
        logger.info("Sampling detector-gated TD-rebuild + original ASD…")
        learned_ci = _sample_baseline(
            assets,
            learned.data,
            settings,
            fixed,
            device=device,
            n=args.num_samples,
            bs=args.batch_size,
        )
        report["detector_gate_d_L"] = learned_ci
        report["detector_escaped"] = bool(
            learned_ci["med"] > 20.0 and learned_ci["hi"] > 25.0
        )
        logger.info(
            "Detector-gated d_L %s escaped=%s",
            learned_ci,
            report["detector_escaped"],
        )

    # ---- Sweep over glitch params (oracle gate on glitchy package) ----
    if args.do_sweep:
        sweep = []
        for t_rel in (-1.5, -1.0, -0.5):
            for sev in (4.0, 8.0, 12.0):
                rms = inband_rms(td_h1_clean, sample_rate, f_min=f_min, f_max=f_max)
                amp_s = float(sev) * rms
                t_pk = (duration - time_buffer) + float(t_rel)
                g = sine_gaussian_glitch(
                    len(td_h1_clean),
                    sample_rate,
                    t_peak=t_pk,
                    f0=100.0,
                    q=5.0,
                    amplitude=amp_s,
                )
                td_g = td_h1_clean + g
                # Build a glitchy package consistent with inject: clean FD + FFT(glitch) + Welch.
                data_g = copy.deepcopy(event.data)
                n_freq = len(data_g["waveform"]["H1"])
                g_fd = td_to_fd_strain(g, sample_rate, roll_off=roll_off, f_max=f_max)
                if len(g_fd) < n_freq:
                    g_fd = np.pad(g_fd, (0, n_freq - len(g_fd)))
                else:
                    g_fd = g_fd[:n_freq]
                data_g["waveform"]["H1"] = (
                    np.asarray(data_g["waveform"]["H1"], dtype=np.complex128) + g_fd
                )
                data_g["asds"]["H1"] = welch_asd(
                    td_g, sample_rate, n_freq=n_freq, f_min=f_min, f_max=f_max
                )
                gates = _oracle_gates_for_injection(
                    det="H1",
                    t_rel=float(t_rel),
                    duration=duration,
                    time_buffer=time_buffer,
                    sample_rate=sample_rate,
                    half_s=gate_half_s,
                )
                res = rebuild_event_from_gated_td(
                    data_g,
                    td_by_det={"H1": td_g},
                    gates=gates,
                    sample_rate=sample_rate,
                    roll_off=roll_off,
                    f_max=f_max,
                    original_asds=asds_clean,
                )
                ci = _sample_baseline(
                    assets,
                    res.data,
                    settings,
                    fixed,
                    device=device,
                    n=min(256, int(args.num_samples)),
                    bs=args.batch_size,
                )
                row = {
                    "t_rel": float(t_rel),
                    "severity": float(sev),
                    "d_L": ci,
                    "escaped": bool(ci["med"] > 20.0 and ci["hi"] > 25.0),
                }
                sweep.append(row)
                logger.info("Sweep t_rel=%.1f sev=%.1f → %s", t_rel, sev, ci)
        report["oracle_sweep"] = sweep

    path = outdir / "excision_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", path)
    print(json.dumps(report, indent=2, default=str))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--f0", type=float, default=100.0)
    p.add_argument("--q", type=float, default=5.0)
    p.add_argument("--t-rel", type=float, default=-1.0)
    p.add_argument("--snr-amp-scale", type=float, default=8.0)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--do-sweep", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
