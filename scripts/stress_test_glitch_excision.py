#!/usr/bin/env python3
"""Paper-scale detect-and-gate stress test on GW170817 (frozen DINGO).

Locked factorial over all 8 glitch families × severity × ASD × seeds.
Poison = glitchy FD + ASD policy; gated = detect-and-gate + original ASD.
Artifacts under ``results/stress_test_excision_v1/``.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
    python scripts/stress_test_glitch_excision.py \\
      --seed 0 --n-seeds-per-cell 5 \\
      --num-samples 512 --hf-samples 2000 \\
      --outdir results/stress_test_excision_v1
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
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

DEFAULT_OUTDIR = REPO_ROOT / "results" / "stress_test_excision_v1"
DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)
DEFAULT_GOLD = REPO_ROOT / "results" / "dingo_official_control" / "comparison_report.json"

FAMILIES = (
    "sine_gaussian",
    "broadband_burst",
    "whistle",
    "scattered_light",
    "glitch_train",
    "ringing",
    "double_blip",
    "narrowband_tone",
)
SEVERITY_MIDS = (3.0, 6.0, 10.0)
ASD_POLICIES = ("welch", "stationary")

logger = logging.getLogger("stress_test_glitch_excision")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _dl_ci(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"lo": float("nan"), "med": float("nan"), "hi": float("nan"), "n": 0}
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return {"lo": float(lo), "med": float(med), "hi": float(hi), "n": int(x.size)}


def _ci_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


def _poison_collapsed(ci: Dict[str, float]) -> bool:
    return bool(ci["hi"] < 15.0 or ci["lo"] > 90.0)


def _gated_recovers(ci: Dict[str, float], clean: Dict[str, float]) -> bool:
    if not (20.0 <= ci["med"] <= 50.0):
        return False
    if not _ci_overlap(ci, clean):
        return False
    if abs(ci["med"] - clean["med"]) > 10.0:
        return False
    return True


def _sample_family_params(family: str, rng: np.random.Generator) -> Dict[str, Any]:
    if family == "sine_gaussian":
        return {"f0": float(rng.uniform(30.0, 300.0)), "q": float(rng.uniform(3.0, 15.0))}
    if family == "broadband_burst":
        return {"duration": float(rng.uniform(0.02, 0.25))}
    if family == "whistle":
        f0 = float(rng.uniform(40.0, 200.0))
        return {
            "duration": float(rng.uniform(0.2, 1.5)),
            "f0": f0,
            "f1": float(rng.uniform(f0 + 20.0, min(500.0, f0 + 250.0))),
        }
    if family == "scattered_light":
        return {
            "duration": float(rng.uniform(0.5, 2.5)),
            "f_min": float(rng.uniform(20.0, 40.0)),
            "f_max": float(rng.uniform(60.0, 120.0)),
        }
    if family == "glitch_train":
        return {
            "n_pulses": int(rng.integers(2, 6)),
            "spacing": float(rng.uniform(0.05, 0.25)),
            "f0": float(rng.uniform(40.0, 200.0)),
            "q": float(rng.uniform(3.0, 10.0)),
        }
    if family == "ringing":
        return {
            "f0": float(rng.uniform(50.0, 250.0)),
            "decay": float(rng.uniform(0.05, 0.4)),
        }
    if family == "double_blip":
        return {
            "sep": float(rng.uniform(0.05, 0.3)),
            "f0": float(rng.uniform(40.0, 200.0)),
            "q": float(rng.uniform(3.0, 12.0)),
        }
    if family == "narrowband_tone":
        return {
            "duration": float(rng.uniform(0.3, 1.5)),
            "f0": float(rng.uniform(50.0, 300.0)),
        }
    raise ValueError(family)


def build_locked_grid(
    *,
    master_seed: int,
    n_seeds_per_cell: int,
    max_cells: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from adapt.glitch_augmentation import HELD_OUT_FAMILIES

    held_out_set = set(HELD_OUT_FAMILIES)
    cells: List[Dict[str, Any]] = []
    cell_id = 0
    for family in FAMILIES:
        for sev in SEVERITY_MIDS:
            for asd in ASD_POLICIES:
                for k in range(int(n_seeds_per_cell)):
                    seed = int(master_seed) + 100_000 * cell_id + 17 * k
                    rng = np.random.default_rng(seed)
                    # 1/5 of seeds force L1; else H1 (canonical collapse IFO).
                    det = "L1" if (k % 5 == 4) else "H1"
                    t_rel = float(rng.uniform(-1.5, -0.3))
                    params = _sample_family_params(family, rng)
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "family": family,
                            "held_out": family in held_out_set,
                            "severity": float(sev),
                            "asd_policy": asd,
                            "seed": seed,
                            "seed_idx": k,
                            "detectors": [det],
                            "t_rel": t_rel,
                            "params": params,
                        }
                    )
                    cell_id += 1
                    if max_cells is not None and cell_id >= int(max_cells):
                        return cells
    return cells


def inject_spec_into_event(
    event,
    assets: Dict[str, Any],
    cell: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Inject GlitchSpec onto analysis-segment TD; return poison package + td_full."""
    from adapt.glitch_augmentation import GlitchSpec, synthesize_glitch_td
    from adapt.stft_context import inband_rms
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from evaluate_glitch_robustness import load_full_event_td, td_to_fd_strain, welch_asd
    from evaluate_gw170817_comparison import load_event_td_crops

    settings = dict(event.settings)
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    roll_off = float(settings.get("roll_off", 0.4))
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    sample_rate = float(settings.get("f_s", 4096.0))

    spec = GlitchSpec(
        family=str(cell["family"]),
        detectors=list(cell["detectors"]),
        t_rel=float(cell["t_rel"]),
        severity=float(cell["severity"]),
        params=dict(cell["params"]),
        asd_policy=str(cell["asd_policy"]),
        held_out=bool(cell["held_out"]),
    )
    # Analysis-segment peak (trigger at duration - time_buffer).
    t_peak = (duration - time_buffer) + float(spec.t_rel)
    # synthesize_glitch_td uses mid-crop peak; override via temporary t_rel remap.
    t_rel_for_synth = t_peak - 0.5 * duration

    data = copy.deepcopy(event.data)
    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))
    td_full: Dict[str, np.ndarray] = {}
    td_clean_full: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(cell["seed"]))

    for det in ("H1", "L1", "V1"):
        td, f_s, _ = load_full_event_td(assets, settings, det)
        sample_rate = float(f_s)
        td_clean_full[det] = td
        if det in spec.detectors:
            rms = inband_rms(td, sample_rate, f_min=f_min, f_max=f_max)
            spec_synth = GlitchSpec(
                family=spec.family,
                detectors=[det],
                t_rel=t_rel_for_synth,
                severity=spec.severity,
                params=spec.params,
                asd_policy=spec.asd_policy,
                held_out=spec.held_out,
            )
            g = synthesize_glitch_td(
                len(td), sample_rate, spec_synth, rng, rms=rms
            )
            td_g = td + g
            td_full[det] = td_g
            g_fd = td_to_fd_strain(g, sample_rate, roll_off=roll_off, f_max=f_max)
            if len(g_fd) < n_freq:
                g_fd = np.pad(g_fd, (0, n_freq - len(g_fd)))
            else:
                g_fd = g_fd[:n_freq]
            data["waveform"][det] = (
                np.asarray(data["waveform"][det], dtype=np.complex128) + g_fd
            )
            if spec.asd_policy == "welch":
                data["asds"][det] = welch_asd(
                    td_g, sample_rate, n_freq=n_freq, f_min=f_min, f_max=f_max
                )
        else:
            td_full[det] = td.copy()

    # STFT crops
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    td_stft = {}
    for det in ("H1", "L1", "V1"):
        x = td_full[det]
        if end > len(x):
            end_i = len(x)
            start_i = end_i - n_crop
        else:
            start_i, end_i = start, end
        td_stft[det] = x[start_i:end_i].copy()

    meta = {
        "spec": spec.to_dict(),
        "t_peak_in_segment": float(t_peak),
        "sample_rate": float(sample_rate),
        "duration": float(duration),
        "time_buffer": float(time_buffer),
        "roll_off": float(roll_off),
        "f_min": float(f_min),
        "f_max": float(f_max),
        "crop_start": int(start),
        "td_full": td_full,
        "td_stft": td_stft,
    }
    return data, td_stft, meta


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
    ifo_whitelist: Optional[Sequence[str]] = None,
):
    from adapt.glitch_excision import time_bin_mask_to_windows
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

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


def _oracle_gates(meta: Dict[str, Any], cell: Dict[str, Any], gate_half_s: float):
    from adapt.glitch_excision import GateWindow

    t_peak = float(meta["t_peak_in_segment"])
    return [
        GateWindow(
            detector=str(d),
            t_start=t_peak - float(gate_half_s),
            t_end=t_peak + float(gate_half_s),
            score=1.0,
        )
        for d in cell["detectors"]
    ]


def _sample_dl(assets, event_data, settings, fixed, *, device, n, bs) -> Dict[str, float]:
    from evaluate_gw170817_comparison import run_baseline_sampling

    df = run_baseline_sampling(
        assets["baseline_ckpt"],
        SimpleNamespace(data=event_data, settings=settings),
        fixed,
        device=device,
        num_samples=int(n),
        batch_size=int(bs),
    )
    return _dl_ci(df["luminosity_distance"].to_numpy())


def _build_spec_stack(
    td_stft,
    *,
    asds_clean,
    detectors,
    sample_rate,
    delta_f,
    noise_std,
    norm_stats,
    stft_kwargs,
):
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )

    crops_w = whiten_td_map_with_asds(
        td_stft,
        asds_clean,
        sample_rate=sample_rate,
        delta_f=delta_f,
        noise_std=noise_std,
        detectors=detectors,
    )
    return build_robust_spectrogram_from_td(
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


# ---------------------------------------------------------------------------
# Main stress run
# ---------------------------------------------------------------------------


def write_config(outdir: Path, args: argparse.Namespace, extra: Dict[str, Any]) -> Dict[str, Any]:
    det = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    cfg = {
        "outdir": str(outdir),
        "master_seed": int(args.seed),
        "n_seeds_per_cell": int(args.n_seeds_per_cell),
        "num_samples": int(args.num_samples),
        "hf_samples": int(args.hf_samples),
        "batch_size": int(args.batch_size),
        "gate_half_s": float(args.gate_half_s),
        "families": list(FAMILIES),
        "severity_mids": list(SEVERITY_MIDS),
        "asd_policies": list(ASD_POLICIES),
        "n_cells_planned": 8 * 3 * 2 * int(args.n_seeds_per_cell),
        "detector_ckpt": str(det),
        "detector_sha256": _sha256(det),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "success_criteria": {
            "poison_collapsed": "d_L hi<15 or lo>90",
            "gated_recovers": "med in [20,50], CI overlaps clean, |med-clean_med|<=10",
        },
        "scope_limitation": (
            "GW170817 only (in-repo DINGO-BNS model). Synthetic glitch families; "
            "demo strain is LOSC_CLN (already LVK-cleaned)."
        ),
        "gold_control": str(DEFAULT_GOLD) if DEFAULT_GOLD.is_file() else None,
        **extra,
    }
    (outdir / "stress_config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    new_file = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata
    from evaluate_gw170817_comparison import (
        discover_assets,
        load_event_dataset,
        load_event_td_crops,
        select_device,
    )
    from train_bns_spectrogram import load_bns_checkpoint

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    logger.info("Device: %s outdir=%s", device, outdir)

    pe = (
        REPO_ROOT
        / "DINGO-BNS"
        / "dingo"
        / "binary-neutron-star-demo"
        / "GW170817"
        / "downloads"
        / "dingo-bns-model_GW170817.pt"
    )
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=pe if pe.is_file() else None,
    )
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = assets["fixed_context"]
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)
    asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    model, det_raw = _load_detector(det_path, device)
    threshold = float(det_raw.get("threshold", 0.5))
    gate_half_s = float(det_raw.get("gate_half_s", args.gate_half_s))
    norm_stats = det_raw.get("norm_stats")
    stft_kwargs = dict(det_raw.get("stft_kwargs") or {})

    cfg = write_config(
        outdir,
        args,
        {
            "baseline_ckpt": str(assets["baseline_ckpt"]),
            "baseline_sha256": _sha256(Path(assets["baseline_ckpt"])),
            "threshold_base": threshold,
            "gate_half_s_effective": gate_half_s,
        },
    )

    # ---- Clean reference PE (once) ----
    logger.info("===== Clean reference PE N=%d =====", args.num_samples)
    clean_ci = _sample_dl(
        assets,
        event.data,
        settings,
        fixed,
        device=device,
        n=args.num_samples,
        bs=args.batch_size,
    )
    logger.info("Clean d_L %s", clean_ci)
    (outdir / "clean_reference.json").write_text(json.dumps(clean_ci, indent=2))

    # Calibrate event threshold on clean STFT
    clean_crops = load_event_td_crops(
        assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
    )
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    crop_start = max(0, trig_idx - n_crop // 2)
    spec_c, _ = _build_spec_stack(
        clean_crops,
        asds_clean=asds_clean,
        detectors=detectors,
        sample_rate=sample_rate,
        delta_f=delta_f,
        noise_std=noise_std,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
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
    clean_gates, _ = _detector_gates(
        model,
        spec_c,
        detectors=detectors,
        crop_start=crop_start,
        sample_rate=sample_rate,
        threshold=thr_event,
        gate_half_s=gate_half_s,
        device=device,
    )
    logger.info(
        "Clean FP calibrate thr_event=%.3f n_gates=%d", thr_event, len(clean_gates)
    )
    cfg["threshold_event"] = thr_event
    cfg["clean_fp_gates"] = len(clean_gates)
    (outdir / "stress_config.json").write_text(json.dumps(cfg, indent=2))

    cells = build_locked_grid(
        master_seed=int(args.seed),
        n_seeds_per_cell=int(args.n_seeds_per_cell),
        max_cells=args.max_cells,
    )
    logger.info("Tier A cells: %d", len(cells))

    fieldnames = [
        "cell_id",
        "family",
        "held_out",
        "severity",
        "asd_policy",
        "seed",
        "detectors",
        "t_rel",
        "params_json",
        "poison_lo",
        "poison_med",
        "poison_hi",
        "poison_collapsed",
        "gated_lo",
        "gated_med",
        "gated_hi",
        "gated_recovers",
        "oracle_lo",
        "oracle_med",
        "oracle_hi",
        "oracle_recovers",
        "detector_n_gates",
        "detector_fired",
        "oracle_only_recovery",
        "residual_power_H1",
        "residual_power_L1",
        "error",
        "elapsed_s",
    ]
    csv_path = outdir / "results.csv"
    # Resume: skip completed cell_ids
    done_ids = set()
    if csv_path.is_file() and not args.overwrite:
        try:
            prev = pd.read_csv(csv_path)
            done_ids = set(int(x) for x in prev["cell_id"].tolist())
            logger.info("Resuming; %d cells already done", len(done_ids))
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    t0_all = time.time()
    for i, cell in enumerate(cells):
        if int(cell["cell_id"]) in done_ids:
            continue
        t0 = time.time()
        row: Dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "held_out": cell["held_out"],
            "severity": cell["severity"],
            "asd_policy": cell["asd_policy"],
            "seed": cell["seed"],
            "detectors": ",".join(cell["detectors"]),
            "t_rel": cell["t_rel"],
            "params_json": json.dumps(cell["params"]),
            "error": "",
        }
        try:
            poison, td_stft, meta = inject_spec_into_event(event, assets, cell)
            # Poison PE
            poison_ci = _sample_dl(
                assets,
                poison,
                settings,
                fixed,
                device=device,
                n=args.num_samples,
                bs=args.batch_size,
            )
            row.update(
                {
                    "poison_lo": poison_ci["lo"],
                    "poison_med": poison_ci["med"],
                    "poison_hi": poison_ci["hi"],
                    "poison_collapsed": _poison_collapsed(poison_ci),
                }
            )

            # Detector gates on glitchy STFT (clean ASD whitening)
            spec_g, _ = _build_spec_stack(
                meta["td_stft"],
                asds_clean=asds_clean,
                detectors=detectors,
                sample_rate=sample_rate,
                delta_f=delta_f,
                noise_std=noise_std,
                norm_stats=norm_stats,
                stft_kwargs=stft_kwargs,
            )
            whitelist = list(cell["detectors"])
            gates, probs = _detector_gates(
                model,
                spec_g,
                detectors=detectors,
                crop_start=int(meta["crop_start"]),
                sample_rate=sample_rate,
                threshold=thr_event,
                gate_half_s=gate_half_s,
                device=device,
                ifo_whitelist=whitelist,
            )
            row["detector_n_gates"] = len(gates)
            row["detector_fired"] = bool(len(gates) > 0)

            gated = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
            )
            rp = gated.meta.get("residual_power_frac") or {}
            row["residual_power_H1"] = float(rp.get("H1", 0.0) or 0.0)
            row["residual_power_L1"] = float(rp.get("L1", 0.0) or 0.0)

            gated_ci = _sample_dl(
                assets,
                gated.data,
                settings,
                fixed,
                device=device,
                n=args.num_samples,
                bs=args.batch_size,
            )
            row.update(
                {
                    "gated_lo": gated_ci["lo"],
                    "gated_med": gated_ci["med"],
                    "gated_hi": gated_ci["hi"],
                    "gated_recovers": _gated_recovers(gated_ci, clean_ci),
                }
            )

            # Oracle
            ogates = _oracle_gates(meta, cell, gate_half_s)
            oracle = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=ogates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
            )
            oracle_ci = _sample_dl(
                assets,
                oracle.data,
                settings,
                fixed,
                device=device,
                n=args.num_samples,
                bs=args.batch_size,
            )
            row.update(
                {
                    "oracle_lo": oracle_ci["lo"],
                    "oracle_med": oracle_ci["med"],
                    "oracle_hi": oracle_ci["hi"],
                    "oracle_recovers": _gated_recovers(oracle_ci, clean_ci),
                }
            )
            row["oracle_only_recovery"] = bool(
                row["oracle_recovers"] and not row["gated_recovers"]
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Cell %d failed: %s", cell["cell_id"], exc)
            traceback.print_exc()

        row["elapsed_s"] = round(time.time() - t0, 3)
        append_csv_row(csv_path, row, fieldnames)
        rows.append(row)
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(
                "[%d/%d] fam=%s sev=%.1f asd=%s gated_ok=%s poison_col=%s (%.1fs)",
                i + 1,
                len(cells),
                cell["family"],
                cell["severity"],
                cell["asd_policy"],
                row.get("gated_recovers"),
                row.get("poison_collapsed"),
                row["elapsed_s"],
            )

    # Reload full CSV for aggregates
    df = pd.read_csv(csv_path)
    logger.info("Tier A done: %d rows in %.1f min", len(df), (time.time() - t0_all) / 60.0)

    # ---- Tier C: clean FP audit (50 detector passes) ----
    logger.info("===== Tier C clean FP audit (50) =====")
    fp_hits = 0
    for j in range(50):
        # Deterministic clean spectrogram; optionally tiny float noise for seed j.
        spec_j = np.asarray(spec_c, dtype=np.float32).copy()
        if j > 0:
            rng = np.random.default_rng(int(args.seed) + 10_000 + j)
            spec_j = spec_j + rng.normal(scale=1e-6, size=spec_j.shape).astype(np.float32)
        g_j, _ = _detector_gates(
            model,
            spec_j,
            detectors=detectors,
            crop_start=crop_start,
            sample_rate=sample_rate,
            threshold=thr_event,
            gate_half_s=gate_half_s,
            device=device,
        )
        if len(g_j) > 0:
            fp_hits += 1
    tier_c = {
        "n": 50,
        "fp_fires": int(fp_hits),
        "fp_rate": float(fp_hits) / 50.0,
        "threshold_event": thr_event,
        "clean_d_L": clean_ci,
        "note": (
            "Pass 0 is exact clean spectrogram; passes 1–49 add 1e-6 Gaussian "
            "jitter to test threshold robustness."
        ),
    }
    (outdir / "tier_c_clean_fp.json").write_text(json.dumps(tier_c, indent=2))
    logger.info("Clean FP: %d/50 fires", fp_hits)

    # ---- Tier B: high-fidelity follow-ups ----
    logger.info("===== Tier B high-fidelity N=%d =====", args.hf_samples)
    tier_b: Dict[str, Any] = {"hf_samples": int(args.hf_samples), "cells": []}

    # Canonical SG
    canon = {
        "cell_id": "canonical_sg",
        "family": "sine_gaussian",
        "held_out": False,
        "severity": 8.0,
        "asd_policy": "welch",
        "seed": int(args.seed) + 999_001,
        "detectors": ["H1"],
        "t_rel": -1.0,
        "params": {"f0": 100.0, "q": 5.0},
    }
    follow = [canon]
    # Worst gated failures
    fail = df[df["gated_recovers"] == False] if "gated_recovers" in df.columns else df.iloc[0:0]
    if len(fail):
        fail2 = fail[fail["error"].fillna("") == ""].copy()
        if len(fail2):
            fail2["delta"] = (fail2["gated_med"] - clean_ci["med"]).abs()
            for _, r in fail2.nlargest(min(12, len(fail2)), "delta").iterrows():
                follow.append(
                    {
                        "cell_id": int(r["cell_id"]),
                        "family": r["family"],
                        "held_out": bool(r["held_out"]),
                        "severity": float(r["severity"]),
                        "asd_policy": r["asd_policy"],
                        "seed": int(r["seed"]),
                        "detectors": str(r["detectors"]).split(","),
                        "t_rel": float(r["t_rel"]),
                        "params": json.loads(r["params_json"]),
                        "tag": "worst_failure",
                    }
                )
    # Best held-out successes
    ok_ood = df[(df["held_out"] == True) & (df["gated_recovers"] == True)]
    if len(ok_ood):
        ok_ood = ok_ood.copy()
        ok_ood["delta"] = (ok_ood["gated_med"] - clean_ci["med"]).abs()
        for _, r in ok_ood.nsmallest(min(4, len(ok_ood)), "delta").iterrows():
            follow.append(
                {
                    "cell_id": int(r["cell_id"]),
                    "family": r["family"],
                    "held_out": True,
                    "severity": float(r["severity"]),
                    "asd_policy": r["asd_policy"],
                    "seed": int(r["seed"]),
                    "detectors": str(r["detectors"]).split(","),
                    "t_rel": float(r["t_rel"]),
                    "params": json.loads(r["params_json"]),
                    "tag": "best_held_out",
                }
            )

    for fb in follow:
        try:
            poison, td_stft, meta = inject_spec_into_event(event, assets, fb)
            poison_ci = _sample_dl(
                assets,
                poison,
                settings,
                fixed,
                device=device,
                n=args.hf_samples,
                bs=args.batch_size,
            )
            spec_g, _ = _build_spec_stack(
                meta["td_stft"],
                asds_clean=asds_clean,
                detectors=detectors,
                sample_rate=sample_rate,
                delta_f=delta_f,
                noise_std=noise_std,
                norm_stats=norm_stats,
                stft_kwargs=stft_kwargs,
            )
            gates, _ = _detector_gates(
                model,
                spec_g,
                detectors=detectors,
                crop_start=int(meta["crop_start"]),
                sample_rate=sample_rate,
                threshold=thr_event,
                gate_half_s=gate_half_s,
                device=device,
                ifo_whitelist=list(fb["detectors"]),
            )
            gated = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
            )
            gated_ci = _sample_dl(
                assets,
                gated.data,
                settings,
                fixed,
                device=device,
                n=args.hf_samples,
                bs=args.batch_size,
            )
            entry = {
                **{k: v for k, v in fb.items() if k != "params"},
                "params": fb["params"],
                "poison_d_L": poison_ci,
                "gated_d_L": gated_ci,
                "poison_collapsed": _poison_collapsed(poison_ci),
                "gated_recovers": _gated_recovers(gated_ci, clean_ci),
                "n_gates": len(gates),
            }
            tier_b["cells"].append(entry)
            logger.info(
                "HF %s fam=%s gated=%s",
                fb.get("tag", fb["cell_id"]),
                fb["family"],
                entry["gated_recovers"],
            )
        except Exception as exc:
            tier_b["cells"].append(
                {
                    "cell_id": fb.get("cell_id"),
                    "tag": fb.get("tag"),
                    "family": fb.get("family"),
                    "error": repr(exc) or type(exc).__name__,
                }
            )
            logger.exception("Tier B cell failed")

    if DEFAULT_GOLD.is_file():
        tier_b["gold_control_20k_is"] = json.loads(DEFAULT_GOLD.read_text())
    (outdir / "tier_b_results.json").write_text(json.dumps(tier_b, indent=2, default=str))

    # ---- Aggregates + PDF + REPRODUCE ----
    write_summaries_and_report(outdir, df, clean_ci, tier_c, tier_b, cfg)
    logger.info("Stress test complete → %s", outdir)


def write_summaries_and_report(
    outdir: Path,
    df: pd.DataFrame,
    clean_ci: Dict[str, float],
    tier_c: Dict[str, Any],
    tier_b: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    ok = df[df["error"].fillna("") == ""].copy() if "error" in df.columns else df.copy()
    # coerce bools
    for col in ("held_out", "poison_collapsed", "gated_recovers", "oracle_recovers"):
        if col in ok.columns:
            ok[col] = ok[col].astype(bool)

    failures = ok[ok["gated_recovers"] == False]
    failures.to_csv(outdir / "failures.csv", index=False)

    def _rate(sub, col):
        if len(sub) == 0:
            return None
        return float(sub[col].mean())

    summary = {
        "n_rows": int(len(df)),
        "n_ok": int(len(ok)),
        "n_errors": int((df["error"].fillna("") != "").sum()) if "error" in df.columns else 0,
        "clean_reference_d_L": clean_ci,
        "overall": {
            "gated_recovery_rate": _rate(ok, "gated_recovers"),
            "oracle_recovery_rate": _rate(ok, "oracle_recovers"),
            "poison_collapse_rate": _rate(ok, "poison_collapsed"),
            "mean_abs_delta_med_d_L_successes": float(
                (ok.loc[ok["gated_recovers"], "gated_med"] - clean_ci["med"]).abs().mean()
            )
            if ok["gated_recovers"].any()
            else None,
        },
        "by_held_out": ok.groupby("held_out")["gated_recovers"]
        .mean()
        .astype(float)
        .to_dict()
        if len(ok)
        else {},
        "by_family": ok.groupby("family")
        .agg(
            n=("gated_recovers", "size"),
            gated_recovery=("gated_recovers", "mean"),
            oracle_recovery=("oracle_recovers", "mean"),
            poison_collapse=("poison_collapsed", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
        if len(ok)
        else [],
        "by_severity": ok.groupby("severity")["gated_recovers"].mean().astype(float).to_dict()
        if len(ok)
        else {},
        "by_asd_policy": {
            "poison_collapse": ok.groupby("asd_policy")["poison_collapsed"]
            .mean()
            .astype(float)
            .to_dict()
            if len(ok)
            else {},
            "gated_recovery": ok.groupby("asd_policy")["gated_recovers"]
            .mean()
            .astype(float)
            .to_dict()
            if len(ok)
            else {},
        },
        "tier_c_clean_fp": tier_c,
        "n_failures": int(len(failures)),
        "config_ref": "stress_config.json",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # PDF
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = outdir / "stress_test_report.pdf"
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")
            lines = [
                "Detect-and-gate stress test (GW170817, frozen DINGO)",
                f"N cells (ok): {summary['n_ok']}  errors: {summary['n_errors']}",
                f"Gated recovery: {summary['overall']['gated_recovery_rate']}",
                f"Oracle recovery: {summary['overall']['oracle_recovery_rate']}",
                f"Poison collapse: {summary['overall']['poison_collapse_rate']}",
                f"Clean FP rate: {tier_c['fp_rate']} ({tier_c['fp_fires']}/50)",
                f"Clean d_L: {clean_ci}",
                "",
                "Scope: synthetic families on LOSC_CLN GW170817; no GW190425 model in-repo.",
            ]
            ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            if summary["by_family"]:
                fam = pd.DataFrame(summary["by_family"])
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(fam))
                ax.bar(x - 0.2, fam["gated_recovery"], width=0.4, label="gated")
                ax.bar(x + 0.2, fam["poison_collapse"], width=0.4, label="poison collapse")
                ax.set_xticks(x)
                ax.set_xticklabels(fam["family"], rotation=30, ha="right")
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("rate")
                ax.legend()
                ax.set_title("Recovery / collapse by family")
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

            if len(ok):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(
                    ok["poison_med"],
                    ok["gated_med"],
                    c=ok["held_out"].map({True: "C1", False: "C0"}),
                    alpha=0.7,
                    s=28,
                )
                ax.axhline(clean_ci["med"], color="k", ls="--", label="clean med")
                ax.axvline(10, color="gray", ls=":", alpha=0.5)
                ax.set_xlabel("poison d_L med")
                ax.set_ylabel("gated d_L med")
                ax.legend()
                ax.set_title("Poison vs gated d_L medians")
                fig.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
        logger.info("Wrote %s", pdf_path)
    except Exception as exc:
        logger.warning("PDF failed: %s", exc)

    repro = f"""# Reproduce detect-and-gate stress test

Event: **GW170817** only (in-repo DINGO-BNS Zenodo model).
Glitches: synthetic families from `adapt.glitch_augmentation` (held-in + held-out).
PE: frozen official DINGO; gated path = detect-and-gate + original ASD.

## Environment

```bash
conda activate adapt_env
cd {REPO_ROOT}
export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
```

## Run

```bash
python scripts/stress_test_glitch_excision.py \\
  --seed {cfg.get('master_seed', 0)} --n-seeds-per-cell {cfg.get('n_seeds_per_cell', 5)} \\
  --num-samples {cfg.get('num_samples', 512)} --hf-samples {cfg.get('hf_samples', 2000)} \\
  --outdir {outdir}
```

Resume is automatic if `results.csv` exists (skip completed `cell_id`s).
Use `--overwrite` to restart.

## Outputs

- `stress_config.json` — locked recipe + ckpt hashes
- `results.csv` — per-cell metrics
- `summary.json` — aggregates
- `failures.csv` — gated non-recoveries
- `tier_b_results.json` / `tier_c_clean_fp.json`
- `stress_test_report.pdf`
- Gold 20k+IS control (canonical SG): `results/dingo_official_control/`

## Success criteria

- Poison collapsed: `d_L` hi < 15 or lo > 90
- Gated recovers: med in [20, 50], CI overlaps clean, |med − clean_med| ≤ 10 Mpc
"""
    (outdir / "REPRODUCE.md").write_text(repro)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds-per-cell", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--hf-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--max-cells", type=int, default=None, help="Cap cells (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.overwrite:
        out = Path(args.outdir)
        for name in ("results.csv",):
            p = out / name
            if p.is_file():
                p.unlink()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
