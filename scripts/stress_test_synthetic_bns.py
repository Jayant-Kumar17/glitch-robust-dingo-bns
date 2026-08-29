#!/usr/bin/env python3
"""Synthetic BNS ADAPT stress: frozen DINGO + detect-and-gate on injected events.

Draws GW170817-conditioned CBC injections, runs a clean known-answer check,
then a locked 8-cell glitch panel (poison vs detector-gated). No oracle arm.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
    python -u scripts/stress_test_synthetic_bns.py \\
      --seed 0 --n-events 20 --num-samples 512 --batch-size 256 --device cpu \\
      --outdir results/stress_test_synthetic_bns_v1
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

DEFAULT_OUTDIR = REPO_ROOT / "results" / "stress_test_synthetic_bns_v1"
DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)
DEFAULT_BASELINE = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "downloads"
    / "dingo-bns-model_GW170817.pt"
)

# Tight panel for tonight: 4 families × 2 severities × Welch × 1 seed = 8 cells/event
PANEL_FAMILIES = (
    "sine_gaussian",
    "broadband_burst",
    "scattered_light",
    "ringing",
)
PANEL_SEVERITIES = (6.0, 10.0)

logger = logging.getLogger("stress_test_synthetic_bns")


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
    """Recover relative to this event's clean PE (synthetics vary d_L)."""
    if not _ci_overlap(ci, clean):
        return False
    if abs(ci["med"] - clean["med"]) > 10.0:
        return False
    # Keep med near the clean answer (not a global GW170817-only band).
    if abs(ci["med"] - clean["med"]) > 0.35 * max(abs(clean["med"]), 1.0):
        return False
    return True


def known_answer_pass(
    df: pd.DataFrame,
    truth: Dict[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    dl = _dl_ci(df["luminosity_distance"].to_numpy())
    if "chirp_mass" in df.columns:
        mc_med = float(np.median(df["chirp_mass"].to_numpy()))
    elif "delta_chirp_mass" in df.columns:
        mc_med = float(np.median(df["delta_chirp_mass"].to_numpy())) + float(
            truth["chirp_mass_proxy"]
        )
    else:
        mc_med = float("nan")
    mc_ok = abs(mc_med - float(truth["chirp_mass"])) <= 0.05
    # Strict: median must land near truth (exclude injections this network can't do).
    dl_ok = abs(dl["med"] - float(truth["luminosity_distance"])) <= 12.0
    info = {
        "clean_d_L": dl,
        "clean_chirp_mass_med": mc_med,
        "kat_mc_ok": bool(mc_ok),
        "kat_dl_ok": bool(dl_ok),
        "kat_pass": bool(mc_ok and dl_ok),
    }
    return bool(mc_ok and dl_ok), info


def _calibrate_thr_event(
    threshold: float,
    clean_probs: np.ndarray,
) -> float:
    """Avoid thr>=1 (detector can never fire) on domain-shifted synthetics."""
    peak = float(np.max(clean_probs)) if clean_probs.size else 0.0
    raw = max(float(threshold), peak + 0.05)
    if raw >= 0.99:
        logger.warning(
            "Clean synthetic STFT saturates detector (peak=%.3f); "
            "using checkpoint threshold %.3f",
            peak,
            threshold,
        )
        return float(threshold)
    return float(raw)

def _sample_family_params(family: str, rng: np.random.Generator) -> Dict[str, Any]:
    from stress_test_glitch_excision import _sample_family_params as _sfp

    return _sfp(family, rng)


def build_panel_cells(
    *,
    event_id: int,
    master_seed: int,
    max_cells: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from adapt.glitch_augmentation import HELD_OUT_FAMILIES

    held_out_set = set(HELD_OUT_FAMILIES)
    cells: List[Dict[str, Any]] = []
    cell_id = 0
    for family in PANEL_FAMILIES:
        for sev in PANEL_SEVERITIES:
            seed = int(master_seed) + 10_000 * int(event_id) + 97 * cell_id
            rng = np.random.default_rng(seed)
            t_rel = float(rng.uniform(-1.5, -0.3))
            params = _sample_family_params(family, rng)
            cells.append(
                {
                    "event_id": int(event_id),
                    "cell_id": cell_id,
                    "family": family,
                    "held_out": family in held_out_set,
                    "severity": float(sev),
                    "asd_policy": "welch",
                    "seed": seed,
                    "detectors": ["H1"],
                    "t_rel": t_rel,
                    "params": params,
                }
            )
            cell_id += 1
            if max_cells is not None and cell_id >= int(max_cells):
                return cells
    return cells


def draw_synthetic_theta(
    injection,
    rng: np.random.Generator,
    *,
    fixed_sky: Dict[str, float],
) -> Dict[str, float]:
    """GW170817-conditioned prior slice for the frozen demo network."""
    theta = {k: float(v) for k, v in injection.prior.sample().items()}
    # Stay close to the demo event the network was packaged for.
    theta["chirp_mass"] = float(rng.uniform(1.190, 1.205))
    theta["mass_ratio"] = float(rng.uniform(0.85, 1.0))
    theta["luminosity_distance"] = float(rng.uniform(28.0, 42.0))
    # Prefer face-on-ish to edge-on like GW170817 (high inclination → large d_L).
    theta["theta_jn"] = float(rng.uniform(2.0, np.pi))  # near edge-on like GW170817
    theta["ra"] = float(fixed_sky["ra"])
    theta["dec"] = float(fixed_sky["dec"])
    theta["geocent_time"] = 0.0
    # Mild spins / tides — avoid extreme prior draws that break GNPE.
    for k in ("a_1", "a_2"):
        if k in theta:
            theta[k] = float(rng.uniform(0.0, 0.05))
    for k in ("lambda_1", "lambda_2"):
        if k in theta:
            theta[k] = float(rng.uniform(0.0, 1000.0))
    mc = float(theta["chirp_mass"])
    theta["chirp_mass_proxy"] = mc + 1e-4 * mc * float(rng.normal())
    theta["delta_chirp_mass"] = mc - float(theta["chirp_mass_proxy"])
    return theta


def fd_to_analysis_td(
    fd: np.ndarray,
    *,
    sample_rate: float,
    duration: float,
    time_buffer: float,
    trigger_time: float = 0.0,
) -> np.ndarray:
    """IFFT FD → TD with trigger at ``duration - time_buffer`` (demo segment)."""
    n_td = int(round(float(duration) * float(sample_rate)))
    n_rfft = n_td // 2 + 1
    fd_arr = np.asarray(fd, dtype=np.complex128).ravel()
    fd_full = np.zeros(n_rfft, dtype=np.complex128)
    n_copy = min(fd_arr.size, n_rfft)
    fd_full[:n_copy] = fd_arr[:n_copy]
    td = np.fft.irfft(fd_full, n=n_td) * float(sample_rate)
    # First put trigger at mid (common injection convention), then to analysis.
    trig_idx = int(round(float(trigger_time) * float(sample_rate))) % n_td
    td = np.roll(td, n_td // 2 - trig_idx)
    target = int(round((float(duration) - float(time_buffer)) * float(sample_rate)))
    td = np.roll(td, target - n_td // 2)
    return td.astype(np.float64)


def build_synthetic_event(
    injection,
    theta: Dict[str, float],
    *,
    asds_ref: Dict[str, np.ndarray],
    settings: Dict[str, Any],
    detectors: Sequence[str],
) -> Tuple[Any, Dict[str, np.ndarray], Dict[str, float]]:
    """CBC inject with ref ASDs → EventDataset + analysis-segment TD."""
    from dingo.gw.data.event_dataset import EventDataset

    # Force GW170817 ASDs onto the injector for this draw.
    injection.asd = {d: np.asarray(asds_ref[d], dtype=np.float64).copy() for d in detectors}
    noise_seed = int(abs(hash(tuple(sorted((k, round(float(v), 12) if isinstance(v, float) else v) for k, v in theta.items())))) % (2**31 - 1))
    np.random.seed(noise_seed)
    try:
        import bilby

        bilby.core.utils.random.seed(noise_seed)
    except Exception:
        pass
    inj = injection.injection(theta)
    params = dict(inj.get("parameters") or {})
    geocent = float(params.get("geocent_time", theta.get("geocent_time", 0.0)))

    data = {
        "waveform": {
            d: np.asarray(inj["waveform"][d], dtype=np.complex128).copy() for d in detectors
        },
        "asds": {
            d: np.asarray(inj["asds"][d], dtype=np.float64).copy() for d in detectors
        },
    }
    # Prefer ref ASDs exactly (stationary off-source).
    for d in detectors:
        data["asds"][d] = np.asarray(asds_ref[d], dtype=np.float64).copy()

    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    sample_rate = float(settings.get("f_s", 4096.0))
    td_full: Dict[str, np.ndarray] = {}
    for d in detectors:
        trig = float(params.get(f"{d}_time", geocent))
        td_full[d] = fd_to_analysis_td(
            data["waveform"][d],
            sample_rate=sample_rate,
            duration=duration,
            time_buffer=time_buffer,
            trigger_time=trig,
        )

    event = EventDataset(
        dictionary={"data": data, "settings": dict(settings)}
    )
    truth = {
        "chirp_mass": float(theta["chirp_mass"]),
        "chirp_mass_proxy": float(theta["chirp_mass_proxy"]),
        "mass_ratio": float(theta["mass_ratio"]),
        "luminosity_distance": float(theta["luminosity_distance"]),
        "theta_jn": float(theta["theta_jn"]),
        "ra": float(theta["ra"]),
        "dec": float(theta["dec"]),
        "geocent_time": float(geocent),
    }
    return event, td_full, truth


def inject_glitch_on_synthetic(
    event_data: Dict[str, Any],
    td_clean: Dict[str, np.ndarray],
    cell: Dict[str, Any],
    settings: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Glitch overlay on in-memory synthetic TD/FD (no GWF)."""
    from adapt.glitch_augmentation import GlitchSpec, synthesize_glitch_td
    from adapt.stft_context import inband_rms
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from evaluate_glitch_robustness import td_to_fd_strain, welch_asd

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
    t_peak = (duration - time_buffer) + float(spec.t_rel)
    t_rel_for_synth = t_peak - 0.5 * duration

    data = copy.deepcopy(event_data)
    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))
    td_full: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(cell["seed"]))

    for det in ("H1", "L1", "V1"):
        if det not in td_clean:
            continue
        td = np.asarray(td_clean[det], dtype=np.float64).copy()
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
            g = synthesize_glitch_td(len(td), sample_rate, spec_synth, rng, rms=rms)
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
            td_full[det] = td

    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    td_stft = {}
    for det, x in td_full.items():
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
    from stress_test_glitch_excision import _load_detector as _ld

    return _ld(path, device)


def _detector_gates(*args, **kwargs):
    from stress_test_glitch_excision import _detector_gates as _dg

    return _dg(*args, **kwargs)


def _build_spec_stack(*args, **kwargs):
    from stress_test_glitch_excision import _build_spec_stack as _bs

    return _bs(*args, **kwargs)


def _sample_posterior(
    assets,
    event_data,
    settings,
    fixed,
    *,
    device,
    n,
    bs,
) -> pd.DataFrame:
    from evaluate_gw170817_comparison import run_baseline_sampling

    return run_baseline_sampling(
        assets["baseline_ckpt"],
        SimpleNamespace(data=event_data, settings=settings),
        fixed,
        device=device,
        num_samples=int(n),
        batch_size=int(bs),
    )


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    new_file = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def write_artifacts(
    outdir: Path,
    events_df: pd.DataFrame,
    results_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ok_ev = events_df.copy()
    if "kat_pass" in ok_ev.columns:
        ok_ev["kat_pass"] = ok_ev["kat_pass"].astype(bool)
    kat_rate = float(ok_ev["kat_pass"].mean()) if len(ok_ev) else 0.0

    res = results_df.copy()
    if len(res) and "error" in res.columns:
        ok = res[res["error"].fillna("") == ""].copy()
    else:
        ok = res.copy()
    for col in ("held_out", "poison_collapsed", "gated_recovers", "detector_fired"):
        if col in ok.columns:
            ok[col] = ok[col].map(lambda x: bool(x) if pd.notna(x) else False)

    failures = ok[ok["gated_recovers"] == False] if len(ok) else ok
    failures.to_csv(outdir / "failures.csv", index=False)

    def _rate(sub, col):
        if len(sub) == 0 or col not in sub.columns:
            return None
        return float(sub[col].mean())

    mean_delta = None
    if len(ok) and ok["gated_recovers"].any():
        succ = ok[ok["gated_recovers"]]
        mean_delta = float((succ["gated_med"] - succ["clean_med"]).abs().mean())

    summary = {
        "n_events": int(len(events_df)),
        "kat_pass_rate": kat_rate,
        "n_glitch_rows": int(len(results_df)),
        "n_glitch_ok": int(len(ok)),
        "overall": {
            "gated_recovery_rate": _rate(ok, "gated_recovers"),
            "poison_collapse_rate": _rate(ok, "poison_collapsed"),
            "mean_abs_delta_med_d_L_successes": mean_delta,
            "detector_fire_rate": _rate(ok, "detector_fired"),
        },
        "by_held_out": ok.groupby("held_out")["gated_recovers"].mean().astype(float).to_dict()
        if len(ok)
        else {},
        "by_family": ok.groupby("family")
        .agg(
            n=("gated_recovers", "size"),
            gated_recovery=("gated_recovers", "mean"),
            poison_collapse=("poison_collapsed", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
        if len(ok)
        else [],
        "by_severity": ok.groupby("severity")["gated_recovers"].mean().astype(float).to_dict()
        if len(ok)
        else {},
        "n_failures": int(len(failures)),
        "config_ref": "synth_config.json",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = outdir / "stress_test_report.pdf"
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(8, 4))
            if len(ok):
                fam = (
                    ok.groupby("family")["gated_recovers"]
                    .mean()
                    .reindex(PANEL_FAMILIES)
                )
                ax.bar([str(x) for x in fam.index], fam.values.astype(float))
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Gated recovery rate")
                ax.set_title(
                    f"Synthetic BNS ADAPT — gated recovery "
                    f"(overall={summary['overall']['gated_recovery_rate']:.2f})"
                )
                plt.xticks(rotation=20, ha="right")
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            if len(ok):
                ax.scatter(ok["poison_med"], ok["gated_med"], alpha=0.6, s=20)
                lims = [0, 120]
                ax.plot(lims, lims, "k--", lw=0.8)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_xlabel("Poison d_L med")
                ax.set_ylabel("Gated d_L med")
                ax.set_title("Poison vs gated luminosity distance")
            pdf.savefig(fig)
            plt.close(fig)
        logger.info("Wrote %s", pdf_path)
    except Exception:
        logger.exception("PDF report failed")

    (outdir / "REPRODUCE.md").write_text(
        "\n".join(
            [
                "# Reproduce synthetic BNS ADAPT stress",
                "",
                "Frozen GW170817 DINGO + detect-and-gate on synthetic CBC injections.",
                "",
                "## Environment",
                "",
                "```bash",
                "conda activate adapt_env",
                f"cd {REPO_ROOT}",
                "export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE",
                "```",
                "",
                "## Run",
                "",
                "```bash",
                "python -u scripts/stress_test_synthetic_bns.py \\",
                f"  --seed {cfg.get('master_seed', 0)} --n-events {cfg.get('n_events', 20)} \\",
                f"  --num-samples {cfg.get('num_samples', 512)} --batch-size {cfg.get('batch_size', 256)} \\",
                "  --device cpu \\",
                f"  --outdir {outdir}",
                "```",
                "",
                "Resume skips completed `(event_id, cell_id)` pairs. Use `--overwrite` to restart.",
                "",
                "## Outputs",
                "",
                "- `synth_config.json`, `events.csv`, `results.csv`, `summary.json`,",
                "  `failures.csv`, `stress_test_report.pdf`",
                "",
            ]
        )
    )
    return summary


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata
    from evaluate_gw170817_comparison import (
        discover_assets,
        load_event_dataset,
        select_device,
    )
    from train_bns_spectrogram import build_base_domain_injection, load_bns_checkpoint

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    logger.info("Device: %s outdir=%s", device, outdir)

    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else DEFAULT_BASELINE,
        custom_ckpt=None,
    )
    ref_event = load_event_dataset(assets)
    settings = dict(ref_event.settings)
    demo_fixed = dict(assets["fixed_context"])
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)
    asds_ref = {d: np.asarray(ref_event.data["asds"][d]).copy() for d in detectors}
    # Align ASD length to base domain if needed.
    n_base = len(base_domain)
    for d in detectors:
        a = asds_ref[d]
        if len(a) < n_base:
            asds_ref[d] = np.pad(a, (0, n_base - len(a)), constant_values=1.0)
        elif len(a) > n_base:
            asds_ref[d] = a[:n_base]

    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))

    injection, _base = build_base_domain_injection(metadata)

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    model, det_raw = _load_detector(det_path, device)
    threshold = float(det_raw.get("threshold", 0.5))
    gate_half_s = float(det_raw.get("gate_half_s", args.gate_half_s))
    norm_stats = det_raw.get("norm_stats")
    stft_kwargs = dict(det_raw.get("stft_kwargs") or {})

    cfg = {
        "outdir": str(outdir),
        "master_seed": int(args.seed),
        "n_events": int(args.n_events),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "gate_half_s": float(gate_half_s),
        "panel_families": list(PANEL_FAMILIES),
        "panel_severities": list(PANEL_SEVERITIES),
        "cells_per_event": len(PANEL_FAMILIES) * len(PANEL_SEVERITIES),
        "detector_ckpt": str(det_path),
        "detector_sha256": _sha256(det_path),
        "baseline_ckpt": str(assets["baseline_ckpt"]),
        "baseline_sha256": _sha256(Path(assets["baseline_ckpt"])),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "scope": (
            "Synthetic CBC injections conditioned for frozen GW170817 DINGO; "
            "ASDs copied from GW170817 package; detect-and-gate ADAPT path."
        ),
        "success_criteria": {
            "kat": "|med chirp_mass - truth|<=0.05 and |med d_L - truth|<=12",
            "poison_collapsed": "d_L hi<15 or lo>90",
            "gated_recovers": "CI overlaps clean and |med-clean_med|<=10",
        },
        "threshold_base": threshold,
    }
    (outdir / "synth_config.json").write_text(json.dumps(cfg, indent=2))

    events_path = outdir / "events.csv"
    results_path = outdir / "results.csv"
    event_fields = [
        "event_id",
        "kat_pass",
        "kat_mc_ok",
        "kat_dl_ok",
        "truth_chirp_mass",
        "truth_d_L",
        "truth_q",
        "proxy",
        "clean_lo",
        "clean_med",
        "clean_hi",
        "clean_chirp_mass_med",
        "thr_event",
        "error",
        "elapsed_s",
    ]
    result_fields = [
        "event_id",
        "cell_id",
        "family",
        "held_out",
        "severity",
        "asd_policy",
        "seed",
        "detectors",
        "t_rel",
        "params_json",
        "clean_med",
        "poison_lo",
        "poison_med",
        "poison_hi",
        "poison_collapsed",
        "gated_lo",
        "gated_med",
        "gated_hi",
        "gated_recovers",
        "detector_n_gates",
        "detector_fired",
        "error",
        "elapsed_s",
    ]

    done_events: set = set()
    done_cells: set = set()
    if events_path.is_file() and not args.overwrite:
        edf0 = pd.read_csv(events_path)
        done_events = set(int(x) for x in edf0["event_id"].tolist())
    if results_path.is_file() and not args.overwrite:
        rdf0 = pd.read_csv(results_path)
        done_cells = set(
            (int(r.event_id), int(r.cell_id)) for r in rdf0.itertuples(index=False)
        )
    if args.overwrite:
        for p in (events_path, results_path, outdir / "failures.csv", outdir / "summary.json"):
            if p.is_file():
                p.unlink()

    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    crop_start_nom = max(0, trig_idx - n_crop // 2)

    # Oversample draws until we have ``n_events`` KAT-pass events (or hit cap).
    target_kat = int(args.n_events)
    max_attempts = max(target_kat * 4, target_kat)
    n_kat_pass = 0
    if events_path.is_file() and not args.overwrite:
        edf_exist = pd.read_csv(events_path)
        if "kat_pass" in edf_exist.columns:
            n_kat_pass = int(edf_exist["kat_pass"].astype(bool).sum())

    for event_id in range(max_attempts):
        if n_kat_pass >= target_kat:
            logger.info("Reached %d KAT-pass events — stopping", target_kat)
            break
        t_ev0 = time.time()
        # Deterministic per event_id so resume rebuilds the same injection.
        ev_seed = int(args.seed) + 1_000_003 * int(event_id)
        rng = np.random.default_rng(ev_seed)

        # Always need event object for cells; rebuild even if event row exists.
        theta = draw_synthetic_theta(injection, rng, fixed_sky=demo_fixed)
        fixed = {
            "chirp_mass_proxy": float(theta["chirp_mass_proxy"]),
            "ra": float(demo_fixed["ra"]),
            "dec": float(demo_fixed["dec"]),
        }
        try:
            event, td_clean, truth = build_synthetic_event(
                injection,
                theta,
                asds_ref=asds_ref,
                settings=settings,
                detectors=detectors,
            )
            asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}

            if event_id not in done_events:
                logger.info(
                    "===== Event %d/%d  Mc=%.4f d_L=%.1f =====",
                    event_id + 1,
                    max_attempts,
                    truth["chirp_mass"],
                    truth["luminosity_distance"],
                )
                df_clean = _sample_posterior(
                    assets,
                    event.data,
                    settings,
                    fixed,
                    device=device,
                    n=args.num_samples,
                    bs=args.batch_size,
                )
                kat_ok, kat_info = known_answer_pass(df_clean, truth)
                clean_ci = kat_info["clean_d_L"]

                # Event-calibrated detector threshold on clean synthetic STFT
                td_stft_c = {}
                for det in detectors:
                    x = td_clean[det]
                    end = crop_start_nom + n_crop
                    if end > len(x):
                        start_i = len(x) - n_crop
                        end_i = len(x)
                    else:
                        start_i, end_i = crop_start_nom, end
                    td_stft_c[det] = x[start_i:end_i].copy()
                spec_c, _ = _build_spec_stack(
                    td_stft_c,
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
                    crop_start=crop_start_nom,
                    sample_rate=sample_rate,
                    threshold=threshold,
                    gate_half_s=gate_half_s,
                    device=device,
                )
                thr_event = _calibrate_thr_event(threshold, clean_probs)

                append_csv_row(
                    events_path,
                    {
                        "event_id": event_id,
                        "kat_pass": kat_ok,
                        "kat_mc_ok": kat_info["kat_mc_ok"],
                        "kat_dl_ok": kat_info["kat_dl_ok"],
                        "truth_chirp_mass": truth["chirp_mass"],
                        "truth_d_L": truth["luminosity_distance"],
                        "truth_q": truth["mass_ratio"],
                        "proxy": truth["chirp_mass_proxy"],
                        "clean_lo": clean_ci["lo"],
                        "clean_med": clean_ci["med"],
                        "clean_hi": clean_ci["hi"],
                        "clean_chirp_mass_med": kat_info["clean_chirp_mass_med"],
                        "thr_event": thr_event,
                        "error": "",
                        "elapsed_s": round(time.time() - t_ev0, 3),
                    },
                    event_fields,
                )
                done_events.add(event_id)
                if kat_ok:
                    n_kat_pass += 1
                logger.info(
                    "Event %d KAT=%s clean_d_L=%s thr=%.3f (kat_pass_total=%d/%d)",
                    event_id,
                    kat_ok,
                    clean_ci,
                    thr_event,
                    n_kat_pass,
                    target_kat,
                )
            else:
                edf = pd.read_csv(events_path)
                row = edf[edf["event_id"] == event_id].iloc[0]
                kat_ok = bool(row["kat_pass"])
                clean_ci = {
                    "lo": float(row["clean_lo"]),
                    "med": float(row["clean_med"]),
                    "hi": float(row["clean_hi"]),
                    "n": int(args.num_samples),
                }
                thr_event = float(row["thr_event"])
                if kat_ok and event_id not in done_events:
                    pass  # n_kat_pass already counted from file
                logger.info("Event %d resumed (kat=%s)", event_id, kat_ok)

            if not kat_ok:
                logger.warning("Event %d failed KAT — skipping glitch panel", event_id)
                continue

            cells = build_panel_cells(
                event_id=event_id,
                master_seed=int(args.seed),
                max_cells=args.max_cells,
            )
            for cell in cells:
                key = (event_id, int(cell["cell_id"]))
                if key in done_cells:
                    continue
                t0 = time.time()
                row: Dict[str, Any] = {
                    "event_id": event_id,
                    "cell_id": cell["cell_id"],
                    "family": cell["family"],
                    "held_out": cell["held_out"],
                    "severity": cell["severity"],
                    "asd_policy": cell["asd_policy"],
                    "seed": cell["seed"],
                    "detectors": ",".join(cell["detectors"]),
                    "t_rel": cell["t_rel"],
                    "params_json": json.dumps(cell["params"]),
                    "clean_med": clean_ci["med"],
                    "error": "",
                }
                try:
                    poison, td_stft, meta = inject_glitch_on_synthetic(
                        event.data, td_clean, cell, settings
                    )
                    poison_df = _sample_posterior(
                        assets,
                        poison,
                        settings,
                        fixed,
                        device=device,
                        n=args.num_samples,
                        bs=args.batch_size,
                    )
                    poison_ci = _dl_ci(poison_df["luminosity_distance"].to_numpy())

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
                        ifo_whitelist=list(cell["detectors"]),
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
                    gated_df = _sample_posterior(
                        assets,
                        gated.data,
                        settings,
                        fixed,
                        device=device,
                        n=args.num_samples,
                        bs=args.batch_size,
                    )
                    gated_ci = _dl_ci(gated_df["luminosity_distance"].to_numpy())

                    row.update(
                        {
                            "poison_lo": poison_ci["lo"],
                            "poison_med": poison_ci["med"],
                            "poison_hi": poison_ci["hi"],
                            "poison_collapsed": _poison_collapsed(poison_ci),
                            "gated_lo": gated_ci["lo"],
                            "gated_med": gated_ci["med"],
                            "gated_hi": gated_ci["hi"],
                            "gated_recovers": _gated_recovers(gated_ci, clean_ci),
                            "detector_n_gates": len(gates),
                            "detector_fired": len(gates) > 0,
                            "elapsed_s": round(time.time() - t0, 3),
                        }
                    )
                    logger.info(
                        "[e%d c%d] fam=%s sev=%.0f gated=%s poison_col=%s (%.1fs)",
                        event_id,
                        cell["cell_id"],
                        cell["family"],
                        cell["severity"],
                        row["gated_recovers"],
                        row["poison_collapsed"],
                        row["elapsed_s"],
                    )
                except Exception as exc:
                    row["error"] = repr(exc) or type(exc).__name__
                    row["elapsed_s"] = round(time.time() - t0, 3)
                    logger.exception(
                        "Cell failed event=%d cell=%d", event_id, cell["cell_id"]
                    )
                append_csv_row(results_path, row, result_fields)
                done_cells.add(key)

        except Exception as exc:
            if event_id not in done_events:
                append_csv_row(
                    events_path,
                    {
                        "event_id": event_id,
                        "kat_pass": False,
                        "error": repr(exc) or type(exc).__name__,
                        "elapsed_s": round(time.time() - t_ev0, 3),
                    },
                    event_fields,
                )
            logger.exception("Event %d failed", event_id)

    events_df = (
        pd.read_csv(events_path) if events_path.is_file() else pd.DataFrame()
    )
    results_df = (
        pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    )
    summary = write_artifacts(outdir, events_df, results_df, cfg)
    logger.info(
        "Done. KAT=%.2f gated_recovery=%s → %s",
        summary.get("kat_pass_rate"),
        summary.get("overall", {}).get("gated_recovery_rate"),
        outdir,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--baseline-ckpt", type=str, default=None)
    p.add_argument("--detector-ckpt", type=str, default=None)
    p.add_argument("--max-cells", type=int, default=None, help="Cap glitch cells per event")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
