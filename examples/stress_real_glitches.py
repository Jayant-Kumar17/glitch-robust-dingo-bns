#!/usr/bin/env python3
"""Real Gravity Spy glitch panel on GW170817 (frozen DINGO-BNS).

Real O3 H1 glitch excerpts (Gravity Spy, Glanzer et al. 2023) are cut to
``event_time ± max(duration, 0.5 s)``, Tukey-tapered (alpha 0.1), scaled so
their *excess* in-band RMS over their own off-glitch background equals
``severity × inband_rms(clean GW170817 H1)``, and injected additively into the
GW170817 H1 analysis segment. The same three arms as ``stress_gw170817.py``
are run per cell: poisoned, detector-gated (``adapt_full``), centred oracle.

A noise-only control injects trigger-free O3 H1 excerpts with the identical
procedure (severity 6 and 10) so collapse can be attributed to the glitch and
not to the short window of added O3 noise.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/stress_real_glitches.py --outdir results/stress_real_glitches_v1
    python examples/stress_real_glitches.py --smoke   # 2 cells / 64 samples
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
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

from stress_gw170817 import (  # noqa: E402
    DEFAULT_DETECTOR,
    _build_spec_stack,
    _detector_gates,
    _gated_recovers,
    _git_commit,
    _load_detector,
    _poison_collapsed,
    _sample_dl,
    _sha256,
    append_csv_row,
)

DEFAULT_OUTDIR = REPO_ROOT / "results" / "stress_real_glitches_v1"
DEFAULT_FIGDIR = REPO_ROOT / "paper" / "figures"
SEVERITIES = (3.0, 6.0, 10.0)
NOISE_SEVERITIES = (6.0, 10.0)
NOISE_LABEL = "Noise_Only"
T_REL_RANGE = (-1.5, -0.3)  # same as stress_gw170817.build_locked_grid

logger = logging.getLogger("stress_real_glitches")

FIELDNAMES = [
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
    # real-glitch specific
    "gs_label",
    "gs_event_time",
    "gs_snr",
    "gs_duration",
    "gs_confidence",
    "excerpt_window_s",
    "severity_mode",
    "scale_k",
    "native_snr",
    "injected_snr_w",
    "injected_inband_rms_ratio",
    "is_noise_control",
    "gates_json",
]


# ---------------------------------------------------------------------------
# Catalogue + excerpts
# ---------------------------------------------------------------------------


def prepare_catalogue(args: argparse.Namespace) -> Dict[str, Any]:
    from adapt import gravity_spy_io as gs

    raw_dir = Path(args.raw_dir)
    selected_csv = Path(args.selected_csv)
    if selected_csv.is_file() and not args.rebuild_catalogue:
        selected = gs.load_selected_catalogue(selected_csv)
        logger.info("Loaded cached selection %s (%d rows)", selected_csv, len(selected))
        raw_paths = [raw_dir / n for n in gs.ZENODO_FILES if (raw_dir / n).is_file()]
    else:
        raw_paths = gs.download_h1_o3_tables(raw_dir, verify=not args.no_verify)
        selected = gs.build_selected_catalogue(raw_paths, selected_csv)

    # Full H1 trigger times (for noise-only guard). Only event_time is needed.
    all_times: Optional[pd.DataFrame] = None
    if raw_paths:
        frames = []
        for p in raw_paths:
            hdr = pd.read_csv(p, nrows=0).columns
            col = "event_time" if "event_time" in hdr else ("peak_time" if "peak_time" in hdr else None)
            if col is None:
                continue
            frames.append(pd.read_csv(p, usecols=[col]).rename(columns={col: "event_time"}))
        if frames:
            all_times = pd.concat(frames, ignore_index=True)
    return {"selected": selected, "all_times": all_times, "raw_paths": raw_paths}


def fetch_excerpts(
    selected: pd.DataFrame,
    *,
    sample_rate: float,
    f_min: float,
    f_max: float,
    outdir: Path,
    n_workers: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """Fetch 8 s of H1 open strain per catalogue row (parallel); skip data gaps."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from adapt import gravity_spy_io as gs

    rows = list(selected.reset_index(drop=True).iterrows())

    def _one(i_row):
        i, row = i_row
        t = float(row["event_time"])
        raw = gs.fetch_h1_excerpt(t, sample_rate=sample_rate)
        exc = gs.extract_glitch_excerpt(
            raw, sample_rate=sample_rate, duration_s=float(row["duration"]), f_min=f_min, f_max=f_max
        )
        return int(i), row, exc

    out: Dict[int, Dict[str, Any]] = {}
    fails: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
        futs = {ex.submit(_one, ir): ir for ir in rows}
        for fu in as_completed(futs):
            i, row = futs[fu]
            try:
                i, row, exc = fu.result()
                out[int(i)] = {"row": row.to_dict(), "excerpt": exc}
                logger.info(
                    "[fetched %d/%d] %s t=%.2f dur=%.2fs window=%.2fs excess/bg=%.2f",
                    len(out), len(rows), row["ml_label"], float(row["event_time"]),
                    float(row["duration"]), exc.window_s,
                    exc.excess_rms_inband / max(exc.background_rms_inband, 1e-30),
                )
            except Exception as e:  # data gap / network
                logger.warning("fetch failed for %s @ %.2f: %s", row["ml_label"], float(row["event_time"]), e)
                fails.append({**row.to_dict(), "error": f"{type(e).__name__}: {e}"})
    pd.DataFrame(fails).to_csv(outdir / "fetch_failures.csv", index=False)
    return out


def fetch_noise_excerpts(
    times: Sequence[float],
    *,
    sample_rate: float,
    f_min: float,
    f_max: float,
    window_s: float,
    n_workers: int = 10,
) -> Dict[int, Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from adapt import gravity_spy_io as gs

    def _one(i_t):
        i, t = i_t
        raw = gs.fetch_h1_excerpt(float(t), sample_rate=sample_rate)
        exc = gs.extract_glitch_excerpt(
            raw, sample_rate=sample_rate, duration_s=0.5 * window_s, min_window_s=0.5 * window_s,
            f_min=f_min, f_max=f_max,
        )
        return i, t, exc

    out: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
        futs = {ex.submit(_one, it): it for it in enumerate(times)}
        for fu in as_completed(futs):
            i, t = futs[fu]
            try:
                i, t, exc = fu.result()
                out[i] = {
                    "row": {
                        "ml_label": NOISE_LABEL,
                        "event_time": float(t),
                        "snr": float("nan"),
                        "duration": float(window_s),
                        "ml_confidence": float("nan"),
                    },
                    "excerpt": exc,
                }
                logger.info("[noise fetched %d/%d] t=%.2f", len(out), len(times), float(t))
            except Exception as e:
                logger.warning("noise fetch failed @ %.2f: %s", float(t), e)
    return out


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


class Transplanter:
    """Maps whitened O3 excerpts into GW170817 H1 strain units.

    ``colour`` gives the strain-domain shape (arbitrary normalisation); the two
    scaling rules fix the amplitude:

    * ``native``: whitened energy of the injected glitch, measured with the
      GW170817 H1 whitening, equals the excess energy observed in O3
      (i.e. same SNR as the real event);
    * ``ladder``: raw in-band RMS of the injected glitch window equals
      ``severity x inband_rms(clean H1)`` -- the synthetic-grid convention.
    """

    def __init__(self, *, asd_h1: np.ndarray, delta_f: float, noise_std: float, sample_rate: float,
                 f_min: float, f_max: float, clean_crop_h1: np.ndarray, rms_inband_h1: float):
        from adapt.stft_context import whiten_td_crop_with_asd

        self.asd = np.asarray(asd_h1, dtype=np.float64)
        self.delta_f = float(delta_f)
        self.noise_std = float(noise_std)
        self.fs = float(sample_rate)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.rms_inband_h1 = float(rms_inband_h1)
        self._whiten = whiten_td_crop_with_asd
        wc = whiten_td_crop_with_asd(clean_crop_h1, self.fs, self.asd, delta_f=self.delta_f, noise_std=self.noise_std)
        self.clean_white_var = float(np.var(wc)) or 1.0

    def colour(self, w: np.ndarray) -> np.ndarray:
        from adapt import gravity_spy_io as gs

        return gs.colour_with_asd(w, self.fs, asd=self.asd, delta_f=self.delta_f, f_min=self.f_min, f_max=self.f_max)

    def whitened_energy(self, g: np.ndarray) -> float:
        # Pad to 4 s so the whitening taper/leakage behaves as in the analysis crop.
        n4 = int(round(4.0 * self.fs))
        x = np.zeros(max(n4, g.size))
        s = (x.size - g.size) // 2
        x[s : s + g.size] = g
        wg = self._whiten(x, self.fs, self.asd, delta_f=self.delta_f, noise_std=self.noise_std, taper=False)
        return float(np.sum(wg**2) / self.clean_white_var)

    def scale_native(self, g0: np.ndarray, excess_energy_w: float) -> float:
        e0 = self.whitened_energy(g0)
        return float(np.sqrt(max(excess_energy_w, 0.0) / max(e0, 1e-300)))

    def scale_energy(self, g0: np.ndarray, target_energy_w: float) -> float:
        e0 = self.whitened_energy(g0)
        return float(np.sqrt(max(target_energy_w, 0.0) / max(e0, 1e-300)))

    def scale_ladder(self, g0: np.ndarray, severity: float) -> float:
        from adapt import gravity_spy_io as gs

        r = gs.inband_rms_local(g0, self.fs, self.f_min, self.f_max)
        return float(severity) * self.rms_inband_h1 / max(r, 1e-300)


def inject_excerpt_into_event(
    event,
    assets,
    *,
    excerpt,
    mode: str,
    severity: float,
    t_rel: float,
    td_clean_full: Dict[str, np.ndarray],
    transplanter: Transplanter,
    f_max: float,
    roll_off: float,
    target_energy_w: Optional[float] = None,
    use_denoised: bool = True,
):
    """Additive injection of a transplanted real excerpt into GW170817 H1.

    ``mode`` in {"native", "ladder", "energy"}; ``energy`` matches a given
    whitened energy (used for the noise-only control).
    """
    from adapt import gravity_spy_io as gs
    from adapt.event_glitch_io import td_to_fd_strain
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

    settings = dict(event.settings)
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    sample_rate = float(excerpt.sample_rate)
    t_peak = (duration - time_buffer) + float(t_rel)

    w = excerpt.denoised if use_denoised else excerpt.waveform
    g0 = transplanter.colour(w)
    if mode == "native":
        k = transplanter.scale_native(g0, excerpt.excess_energy_w)
    elif mode == "ladder":
        k = transplanter.scale_ladder(g0, severity)
    elif mode == "energy":
        k = transplanter.scale_energy(g0, float(target_energy_w))
    else:
        raise ValueError(mode)
    g_win = g0 * k
    injected_energy_w = transplanter.whitened_energy(g_win)
    injected_rms_ratio = gs.inband_rms_local(g_win, sample_rate, transplanter.f_min, f_max) / transplanter.rms_inband_h1

    td_h1 = td_clean_full["H1"]
    g = gs.place_series(g_win, n_samples=td_h1.size, sample_rate=sample_rate, t_peak=t_peak)
    td_full = {d: td_clean_full[d].copy() for d in td_clean_full}
    td_full["H1"] = td_h1 + g

    data = copy.deepcopy(event.data)
    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))
    g_fd = td_to_fd_strain(g, sample_rate, roll_off=roll_off, f_max=f_max)
    if len(g_fd) < n_freq:
        g_fd = np.pad(g_fd, (0, n_freq - len(g_fd)))
    else:
        g_fd = g_fd[:n_freq]
    data["waveform"]["H1"] = np.asarray(data["waveform"]["H1"], dtype=np.complex128) + g_fd
    # ASD policy: stationary (original analysis ASD retained)

    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    td_stft = {}
    for det, x in td_full.items():
        if end > len(x):
            e_i = len(x)
            s_i = e_i - n_crop
        else:
            s_i, e_i = start, end
        td_stft[det] = x[s_i:e_i].copy()

    meta = {
        "t_peak_in_segment": float(t_peak),
        "sample_rate": sample_rate,
        "crop_start": int(start),
        "td_full": td_full,
        "td_stft": td_stft,
        "scale_k": float(k),
        "injected_energy_w": float(injected_energy_w),
        "injected_snr_w": float(np.sqrt(max(injected_energy_w, 0.0))),
        "injected_inband_rms_ratio": float(injected_rms_ratio),
        **excerpt.to_meta(),
    }
    return data, td_stft, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def aggregate_only(args: argparse.Namespace) -> None:
    """Rebuild summary.json, failures.csv and Fig. 8 from an existing results.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = Path(args.outdir)
    df = pd.read_csv(outdir / "results.csv")
    clean_ci = json.loads((outdir / "clean_reference.json").read_text())
    cfg = json.loads((outdir / "stress_config.json").read_text())
    write_summary_and_figure(outdir, df, clean_ci, cfg, figdir=Path(args.figdir))
    logger.info("Aggregated %d rows -> %s", len(df), outdir / "summary.json")


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt import gravity_spy_io as gs
    from adapt.dingo_bns_demo import (
        discover_assets,
        load_bns_checkpoint,
        load_event_dataset,
        load_event_td_crops,
        select_device,
    )
    from adapt.event_glitch_io import load_full_event_td
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from adapt.stft_context import inband_rms
    from dingo.gw.domains import build_domain_from_model_metadata

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    rng = np.random.default_rng(int(args.seed))

    # ---- catalogue + strain excerpts ----
    cat = prepare_catalogue(args)
    selected: pd.DataFrame = cat["selected"]
    if args.smoke:
        selected = selected.head(2)

    # ---- DINGO assets ----
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None
    )
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = assets["fixed_context"]
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"])
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)
    asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))

    td_clean_full: Dict[str, np.ndarray] = {}
    for det in detectors:
        td, _, _ = load_full_event_td(assets, settings, det)
        td_clean_full[det] = td
    rms_h1 = float(inband_rms(td_clean_full["H1"], sample_rate, f_min=f_min, f_max=f_max))
    _, cs0, ce0 = __import__("adapt.glitch_excision", fromlist=["analysis_crop_bounds"]).analysis_crop_bounds(
        duration=duration, time_buffer=time_buffer, sample_rate=sample_rate
    )
    transplanter = Transplanter(
        asd_h1=asds_clean["H1"], delta_f=delta_f, noise_std=noise_std, sample_rate=sample_rate,
        f_min=f_min, f_max=f_max, clean_crop_h1=td_clean_full["H1"][cs0:ce0], rms_inband_h1=rms_h1,
    )

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    model, det_raw = _load_detector(det_path, device)
    threshold = float(det_raw.get("threshold", 0.5))
    gate_half_s = float(det_raw.get("gate_half_s", args.gate_half_s))
    norm_stats = det_raw.get("norm_stats")
    stft_kwargs = dict(det_raw.get("stft_kwargs") or {})

    # ---- clean reference + event threshold (same recipe as stress_gw170817) ----
    n_samples = 64 if args.smoke else int(args.num_samples)
    clean_ci = _sample_dl(assets, event.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
    (outdir / "clean_reference.json").write_text(json.dumps(clean_ci, indent=2))
    clean_crops = load_event_td_crops(assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS)
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    crop_start = max(0, trig_idx - n_crop // 2)
    spec_c, _ = _build_spec_stack(
        clean_crops, asds_clean=asds_clean, detectors=detectors, sample_rate=sample_rate,
        delta_f=delta_f, noise_std=noise_std, norm_stats=norm_stats, stft_kwargs=stft_kwargs,
    )
    _, clean_probs = _detector_gates(
        model, spec_c, detectors=detectors, crop_start=crop_start, sample_rate=sample_rate,
        threshold=threshold, gate_half_s=gate_half_s, device=device,
    )
    thr_event = float(args.threshold_event) if args.threshold_event is not None else float(
        max(threshold, float(np.max(clean_probs)) + 0.05)
    )
    logger.info("Clean d_L %s; thr_event=%.3f; inband_rms(H1)=%.3e", clean_ci, thr_event, rms_h1)

    # ---- excerpts ----
    glitch_ex = fetch_excerpts(selected, sample_rate=sample_rate, f_min=f_min, f_max=f_max, outdir=outdir)
    noise_ex: Dict[int, Dict[str, Any]] = {}
    n_noise_times = 1 if args.smoke else max(1, int(args.n_noise_cells) // len(NOISE_SEVERITIES))
    if cat["all_times"] is not None and not args.skip_noise_control:
        try:
            noise_times = gs.pick_noise_only_times(
                cat["all_times"], selected, n=n_noise_times, guard_s=10.0, rng=rng
            )
            noise_ex = fetch_noise_excerpts(
                noise_times, sample_rate=sample_rate, f_min=f_min, f_max=f_max, window_s=1.0
            )
        except Exception as e:
            logger.warning("noise-only control unavailable: %s", e)

    # Catalogue with fetch status
    cat_rows = []
    for i, row in selected.reset_index(drop=True).iterrows():
        r = dict(row)
        r["fetched"] = int(i) in glitch_ex
        if r["fetched"]:
            r.update(glitch_ex[int(i)]["excerpt"].to_meta())
        cat_rows.append(r)
    pd.DataFrame(cat_rows).to_csv(outdir / "glitch_catalogue.csv", index=False)

    # ---- config ----
    cfg = {
        "outdir": str(outdir),
        "master_seed": int(args.seed),
        "num_samples": n_samples,
        "batch_size": int(args.batch_size),
        "gate_half_s": gate_half_s,
        "severities": list(SEVERITIES),
        "noise_severities": list(NOISE_SEVERITIES),
        "asd_policy": "stationary",
        "t_rel_range": list(T_REL_RANGE),
        "n_glitches_selected": int(len(selected)),
        "n_glitches_fetched": int(len(glitch_ex)),
        "n_noise_excerpts": int(len(noise_ex)),
        "detector_ckpt": str(det_path),
        "detector_sha256": _sha256(det_path),
        "baseline_ckpt": str(assets["baseline_ckpt"]),
        "baseline_sha256": _sha256(Path(assets["baseline_ckpt"])),
        "threshold_base": threshold,
        "threshold_event": thr_event,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "zenodo_record": gs.ZENODO_RECORD,
        "severity_modes": ["native", "ladder"],
        "excerpt_recipe": (
            "8 s H1 open strain (GWOSC, chunk-level HTTP range read of the bulk HDF5) centred on the "
            "Gravity Spy event_time; whitened with a Welch PSD from its own off-window part (unit-variance "
            "background, 20-1500 Hz); cut to event_time +/- max(duration, 0.5 s) with Tukey(alpha=0.1) taper; "
            "STFT-thresholded (8x per-frequency median off-window power) within the trigger duration (+0.15 s) to a "
            "glitch-only estimate; coloured with the line-smoothed (2 Hz running median) GW170817 H1 analysis ASD "
            "(zero-padded, re-tapered) and injected additively into the H1 analysis segment. "
            "Scaling: 'native' = same whitened energy (SNR^2) as observed in O3; 'ladder' = raw in-band RMS "
            "of the injected window equals severity x inband_rms(clean H1), the synthetic-grid convention. "
            "Working in the whitened domain is required because Gravity Spy glitches of Omicron SNR 8-30 "
            "are not measurable as a raw in-band RMS excess over the 20-40 Hz noise wall."
        ),
        "noise_control_recipe": (
            "Trigger-free O3 H1 windows (no Gravity Spy trigger within +/-10 s), whitened and windowed "
            "identically but NOT thresholded, coloured with the GW170817 ASD and scaled to (a) 'noise_native': "
            "whitened energy = window length (unit-variance O3 noise added over the window) and (b) "
            "'noise_matched_<sev>': whitened energy equal to the median injected energy of the ladder glitch "
            "cells at that severity."
        ),
        "success_criteria": {
            "poison_collapsed": "d_L hi<15 or lo>90",
            "gated_recovers": "med in [20,50], CI overlaps clean, |med-clean_med|<=10",
        },
    }
    (outdir / "stress_config.json").write_text(json.dumps(cfg, indent=2))

    # ---- cells ----
    # Glitch cells first (native + ladder); noise-only controls are planned
    # after the glitch cells finish because their 'matched' energy targets are
    # the median injected energies of the ladder cells.
    cells: List[Dict[str, Any]] = []
    cid = 0
    for gi, item in sorted(glitch_ex.items()):
        t_rel = float(rng.uniform(*T_REL_RANGE))
        cells.append({"cell_id": cid, "src": item, "mode": "native", "severity": float("nan"), "noise": False,
                      "t_rel": t_rel, "seed": int(args.seed)})
        cid += 1
        for sev in SEVERITIES:
            cells.append({"cell_id": cid, "src": item, "mode": "ladder", "severity": float(sev), "noise": False,
                          "t_rel": t_rel, "seed": int(args.seed)})
            cid += 1
    if args.smoke:
        cells = cells[:3]
    if args.max_cells:
        cells = cells[: int(args.max_cells)]
    logger.info("Planned glitch cells: %d", len(cells))

    csv_path = outdir / "results.csv"
    done_ids = set()
    if csv_path.is_file() and not args.overwrite:
        try:
            done_ids = set(int(x) for x in pd.read_csv(csv_path)["cell_id"].tolist())
        except Exception:
            pass
    elif csv_path.is_file():
        csv_path.unlink()

    t_all = time.time()

    def _run_cell(i: int, cell: Dict[str, Any], n_total: int) -> Dict[str, Any]:
        t0 = time.time()
        row_src = cell["src"]["row"]
        excerpt = cell["src"]["excerpt"]
        label = str(row_src["ml_label"])
        row: Dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "family": label,
            "held_out": True,  # real glitches were never seen by the detector
            "severity": cell["severity"],
            "asd_policy": "stationary",
            "seed": cell["seed"],
            "detectors": "H1",
            "t_rel": cell["t_rel"],
            "params_json": json.dumps({"gs_event_time": float(row_src["event_time"]),
                                       "window_s": float(excerpt.window_s), "mode": cell["mode"]}),
            "error": "",
            "gs_label": label,
            "gs_event_time": float(row_src["event_time"]),
            "gs_snr": float(row_src.get("snr", float("nan"))),
            "gs_duration": float(row_src.get("duration", float("nan"))),
            "gs_confidence": float(row_src.get("ml_confidence", float("nan"))),
            "excerpt_window_s": float(excerpt.window_s),
            "severity_mode": cell["mode"],
            "native_snr": float(np.sqrt(max(excerpt.excess_energy_w, 0.0))),
            "is_noise_control": bool(cell["noise"]),
        }
        try:
            poison, td_stft, meta = inject_excerpt_into_event(
                event, assets, excerpt=excerpt, mode=cell["inj_mode"], severity=cell["severity"],
                t_rel=cell["t_rel"], td_clean_full=td_clean_full, transplanter=transplanter,
                f_max=f_max, roll_off=roll_off, target_energy_w=cell.get("target_energy_w"),
                use_denoised=not cell["noise"],
            )
            row["scale_k"] = meta["scale_k"]
            row["injected_snr_w"] = meta["injected_snr_w"]
            row["injected_inband_rms_ratio"] = meta["injected_inband_rms_ratio"]
            poison_ci = _sample_dl(assets, poison, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            row.update({"poison_lo": poison_ci["lo"], "poison_med": poison_ci["med"], "poison_hi": poison_ci["hi"],
                        "poison_collapsed": _poison_collapsed(poison_ci)})

            spec_g, _ = _build_spec_stack(
                meta["td_stft"], asds_clean=asds_clean, detectors=detectors, sample_rate=sample_rate,
                delta_f=delta_f, noise_std=noise_std, norm_stats=norm_stats, stft_kwargs=stft_kwargs,
            )
            gates, _ = _detector_gates(
                model, spec_g, detectors=detectors, crop_start=int(meta["crop_start"]), sample_rate=sample_rate,
                threshold=thr_event, gate_half_s=gate_half_s, device=device, ifo_whitelist=["H1"],
            )
            row["detector_n_gates"] = len(gates)
            row["detector_fired"] = bool(len(gates) > 0)
            t_trig = duration - time_buffer
            row["gates_json"] = json.dumps(
                [[round(g.t_start - t_trig, 3), round(g.t_end - t_trig, 3)] for g in gates]
            )
            gated = rebuild_event_from_gated_td(
                poison, td_by_det=meta["td_full"], gates=gates, sample_rate=sample_rate,
                roll_off=roll_off, f_max=f_max, original_asds=asds_clean,
            )
            rp = gated.meta.get("residual_power_frac") or {}
            row["residual_power_H1"] = float(rp.get("H1", 0.0) or 0.0)
            row["residual_power_L1"] = float(rp.get("L1", 0.0) or 0.0)
            gated_ci = _sample_dl(assets, gated.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            row.update({"gated_lo": gated_ci["lo"], "gated_med": gated_ci["med"], "gated_hi": gated_ci["hi"],
                        "gated_recovers": _gated_recovers(gated_ci, clean_ci)})

            t_peak = float(meta["t_peak_in_segment"])
            ogates = [GateWindow(detector="H1", t_start=t_peak - gate_half_s, t_end=t_peak + gate_half_s, score=1.0)]
            oracle = rebuild_event_from_gated_td(
                poison, td_by_det=meta["td_full"], gates=ogates, sample_rate=sample_rate,
                roll_off=roll_off, f_max=f_max, original_asds=asds_clean,
            )
            oracle_ci = _sample_dl(assets, oracle.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            row.update({"oracle_lo": oracle_ci["lo"], "oracle_med": oracle_ci["med"], "oracle_hi": oracle_ci["hi"],
                        "oracle_recovers": _gated_recovers(oracle_ci, clean_ci)})
            row["oracle_only_recovery"] = bool(row["oracle_recovers"] and not row["gated_recovers"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Cell %d failed", cell["cell_id"])
            traceback.print_exc()
        row["elapsed_s"] = round(time.time() - t0, 3)
        append_csv_row(csv_path, row, FIELDNAMES)
        logger.info(
            "[%d/%d] %s mode=%s sev=%s snr_w=%.1f poison_col=%s gated_ok=%s oracle_ok=%s fired=%s (%.1fs)",
            i + 1, n_total, label, cell["mode"], cell["severity"], row.get("injected_snr_w", float("nan")),
            row.get("poison_collapsed"), row.get("gated_recovers"), row.get("oracle_recovers"),
            row.get("detector_fired"), row["elapsed_s"],
        )
        return row

    for c in cells:
        c["inj_mode"] = c["mode"]
    for i, cell in enumerate(cells):
        if cell["cell_id"] in done_ids:
            continue
        _run_cell(i, cell, len(cells))
    logger.info("Glitch cells done in %.1f min", (time.time() - t_all) / 60.0)

    # ---- noise-only controls (energy targets from the ladder cells) ----
    if noise_ex:
        dfg = pd.read_csv(csv_path)
        dfg = dfg[(dfg["error"].fillna("") == "") & (~dfg["is_noise_control"].astype(bool))]
        n_win = None
        noise_cells: List[Dict[str, Any]] = []
        for ni, item in sorted(noise_ex.items()):
            t_rel = float(rng.uniform(*T_REL_RANGE))
            n_win = item["excerpt"].waveform.size
            noise_cells.append({"cell_id": cid, "src": item, "mode": "noise_native", "inj_mode": "energy",
                                "target_energy_w": float(n_win), "severity": float("nan"), "noise": True,
                                "t_rel": t_rel, "seed": int(args.seed)})
            cid += 1
            for sev in NOISE_SEVERITIES:
                sub = dfg[(dfg["severity_mode"] == "ladder") & (dfg["severity"] == float(sev))]
                if not len(sub):
                    continue
                e_target = float(np.median(sub["injected_snr_w"].astype(float) ** 2))
                noise_cells.append({"cell_id": cid, "src": item, "mode": f"noise_matched_{int(sev)}",
                                    "inj_mode": "energy", "target_energy_w": e_target, "severity": float(sev),
                                    "noise": True, "t_rel": t_rel, "seed": int(args.seed)})
                cid += 1
        if args.smoke:
            noise_cells = noise_cells[:2]
        logger.info("Planned noise-only cells: %d", len(noise_cells))
        for i, cell in enumerate(noise_cells):
            if cell["cell_id"] in done_ids:
                continue
            _run_cell(i, cell, len(noise_cells))
    logger.info("All cells done in %.1f min", (time.time() - t_all) / 60.0)

    df = pd.read_csv(csv_path)
    write_summary_and_figure(outdir, df, clean_ci, cfg, figdir=Path(args.figdir))
    logger.info("Real-glitch panel complete -> %s", outdir)


def write_summary_and_figure(outdir: Path, df: pd.DataFrame, clean_ci, cfg, *, figdir: Path) -> None:
    ok = df[df["error"].fillna("") == ""].copy()
    for col in ("held_out", "poison_collapsed", "gated_recovers", "oracle_recovers", "detector_fired", "is_noise_control"):
        if col in ok.columns:
            ok[col] = ok[col].astype(bool)
    glitch_all = ok[~ok["is_noise_control"]]
    ladder = glitch_all[glitch_all["severity_mode"] == "ladder"]
    native_all = glitch_all[glitch_all["severity_mode"] == "native"]
    # Native cells whose excerpt showed no measurable whitened excess inject
    # nothing (k = 0); they are reported separately and excluded from rates.
    native_undetectable = native_all[native_all["injected_snr_w"].astype(float) <= 1.0]
    native = native_all[native_all["injected_snr_w"].astype(float) > 1.0]
    noise = ok[ok["is_noise_control"]]

    def _rate(sub, col):
        return float(sub[col].mean()) if len(sub) else None

    def _mad(sub):
        s = sub[sub["gated_recovers"]]
        return float((s["gated_med"] - clean_ci["med"]).abs().mean()) if len(s) else None

    def _block(sub):
        return {
            "n": int(len(sub)),
            "gated_recovery_rate": _rate(sub, "gated_recovers"),
            "oracle_recovery_rate": _rate(sub, "oracle_recovers"),
            "poison_collapse_rate": _rate(sub, "poison_collapsed"),
            "detector_fire_rate": _rate(sub, "detector_fired"),
            "mean_abs_delta_med_d_L_successes": _mad(sub),
            "median_injected_snr_w": float(sub["injected_snr_w"].median()) if len(sub) else None,
        }

    def _by_family(sub):
        if not len(sub):
            return []
        return (
            sub.groupby("gs_label")
            .agg(n=("gated_recovers", "size"), gated_recovery=("gated_recovers", "mean"),
                 oracle_recovery=("oracle_recovers", "mean"), poison_collapse=("poison_collapsed", "mean"),
                 detector_fire_rate=("detector_fired", "mean"), median_injected_snr_w=("injected_snr_w", "median"))
            .reset_index().rename(columns={"gs_label": "family"}).to_dict(orient="records")
        )

    failures = ladder[~ladder["gated_recovers"]]
    failures.to_csv(outdir / "failures.csv", index=False)

    by_family = _by_family(ladder)
    summary = {
        "n_rows": int(len(df)),
        "n_ok": int(len(ok)),
        "n_errors": int((df["error"].fillna("") != "").sum()),
        "n_glitch_cells": int(len(glitch_all)),
        "n_ladder_cells": int(len(ladder)),
        "n_native_cells": int(len(native)),
        "n_noise_control_cells": int(len(noise)),
        "clean_reference_d_L": clean_ci,
        # 'overall' follows the stress_test_excision_v1 schema and refers to the
        # synthetic-comparable severity ladder (3, 6, 10).
        "overall": _block(ladder),
        "by_held_out": {"true": _rate(ladder, "gated_recovers")} if len(ladder) else {},
        "by_family": by_family,
        "by_severity": ladder.groupby("severity")["gated_recovers"].mean().astype(float).to_dict() if len(ladder) else {},
        "by_severity_poison_collapse": ladder.groupby("severity")["poison_collapsed"].mean().astype(float).to_dict() if len(ladder) else {},
        "by_asd_policy": {
            "poison_collapse": {"stationary": _rate(ladder, "poison_collapsed")},
            "gated_recovery": {"stationary": _rate(ladder, "gated_recovers")},
        },
        "native": {
            **_block(native),
            "by_family": _by_family(native),
            "n_undetectable_excluded": int(len(native_undetectable)),
            "undetectable_labels": sorted(native_undetectable["gs_label"].astype(str).unique().tolist()),
            "note": (
                "Real glitches transplanted at their observed O3 whitened energy (same SNR as in the Gravity Spy "
                "data). Cells whose excerpt had no measurable whitened excess over its own background (injected "
                "SNR <= 1) inject nothing and are excluded here."
            ),
        },
        "noise_control": {
            "n": int(len(noise)),
            "by_mode": (
                noise.groupby("severity_mode")
                .agg(n=("gated_recovers", "size"), poison_collapse=("poison_collapsed", "mean"),
                     gated_recovery=("gated_recovers", "mean"), oracle_recovery=("oracle_recovers", "mean"),
                     detector_fire_rate=("detector_fired", "mean"), median_injected_snr_w=("injected_snr_w", "median"))
                .reset_index().to_dict(orient="records")
                if len(noise) else []
            ),
            "poison_collapse_rate": _rate(noise, "poison_collapsed"),
            "gated_recovery_rate": _rate(noise, "gated_recovers"),
            "note": (
                "Trigger-free O3 H1 windows (no Gravity Spy trigger within +/-10 s), whitened/windowed identically "
                "(not thresholded), coloured with the GW170817 ASD. 'noise_native' adds unit-variance O3 noise over "
                "the window; 'noise_matched_<sev>' matches the median injected whitened energy of the ladder cells."
            ),
        },
        "n_failures": int(len(failures)),
        "data_note": (
            "Excerpt scaling is done in the whitened domain (see stress_config.json:excerpt_recipe). "
            "The synthetic severity ladder corresponds to whitened SNRs far above typical Gravity Spy "
            "glitches; the 'native' block reports the realistic regime."
        ),
        "config_ref": "stress_config.json",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    glitch = ladder  # for the figure

    # ---- Fig 8 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        labels = [r["family"] for r in by_family]
        rows = list(by_family)
        nm = noise[noise["severity_mode"].str.startswith("noise_matched")] if len(noise) else noise
        if len(nm):
            labels.append("noise only\n(energy-matched)")
            rows.append({"poison_collapse": _rate(nm, "poison_collapsed"), "gated_recovery": _rate(nm, "gated_recovers"),
                         "oracle_recovery": _rate(nm, "oracle_recovers"), "detector_fire_rate": _rate(nm, "detector_fired"),
                         "n": len(nm)})
        def _bars(ax, labels, rows, title):
            x = np.arange(len(labels)); w = 0.26
            ax.bar(x - w, [r["poison_collapse"] for r in rows], w, color="#c0392b", label="poisoned: collapsed")
            ax.bar(x, [r["gated_recovery"] for r in rows], w, color="#2a9d8f", label="gated: recovered")
            ax.bar(x + w, [r["oracle_recovery"] for r in rows], w, color="#7f7f7f", label="centred oracle")
            ax.plot(x, [r["detector_fire_rate"] for r in rows], "k_", ms=12, mew=1.2, label="detector fire rate")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{l.replace('_', ' ')}\n(n={r['n']})" for l, r in zip(labels, rows)], fontsize=6.0)
            ax.set_ylim(0, 1.08); ax.set_ylabel("fraction of cells")
            ax.set_title(title, fontsize=7.5)

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.9), gridspec_kw={"hspace": 0.55})
        _bars(a1, labels, rows, "(a) synthetic-comparable severity ladder {3, 6, 10}")
        nat_rows = _by_family(native)
        nat_labels = [r["family"] for r in nat_rows]
        nn = noise[noise["severity_mode"] == "noise_native"] if len(noise) else noise
        if len(nn):
            nat_labels.append("noise only\n(native level)")
            nat_rows.append({"poison_collapse": _rate(nn, "poison_collapsed"), "gated_recovery": _rate(nn, "gated_recovers"),
                             "oracle_recovery": _rate(nn, "oracle_recovers"), "detector_fire_rate": _rate(nn, "detector_fired"),
                             "n": len(nn)})
        if nat_rows:
            _bars(a2, nat_labels, nat_rows, "(b) native loudness (observed O3 SNR)")
        else:
            a2.set_axis_off()
        a2.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax = a2
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / "fig8_real_glitches.pdf"); fig.savefig(figdir / "fig8_real_glitches.png", dpi=300)
        fig.savefig(outdir / "fig8_real_glitches.pdf")
        plt.close(fig)
    except Exception as e:
        logger.warning("figure failed: %s", e)

    repro = f"""# Reproduce the real Gravity Spy glitch panel

Event: **GW170817** (frozen official DINGO-BNS). Glitches: real O3 H1 excerpts
from Gravity Spy (Zenodo {cfg['zenodo_record']}, Glanzer et al. 2023), selected by
`adapt.gravity_spy_io` (H1, ml_confidence >= 0.95, 8 <= snr <= 30, duration <= 3 s;
Blip 10 / Scattered_Light 5 / Whistle 5 / Koi_Fish 5). Cached selection:
`data/gravity_spy/selected_h1_o3.csv`.

## Environment

```bash
conda activate adapt_env
cd {REPO_ROOT}
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
```

## Run

```bash
python examples/stress_real_glitches.py --seed {cfg['master_seed']} \\
  --num-samples {cfg['num_samples']} --outdir {outdir}
# smoke: python examples/stress_real_glitches.py --smoke --outdir results/stress_real_glitches_smoke
```

Raw Zenodo tables download to `data/gravity_spy/raw/` (git-ignored) on first
run; H1 strain is fetched from GWOSC with `gwpy` (8 s per glitch).

## Excerpt recipe

{cfg['excerpt_recipe']}

## Outputs

- `results.csv`, `summary.json`, `failures.csv`, `glitch_catalogue.csv`,
  `fetch_failures.csv`, `clean_reference.json`, `stress_config.json`
- `paper/figures/fig8_real_glitches.pdf`

## Success criteria

- Poison collapsed: `d_L` hi < 15 or lo > 90
- Gated recovers: med in [20, 50], CI overlaps clean, |med - clean_med| <= 10 Mpc
"""
    (outdir / "REPRODUCE.md").write_text(repro)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "gravity_spy" / "raw")
    p.add_argument("--selected-csv", type=Path, default=REPO_ROOT / "data" / "gravity_spy" / "selected_h1_o3.csv")
    p.add_argument("--rebuild-catalogue", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="skip Zenodo API verification")
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--threshold-event", type=float, default=None)
    p.add_argument("--n-noise-cells", type=int, default=10)
    p.add_argument("--skip-noise-control", action="store_true")
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="3 glitch cells + 2 noise cells, 64 samples")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true", help="rebuild summary/figure from results.csv")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.aggregate_only:
        aggregate_only(args)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
