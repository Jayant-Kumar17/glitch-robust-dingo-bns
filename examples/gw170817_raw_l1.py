#!/usr/bin/env python3
"""Raw, unsubtracted GW170817 L1 versus the LVK glitch-subtracted package.

The official demonstration frame is the BayesWave-cleaned LOSC release. This
script fetches the GWOSC open-data strain for the same 128 s segment, which is
the calibrated detector output before that event-specific subtraction, packages
only L1 through ``data_to_domain``, and leaves H1, V1, and the original
analysis ASDs untouched.

Three ``GWSamplerGNPE`` runs at 20_000 samples, each importance-sampled:

  * ``lvk_subtracted_ref`` — official demonstration package
  * ``raw_l1_poisoned`` — raw L1 spliced into that package
  * ``raw_l1_adapt_full`` — matched-delta gate on the raw L1 package

Outputs under ``results/gw170817_raw_l1/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import REPO_ROOT, resolve_under_repo
from dingo_conditioning import (
    base_frequency_domain,
    data_to_domain_package,
    splice_package,
)
from official_control import (
    INFERENCE_PARAMS,
    ci_overlap,
    js_divergence_1d,
    param_summary,
    run_importance_sampling,
)
from pe_session import PESession

logger = logging.getLogger("gw170817_raw_l1")
SAMPLE_RATE = 4096.0
N_SAMPLES = 20000
BATCH = 5000


def _segment(assets: dict, settings: dict) -> tuple[float, float]:
    trigger = float(assets["trigger_time"])
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    return trigger - (duration - time_buffer), trigger + time_buffer


def _read_local_l1(assets: dict, start: float, end: float) -> np.ndarray:
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.read(str(assets["strain_gwf"]["L1"]), channel=assets["channels"]["L1"])
    ts = ts.crop(start, end)
    if float(ts.sample_rate.value) != SAMPLE_RATE:
        ts = ts.resample(SAMPLE_RATE)
    n = int(round(128.0 * SAMPLE_RATE))
    x = np.asarray(ts.value, dtype=np.float64)
    if x.size > n:
        x = x[:n]
    elif x.size < n:
        x = np.pad(x, (0, n - x.size))
    return x


def _fetch_raw_l1(start: float, end: float, strain_hdf5: Path | None = None) -> np.ndarray:
    from gwpy.timeseries import TimeSeries

    if strain_hdf5 is not None:
        # Local copy of L-L1_GWOSC_4KHZ_R1-1187006835-4096.hdf5. Remote
        # fetch_open_data reads that 124 MB file in small HDF5 requests.
        import h5py

        with h5py.File(strain_hdf5, "r") as handle:
            gps0 = float(np.asarray(handle["meta/GPSstart"]))
            i0 = int(round((start - gps0) * SAMPLE_RATE))
            n = int(round((end - start) * SAMPLE_RATE))
            x = np.asarray(handle["strain/Strain"][i0 : i0 + n], dtype=np.float64)
        if x.size != n:
            raise RuntimeError(f"local L1 slice has {x.size} samples, expected {n}")
        return x
    else:
        ts = TimeSeries.fetch_open_data(
            "L1", start, end, sample_rate=int(SAMPLE_RATE), cache=True, verbose=False
        )
    n = int(round((end - start) * SAMPLE_RATE))
    x = np.asarray(ts.value, dtype=np.float64)
    if x.size > n:
        x = x[:n]
    elif x.size < n:
        x = np.pad(x, (0, n - x.size))
    return x


def _weights(frame: pd.DataFrame):
    if "weights" not in frame.columns:
        return None
    return frame["weights"].to_numpy()


def _compare(reference, other) -> dict:
    ref = reference.samples
    alt = other.samples
    rw, aw = _weights(ref), _weights(alt)
    js = {}
    overlap = {}
    medians = {}
    for name in INFERENCE_PARAMS:
        if name not in ref.columns or name not in alt.columns:
            continue
        js[name] = js_divergence_1d(
            ref[name].to_numpy(), alt[name].to_numpy(), wx=rw, wy=aw
        )
        rs = param_summary(ref, [name])[name]
        als = param_summary(alt, [name])[name]
        overlap[name] = bool(ci_overlap(rs, als))
        medians[name] = {"reference": rs, "other": als}
    finite = [v for v in js.values() if np.isfinite(v)]
    return {
        "js_nats": js,
        "js_median": float(np.median(finite)) if finite else None,
        "js_max": float(np.max(finite)) if finite else None,
        "ci90_overlap": overlap,
        "medians": medians,
    }


def _efficiency(result) -> dict:
    return {
        "n_samples": int(result.num_samples),
        "sample_efficiency": None
        if result.sample_efficiency is None
        else float(result.sample_efficiency),
        "n_eff": None if result.n_eff is None else float(result.n_eff),
        "log_evidence": None if result.log_evidence is None else float(result.log_evidence),
    }


def main(argv=None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=resolve_under_repo, default="results/gw170817_raw_l1")
    parser.add_argument("--num-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--strain-hdf5",
        type=Path,
        default=None,
        help="Local GWOSC L1 HDF5 (4 kHz, GPS 1187006835, 4096 s) instead of fetching",
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.dingo_bns_demo import (
        discover_assets,
        load_bns_checkpoint,
        load_event_dataset,
        load_event_td_crops,
    )
    from adapt.glitch_excision import analysis_crop_bounds, rebuild_event_from_gated_td
    from stress_gw170817 import _build_spec_stack, _detector_gates, _load_detector

    assets = discover_assets()
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = dict(assets["fixed_context"])
    raw_model = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw_model["metadata"]
    base = build_domain_from_model_metadata(metadata, base=True)
    domain = base_frequency_domain(base)
    detectors = ("H1", "L1", "V1")
    asds = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    start, end = _segment(assets, settings)

    local = _read_local_l1(assets, start, end)
    logger.info("Fetching GWOSC L1 for [%.2f, %.2f]", start, end)
    fetched = _fetch_raw_l1(start, end, args.strain_hdf5)
    n = min(local.size, fetched.size)
    # The loud pre-merger transient sits about 1.1 s before the trigger.
    trigger = float(assets["trigger_time"])
    i0 = int(round((trigger - 1.6 - start) * SAMPLE_RATE))
    i1 = int(round((trigger - 0.4 - start) * SAMPLE_RATE))
    i0, i1 = max(0, i0), min(n, i1)
    comparison = {
        "segment": [start, end],
        "correlation": float(np.corrcoef(local[:n], fetched[:n])[0, 1]),
        "std_local": float(np.std(local[:n])),
        "std_fetched": float(np.std(fetched[:n])),
        "glitch_window_max_local": float(np.max(np.abs(local[i0:i1]))),
        "glitch_window_max_fetched": float(np.max(np.abs(fetched[i0:i1]))),
        "source": (
            f"local GWOSC HDF5 {args.strain_hdf5}"
            if args.strain_hdf5
            else "gwpy.TimeSeries.fetch_open_data L1, GWOSC open data"
        ),
    }
    logger.info("L1 fetch vs cleaned frame: %s", comparison)
    (outdir / "l1_fetch_comparison.json").write_text(json.dumps(comparison, indent=2))
    if comparison["correlation"] > 0.999:
        logger.warning(
            "Fetched L1 matches the cleaned local frame. The open-data release "
            "returned the BayesWave-subtracted strain, so raw_l1 is not an independent glitch."
        )

    packaged = data_to_domain_package({"L1": fetched}, {"L1": asds["L1"]}, domain)
    raw_event = splice_package(event.data, packaged, {"L1": asds["L1"]})

    # Detector on the raw L1 crop, with H1/V1 from the local frames.
    device_cpu = __import__("torch").device("cpu")
    det_path = REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
    model, det_raw = _load_detector(det_path, device_cpu)
    stft_kwargs = dict(det_raw.get("stft_kwargs") or {})
    crops = load_event_td_crops(assets, sample_rate=SAMPLE_RATE, crop_seconds=4.0)
    _trig, crop0, crop1 = analysis_crop_bounds(
        duration=128.0, time_buffer=2.0, sample_rate=SAMPLE_RATE
    )
    crops["L1"] = fetched[crop0:crop1]
    delta_f = float(base.delta_f)
    noise_std = float(base.noise_std)
    spec, _energy = _build_spec_stack(
        crops,
        asds_clean=asds,
        detectors=list(detectors),
        sample_rate=SAMPLE_RATE,
        delta_f=delta_f,
        noise_std=noise_std,
        norm_stats=det_raw.get("norm_stats"),
        stft_kwargs=stft_kwargs,
    )
    import torch

    with torch.inference_mode():
        probs = (
            model.predict_probs(torch.from_numpy(np.asarray(spec, dtype=np.float32)).unsqueeze(0))
            .detach()
            .cpu()
            .numpy()[0]
        )
    thr_event = 0.7283049106597901
    gates, _ = _detector_gates(
        model,
        spec,
        detectors=list(detectors),
        crop_start=crop0,
        sample_rate=SAMPLE_RATE,
        threshold=thr_event,
        gate_half_s=float(det_raw.get("gate_half_s", 0.4)),
        device=device_cpu,
    )
    detector_log = {
        "threshold_event": thr_event,
        "threshold_validation": 0.46,
        "max_prob": {det: float(np.max(probs[i])) for i, det in enumerate(detectors)},
        "fires_0.46": {det: bool(np.any(probs[i] > 0.46)) for i, det in enumerate(detectors)},
        "fires_0.73": {det: bool(np.any(probs[i] > thr_event)) for i, det in enumerate(detectors)},
        "n_gates": len(gates),
        "gates": [
            {"detector": g.detector, "t_start": g.t_start, "t_end": g.t_end, "score": g.score}
            for g in gates
        ],
    }
    (outdir / "detector_raw_l1.json").write_text(json.dumps(detector_log, indent=2))
    logger.info("Raw L1 detector: %s", detector_log["max_prob"])

    td_full = {}
    for det in detectors:
        if det == "L1":
            td_full[det] = fetched
            continue
        from adapt.event_glitch_io import load_full_event_td

        td_full[det], _, _ = load_full_event_td(assets, settings, det)
    adapted = rebuild_event_from_gated_td(
        raw_event,
        td_by_det=td_full,
        gates=gates,
        sample_rate=SAMPLE_RATE,
        roll_off=0.4,
        f_max=1535.3046875,
        original_asds=asds,
        mode="matched_delta",
    )

    session = PESession(Path(assets["baseline_ckpt"]))
    packages = {
        "lvk_subtracted_ref": event.data,
        "raw_l1_poisoned": raw_event,
        "raw_l1_adapt_full": adapted.data,
    }
    results = {}
    for label, data in packages.items():
        logger.info("Sampling %s N=%d", label, args.num_samples)
        result = session.sample_result(
            data,
            settings,
            fixed,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        logger.info("Importance sampling %s", label)
        result = run_importance_sampling(result, num_processes=int(args.num_processes))
        keep = [c for c in INFERENCE_PARAMS if c in result.samples.columns]
        if "weights" in result.samples.columns:
            keep.append("weights")
        result.samples[keep].to_csv(outdir / f"{label}_samples.csv", index=False)
        results[label] = result
        (outdir / f"{label}_efficiency.json").write_text(
            json.dumps(_efficiency(result), indent=2)
        )

    ref = results["lvk_subtracted_ref"]
    report = {
        "sky_fixed": fixed,
        "fetch": comparison,
        "detector": detector_log,
        "efficiency": {k: _efficiency(v) for k, v in results.items()},
        "raw_l1_poisoned_vs_ref": _compare(ref, results["raw_l1_poisoned"]),
        "raw_l1_adapt_full_vs_ref": _compare(ref, results["raw_l1_adapt_full"]),
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "JS max poisoned %.4f adapt %.4f",
        report["raw_l1_poisoned_vs_ref"]["js_max"],
        report["raw_l1_adapt_full_vs_ref"]["js_max"],
    )


if __name__ == "__main__":
    main()
