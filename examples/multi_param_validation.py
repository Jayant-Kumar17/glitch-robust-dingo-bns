#!/usr/bin/env python3
"""15-parameter validation on collapsed real and synthetic cells.

No sky test: the public GW170817 checkpoint fixes right ascension and
declination. Clean and ``adapt_full`` posteriors are importance-sampled
``GWSamplerGNPE`` draws (10_000 samples). On the synthetic events the injected
parameters that the original script redrew with a seeded numpy generator are
recovered exactly; angles that came from an unseeded bilby prior are redrawn
with the event id as the bilby seed and saved, because they were not archived.

Outputs under ``results/multi_param_validation_v1/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import REPO_ROOT, resolve_under_repo
from official_control import (
    INFERENCE_PARAMS,
    ci_overlap,
    js_divergence_1d,
    param_summary,
    run_importance_sampling,
)
from pe_session import PESession

logger = logging.getLogger("multi_param_validation")
REAL_CSV = REPO_ROOT / "results" / "stress_real_glitches_v2" / "results.csv"
REAL_CAT = REPO_ROOT / "results" / "stress_real_glitches_v2" / "glitch_catalogue.csv"
SYN_CSV = REPO_ROOT / "results" / "stress_test_synthetic_bns_v1" / "results.csv"
SYN_EVENTS = REPO_ROOT / "results" / "stress_test_synthetic_bns_v1" / "events.csv"
FAMILIES = ("broadband_burst", "ringing", "scattered_light", "sine_gaussian")


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _select_real(real_csv=None, koi_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(real_csv or REAL_CSV)
    mask = _as_bool(df["poison_collapsed"])
    if "severity_mode" in df.columns:
        mask &= df["severity_mode"].astype(str) == "native"
    if "is_noise_control" in df.columns:
        mask &= ~_as_bool(df["is_noise_control"])
    out = df.loc[mask].copy()
    if koi_only and "family" in out.columns:
        out = out[out["family"].astype(str).str.contains("Koi", case=False)]
    return out.sort_values("cell_id")


def _select_synthetic() -> pd.DataFrame:
    df = pd.read_csv(SYN_CSV)
    df = df.loc[_as_bool(df["poison_collapsed"])].copy()
    df = df[df["error"].fillna("") == ""]
    picks = []
    for family in FAMILIES:
        sub = df[df["family"] == family].sort_values(["event_id", "cell_id"])
        picks.append(sub.head(5))
    return pd.concat(picks, ignore_index=True)


def _weights(frame: pd.DataFrame):
    if frame is None or "weights" not in frame.columns:
        return None
    return np.asarray(frame["weights"], dtype=np.float64)


def _pair_metrics(clean: pd.DataFrame, gated: pd.DataFrame) -> dict:
    cw, gw = _weights(clean), _weights(gated)
    js = {}
    overlap = {}
    for name in INFERENCE_PARAMS:
        if name not in clean.columns or name not in gated.columns:
            continue
        js[name] = js_divergence_1d(
            clean[name].to_numpy(), gated[name].to_numpy(), wx=cw, wy=gw
        )
        overlap[name] = bool(
            ci_overlap(param_summary(clean, [name])[name], param_summary(gated, [name])[name])
        )
    finite = np.asarray([v for v in js.values() if np.isfinite(v)], dtype=np.float64)
    return {
        "js_nats": js,
        "js_median": float(np.median(finite)) if finite.size else None,
        "js_iqr": float(np.subtract(*np.percentile(finite, [75, 25]))) if finite.size else None,
        "js_max": float(np.max(finite)) if finite.size else None,
        "ci90_overlap": overlap,
        "n_overlap": int(sum(overlap.values())),
    }


def _coverage_one(samples: pd.DataFrame, truth: dict) -> dict:
    w = _weights(samples)
    out = {}
    for name, value in truth.items():
        if name not in samples.columns or not np.isfinite(value):
            continue
        summary = param_summary(samples, [name])[name]
        # param_summary stores the 90% interval as lo/hi. Rebuild 50% from weights.
        x = np.asarray(samples[name], dtype=np.float64)
        if w is None:
            lo50, hi50 = np.quantile(x[np.isfinite(x)], [0.25, 0.75])
        else:
            from official_control import _weighted_quantiles

            lo50, hi50 = _weighted_quantiles(x, w, [0.25, 0.75])
        out[name] = {
            "truth": float(value),
            "in_50": bool(lo50 <= value <= hi50),
            "in_90": bool(summary["lo"] <= value <= summary["hi"]),
        }
    return out


def _save_samples(path: Path, result) -> None:
    keep = [c for c in list(INFERENCE_PARAMS) + ["weights"] if c in result.samples.columns]
    result.samples[keep].to_csv(path, index=False)


def _efficiency(result) -> dict:
    return {
        "sample_efficiency": None
        if result.sample_efficiency is None
        else float(result.sample_efficiency),
        "n_eff": None if result.n_eff is None else float(result.n_eff),
    }


def _sample_is(session, data, settings, fixed, n, batch, processes, *, require_is=True):
    result = session.sample_result(
        data, settings, fixed, num_samples=n, batch_size=batch
    )
    try:
        return run_importance_sampling(result, num_processes=int(processes)), {"ok": True}
    except IndexError:
        # sample_synthetic_phase indexes an empty within-prior set when every
        # draw falls outside the prior. That is a failed importance-sampling
        # run, not a posterior to compare.
        if require_is:
            raise
        logger.warning("Importance sampling rejected every draw; raw NPE kept.")
        return result, {"ok": False, "error": "no samples inside the prior"}


def _aggregate(rows: list[dict]) -> dict:
    js_med, js_max, eff_c, eff_g = [], [], [], []
    per_param = {name: [] for name in INFERENCE_PARAMS}
    n_failed = 0
    for row in rows:
        if row.get("importance_sampling") is False:
            n_failed += 1
            eff_c.append(row["efficiency_clean"]["sample_efficiency"])
            eff_g.append(0.0)
            continue
        metrics = row["metrics"]
        if metrics.get("js_median") is not None:
            js_med.append(metrics["js_median"])
        if metrics.get("js_max") is not None:
            js_max.append(metrics["js_max"])
        eff_c.append(row["efficiency_clean"]["sample_efficiency"])
        eff_g.append(row["efficiency_gated"]["sample_efficiency"])
        for name, value in metrics["js_nats"].items():
            if np.isfinite(value):
                per_param[name].append(value)
    def _summ(vals):
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
        if a.size == 0:
            return None
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
            "max": float(np.max(a)),
        }
    return {
        "n": len(rows),
        "n_importance_sampling_failed": n_failed,
        "js_of_cell_medians": _summ(js_med),
        "js_of_cell_maxes": _summ(js_max),
        "per_parameter_js": {k: _summ(v) for k, v in per_param.items() if v},
        "efficiency_clean": _summ(eff_c),
        "efficiency_gated": _summ(eff_g),
    }


def _coverage_rate(rows: list[dict], which: str) -> dict:
    hits50 = {name: [] for name in INFERENCE_PARAMS}
    hits90 = {name: [] for name in INFERENCE_PARAMS}
    for row in rows:
        block = row.get(which) or {}
        for name, rec in block.items():
            hits50[name].append(bool(rec["in_50"]))
            hits90[name].append(bool(rec["in_90"]))
    out = {}
    for name in INFERENCE_PARAMS:
        if not hits90[name]:
            continue
        out[name] = {
            "n": len(hits90[name]),
            "coverage_50": float(np.mean(hits50[name])),
            "coverage_90": float(np.mean(hits90[name])),
        }
    return out


def _run_real(session, args, outdir: Path) -> list[dict]:
    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.dingo_bns_demo import discover_assets, load_bns_checkpoint, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.stft_context import inband_rms
    from stress_gw170817 import _build_spec_stack, _detector_gates, _load_detector
    from stress_real_glitches import Transplanter, inject_excerpt_into_event
    from adapt import gravity_spy_io as gs

    cells = _select_real(args.real_csv, koi_only=not args.all_morphologies)
    logger.info("Real collapsed Koi Fish cells: %d", len(cells))
    assets = discover_assets()
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = dict(assets["fixed_context"])
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    base = build_domain_from_model_metadata(raw["metadata"], base=True)
    detectors = list(raw["metadata"]["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s", 4096.0))
    asds = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    td_clean = {}
    for det in detectors:
        td_clean[det], _, _ = load_full_event_td(assets, settings, det)
    from adapt.glitch_excision import analysis_crop_bounds

    _t, c0, c1 = analysis_crop_bounds(
        duration=duration, time_buffer=time_buffer, sample_rate=sample_rate
    )
    transplanter = Transplanter(
        asd_h1=asds["H1"],
        delta_f=float(base.delta_f),
        noise_std=float(base.noise_std),
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        clean_crop_h1=td_clean["H1"][c0:c1],
        rms_inband_h1=float(inband_rms(td_clean["H1"], sample_rate, f_min=f_min, f_max=f_max)),
    )
    import torch

    device = torch.device("cpu")
    model, det_raw = _load_detector(
        REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt", device
    )
    catalogue = pd.read_csv(args.real_catalogue or REAL_CAT)
    clean_path = outdir / "real_clean_samples.csv"
    if clean_path.is_file() and not args.overwrite:
        clean_samples = pd.read_csv(clean_path)
        clean_eff = json.loads((outdir / "real_clean_efficiency.json").read_text())
    else:
        logger.info("Sampling clean GW170817 reference")
        clean_result, _ = _sample_is(
            session, event.data, settings, fixed, args.num_samples, args.batch_size, args.num_processes
        )
        _save_samples(clean_path, clean_result)
        clean_eff = _efficiency(clean_result)
        (outdir / "real_clean_efficiency.json").write_text(json.dumps(clean_eff, indent=2))
        clean_samples = clean_result.samples

    rows = []
    for rec in cells.itertuples(index=False):
        dest = outdir / f"real_cell_{int(rec.cell_id)}.json"
        if dest.is_file() and not args.overwrite:
            rows.append(json.loads(dest.read_text()))
            continue
        cat = catalogue.iloc[(catalogue["event_time"] - float(rec.gs_event_time)).abs().argmin()]
        try:
            _cache = REPO_ROOT / "data" / "gravity_spy" / "excerpts" / f"H1_{float(cat['event_time']):.4f}_{int(sample_rate)}.npy"
            if _cache.is_file():
                raw_td = np.load(_cache)
            else:
                raw_td = gs.fetch_h1_excerpt(float(cat["event_time"]), sample_rate=sample_rate)
        except Exception as exc:
            logger.warning("Real cell %s excerpt fetch failed: %s", rec.cell_id, exc)
            continue
        excerpt = gs.extract_glitch_excerpt(
            raw_td,
            sample_rate=sample_rate,
            duration_s=float(cat["duration"]),
            f_min=f_min,
            f_max=f_max,
        )
        poison, _td_stft, meta = inject_excerpt_into_event(
            event,
            assets,
            excerpt=excerpt,
            mode="native",
            severity=float("nan"),
            t_rel=float(rec.t_rel),
            td_clean_full=td_clean,
            transplanter=transplanter,
            f_max=f_max,
            roll_off=roll_off,
        )
        spec, _ = _build_spec_stack(
            meta["td_stft"],
            asds_clean=asds,
            detectors=detectors,
            sample_rate=sample_rate,
            delta_f=float(base.delta_f),
            noise_std=float(base.noise_std),
            norm_stats=det_raw.get("norm_stats"),
            stft_kwargs=dict(det_raw.get("stft_kwargs") or {}),
        )
        _half = (
            float(args.force_gate_half_s)
            if args.force_gate_half_s is not None
            else float(det_raw.get("gate_half_s", 0.4))
        )
        from adapt.localize import select_gates

        gates, _loc = select_gates(
            lambda: _detector_gates(
                model,
                spec,
                detectors=detectors,
                crop_start=int(meta["crop_start"]),
                sample_rate=sample_rate,
                threshold=0.7283049106597901,
                gate_half_s=_half,
                device=device,
            ),
            td_by_det={"H1": meta["td_full"]["H1"]}, asds={"H1": asds["H1"]},
            sample_rate=sample_rate, f_max=f_max, pad_s=_half, detectors=["H1"],
        )
        gated = rebuild_event_from_gated_td(
            poison,
            td_by_det=meta["td_full"],
            gates=gates,
            sample_rate=sample_rate,
            roll_off=roll_off,
            f_max=f_max,
            original_asds=asds,
        )
        logger.info("Real cell %s gates=%d", rec.cell_id, len(gates))
        gated_result, is_meta = _sample_is(
            session, gated.data, settings, fixed, args.num_samples, args.batch_size, args.num_processes,
            require_is=False,
        )
        row = {
            "panel": "real_koi",
            "cell_id": int(rec.cell_id),
            "family": str(rec.family),
            "gs_event_time": float(rec.gs_event_time),
            "n_gates": len(gates),
            "importance_sampling": bool(is_meta["ok"]),
            "importance_sampling_error": is_meta.get("error"),
            "metrics": _pair_metrics(clean_samples, gated_result.samples),
            "efficiency_clean": clean_eff,
            "efficiency_gated": _efficiency(gated_result) if is_meta["ok"] else {
                "sample_efficiency": 0.0, "n_eff": 0.0
            },
        }
        dest.write_text(json.dumps(row, indent=2))
        rows.append(row)
    return rows


def _truth_from_theta(theta: dict, injected: dict) -> dict:
    truth = {}
    for name in INFERENCE_PARAMS:
        if name in theta and np.isfinite(float(theta[name])):
            truth[name] = float(theta[name])
        elif name in injected and np.isfinite(float(injected[name])):
            truth[name] = float(injected[name])
    return truth


def _run_synthetic(session, args, outdir: Path) -> list[dict]:
    import bilby
    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.dingo_bns_demo import (
        build_base_domain_injection,
        discover_assets,
        load_bns_checkpoint,
        load_event_dataset,
    )
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from stress_gw170817 import _build_spec_stack, _detector_gates, _load_detector
    from stress_synthetic_bns import (
        build_panel_cells,
        build_synthetic_event,
        draw_synthetic_theta,
        inject_glitch_on_synthetic,
    )

    selected = _select_synthetic()
    logger.info("Synthetic collapsed cells: %d", len(selected))
    archived = pd.read_csv(SYN_EVENTS).set_index("event_id")
    events_meta = pd.read_csv(SYN_EVENTS)
    thr_by_event = {
        int(r.event_id): float(r.thr_event) for r in events_meta.itertuples(index=False)
    }
    assets = discover_assets()
    ref = load_event_dataset(assets)
    settings = dict(ref.settings)
    demo_fixed = dict(assets["fixed_context"])
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    asds_ref = {d: np.asarray(ref.data["asds"][d]).copy() for d in detectors}
    n_base = len(base)
    for det in detectors:
        a = asds_ref[det]
        if len(a) < n_base:
            asds_ref[det] = np.pad(a, (0, n_base - len(a)), constant_values=1.0)
        elif len(a) > n_base:
            asds_ref[det] = a[:n_base]
    injection, _ = build_base_domain_injection(metadata)
    import torch

    device = torch.device("cpu")
    model, det_raw = _load_detector(
        REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt", device
    )
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    sample_rate = float(settings.get("f_s", 4096.0))
    rows = []
    clean_cache: dict = {}
    for event_id, group in selected.groupby("event_id"):
        event_id = int(event_id)
        ev_seed = 1_000_003 * event_id
        bilby.core.utils.random.seed(ev_seed)
        rng = np.random.default_rng(ev_seed)
        theta = draw_synthetic_theta(injection, rng, fixed_sky=demo_fixed)
        archived_mc = float(archived.loc[event_id, "truth_chirp_mass"])
        if abs(float(theta["chirp_mass"]) - archived_mc) > 1e-8:
            raise RuntimeError(
                f"event {event_id} chirp mass {theta['chirp_mass']} != archived {archived_mc}"
            )
        fixed = {
            "chirp_mass_proxy": float(theta["chirp_mass_proxy"]),
            "ra": float(demo_fixed["ra"]),
            "dec": float(demo_fixed["dec"]),
        }
        event, td_clean, _truth = build_synthetic_event(
            injection, theta, asds_ref=asds_ref, settings=settings, detectors=detectors
        )
        injected_params = dict(getattr(event, "data", {}).get("parameters", {}) or {})
        # build_synthetic_event does not keep the injection parameter dict.
        truth = _truth_from_theta(theta, injected_params)
        clean_csv = outdir / f"synth_event_{event_id}_clean.csv"
        clean_eff_path = outdir / f"synth_event_{event_id}_clean_efficiency.json"
        if event_id not in clean_cache and clean_csv.is_file() and clean_eff_path.is_file() and not args.overwrite:
            clean_cache[event_id] = (
                pd.read_csv(clean_csv),
                json.loads(clean_eff_path.read_text()),
                truth,
            )
        if event_id not in clean_cache:
            logger.info("Sampling synthetic clean event %s", event_id)
            clean_result, _ = _sample_is(
                session,
                event.data,
                settings,
                fixed,
                args.num_samples,
                args.batch_size,
                args.num_processes,
            )
            _save_samples(clean_csv, clean_result)
            eff = _efficiency(clean_result)
            clean_eff_path.write_text(json.dumps(eff, indent=2))
            clean_cache[event_id] = (clean_result.samples, eff, truth)
        clean_samples, clean_eff, truth = clean_cache[event_id]
        panel = build_panel_cells(event_id=event_id, master_seed=0)
        by_id = {int(c["cell_id"]): c for c in panel}
        for rec in group.itertuples(index=False):
            dest = outdir / f"synth_event_{event_id}_cell_{int(rec.cell_id)}.json"
            if dest.is_file() and not args.overwrite:
                rows.append(json.loads(dest.read_text()))
                continue
            cell = by_id[int(rec.cell_id)]
            poison, td_stft, meta = inject_glitch_on_synthetic(
                event.data, td_clean, cell, settings
            )
            spec, _ = _build_spec_stack(
                td_stft,
                asds_clean=asds_ref,
                detectors=detectors,
                sample_rate=sample_rate,
                delta_f=float(base.delta_f),
                noise_std=float(base.noise_std),
                norm_stats=det_raw.get("norm_stats"),
                stft_kwargs=dict(det_raw.get("stft_kwargs") or {}),
            )
            gates, _ = _detector_gates(
                model,
                spec,
                detectors=detectors,
                crop_start=int(meta["crop_start"]),
                sample_rate=sample_rate,
                threshold=thr_by_event.get(event_id, float(det_raw.get("threshold", 0.5))),
                gate_half_s=float(det_raw.get("gate_half_s", 0.4)),
                device=device,
            )
            gated = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds={d: np.asarray(event.data["asds"][d]).copy() for d in detectors},
            )
            logger.info(
                "Synthetic event %s cell %s family %s gates=%d",
                event_id,
                rec.cell_id,
                rec.family,
                len(gates),
            )
            gated_result, is_meta = _sample_is(
                session,
                gated.data,
                settings,
                fixed,
                args.num_samples,
                args.batch_size,
                args.num_processes,
                require_is=False,
            )
            row = {
                "panel": "synthetic",
                "event_id": event_id,
                "cell_id": int(rec.cell_id),
                "family": str(rec.family),
                "n_gates": len(gates),
                "importance_sampling": bool(is_meta["ok"]),
                "importance_sampling_error": is_meta.get("error"),
                "truth": truth,
                "metrics": _pair_metrics(clean_samples, gated_result.samples) if is_meta["ok"] else None,
                "coverage_clean": _coverage_one(clean_samples, truth),
                "coverage_gated": _coverage_one(gated_result.samples, truth) if is_meta["ok"] else None,
                "efficiency_clean": clean_eff,
                "efficiency_gated": _efficiency(gated_result) if is_meta["ok"] else {
                    "sample_efficiency": 0.0, "n_eff": 0.0
                },
            }
            dest.write_text(json.dumps(row, indent=2))
            rows.append(row)
    return rows


def main(argv=None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=resolve_under_repo, default="results/multi_param_validation_v1"
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--real-csv", type=resolve_under_repo, default=None,
                        help="real-panel results.csv to take collapsed cells from (default: stress_real_glitches_v2)")
    parser.add_argument("--real-catalogue", type=resolve_under_repo, default=None,
                        help="matching glitch_catalogue.csv (default: stress_real_glitches_v2)")
    parser.add_argument("--all-morphologies", action="store_true",
                        help="use every collapsed cell, not only Koi Fish")
    parser.add_argument("--force-gate-half-s", type=float, default=None,
                        help="gate half-width in s; overrides the value stored in the detector checkpoint")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from adapt.dingo_bns_demo import discover_assets

    assets = discover_assets()
    session = PESession(Path(assets["baseline_ckpt"]))
    real_rows = [] if args.skip_real else _run_real(session, args, outdir)
    synth_rows = [] if args.skip_synthetic else _run_synthetic(session, args, outdir)
    summary = {
        "sky_test": "skipped; public GW170817 checkpoint fixes ra=3.44616, dec=-0.408084",
        "num_samples": args.num_samples,
        "importance_sampling": True,
        "parameters": list(INFERENCE_PARAMS),
        "real_koi_fish": _aggregate(real_rows),
        "synthetic": _aggregate(synth_rows),
        "synthetic_coverage_clean": _coverage_rate(synth_rows, "coverage_clean"),
        "synthetic_coverage_gated": _coverage_rate(synth_rows, "coverage_gated"),
        "synthetic_truth_note": (
            "chirp_mass, mass_ratio, luminosity_distance, theta_jn, a_1, a_2, "
            "lambda_1, lambda_2, delta_chirp_mass, chirp_mass_proxy and geocent_time "
            "match the original numpy seed. tilt, phi and psi were not archived; they "
            "are redrawn from the prior with bilby seeded by the event id and saved here."
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", outdir / "summary.json")


if __name__ == "__main__":
    main()
