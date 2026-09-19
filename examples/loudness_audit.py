#!/usr/bin/env python3
"""Loudness audit: put every injected glitch on one scale (whitened optimal SNR).

rho_w = sqrt(4 * sum |g(f)|^2 / S_n(f) df) over [f_min, f_max] with the
GW170817 analysis ASD of the injected detector (H1 for 4/5 of the archived
cells, L1 for the rest).

Computes rho_w for
  (i)   every archived synthetic cell in results/stress_test_excision_v1
        (re-synthesised from params_json + seed; no PE is re-run),
  (ii)  the official-control sine-Gaussian (f0=100 Hz, Q=5, 8x in-band RMS),
  (iii) every cached real Gravity Spy excerpt at native loudness,
and writes results/loudness_audit_v1/{loudness.csv, summary.json} plus
paper/figures/figS3_loudness.pdf.

With --threshold-rho it also reports (Task 5D) the fraction of H1 O3 Gravity
Spy triggers (ml_confidence >= 0.95, all labels) whose loudness exceeds that
threshold, both on the raw Omicron-SNR axis and after converting rho_w to
Omicron SNR with the empirical ratio measured on the real excerpts.

Usage::

    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/loudness_audit.py --outdir results/loudness_audit_v1
    python examples/loudness_audit.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "examples", REPO_ROOT / "src", REPO_ROOT / "DINGO-BNS" / "dingo", REPO_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("loudness_audit")

DEFAULT_OUTDIR = REPO_ROOT / "results" / "loudness_audit_v1"
DEFAULT_FIGDIR = REPO_ROOT / "paper" / "figures"
ARCHIVE = REPO_ROOT / "results" / "stress_test_excision_v1" / "results.csv"
SELECTED = REPO_ROOT / "data" / "gravity_spy" / "selected_h1_o3.csv"


def _setup_event(args):
    from adapt.dingo_bns_demo import discover_assets, load_bns_checkpoint, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from adapt.stft_context import inband_rms
    from dingo.gw.domains import build_domain_from_model_metadata

    assets = discover_assets(baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    ctx = {
        "assets": assets,
        "event": event,
        "settings": settings,
        "detectors": detectors,
        "sample_rate": float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"]),
        "delta_f": float(base_domain.delta_f),
        "noise_std": float(base_domain.noise_std),
        "f_min": float(settings.get("f_min", 23.0)),
        "f_max": float(settings.get("f_max", 1535.3046875)),
        "roll_off": float(settings.get("roll_off", 0.4)),
        "duration": float(settings.get("T", 128.0)),
        "time_buffer": float(settings.get("time_buffer", 2.0)),
        "asds": {d: np.asarray(event.data["asds"][d], dtype=np.float64).copy() for d in detectors},
    }
    ctx["td_clean"] = {}
    ctx["rms_inband"] = {}
    for det in ("H1", "L1"):
        td, _, _ = load_full_event_td(assets, settings, det)
        ctx["td_clean"][det] = td
        ctx["rms_inband"][det] = float(inband_rms(td, ctx["sample_rate"], f_min=ctx["f_min"], f_max=ctx["f_max"]))
    return ctx


def _rho(ctx, g: np.ndarray, det: str) -> float:
    from adapt.loudness import whitened_optimal_snr

    return whitened_optimal_snr(
        g, ctx["sample_rate"], asd=ctx["asds"][det], delta_f=ctx["delta_f"],
        f_min=ctx["f_min"], f_max=ctx["f_max"], roll_off=ctx["roll_off"],
    )


# ---------------------------------------------------------------------------
# (i) archived synthetic cells
# ---------------------------------------------------------------------------


def resynthesize_archived(ctx, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Re-create each archived glitch exactly as ``stress_gw170817.inject_spec_into_event``
    did (same RNG seed and consumption order), without running PE."""
    from adapt.glitch_augmentation import GlitchSpec, synthesize_glitch_td

    rows = []
    for _, r in df.iterrows():
        det = str(r["detectors"]).split(",")[0]
        t_rel = float(r["t_rel"])
        t_peak = (ctx["duration"] - ctx["time_buffer"]) + t_rel
        spec = GlitchSpec(
            family=str(r["family"]), detectors=[det], t_rel=t_peak - 0.5 * ctx["duration"],
            severity=float(r["severity"]), params=json.loads(r["params_json"]),
            asd_policy=str(r["asd_policy"]), held_out=bool(r["held_out"]),
        )
        rng = np.random.default_rng(int(r["seed"]))
        td = ctx["td_clean"][det]
        g = synthesize_glitch_td(len(td), ctx["sample_rate"], spec, rng, rms=ctx["rms_inband"][det])
        rows.append({
            "source": "synthetic_archive", "cell_id": int(r["cell_id"]), "family": r["family"],
            "held_out": bool(r["held_out"]), "severity": float(r["severity"]), "asd_policy": r["asd_policy"],
            "seed": int(r["seed"]), "detector": det, "t_rel": t_rel, "rho_w": _rho(ctx, g, det),
            "poison_collapsed": bool(r["poison_collapsed"]) if str(r["poison_collapsed"]) != "nan" else None,
            "gated_recovers": bool(r["gated_recovers"]) if str(r["gated_recovers"]) != "nan" else None,
            "oracle_recovers": bool(r["oracle_recovers"]) if str(r["oracle_recovers"]) != "nan" else None,
            "gs_label": "", "gs_snr": float("nan"), "rho_w_over_gs_snr": float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# (ii) official-control sine-Gaussian
# ---------------------------------------------------------------------------


def control_sine_gaussian(ctx) -> Dict[str, Any]:
    from adapt.stft_context import sine_gaussian_glitch

    det = "H1"
    td = ctx["td_clean"][det]
    t_peak = (ctx["duration"] - ctx["time_buffer"]) - 1.0
    amp = 8.0 * ctx["rms_inband"][det]
    g = sine_gaussian_glitch(len(td), ctx["sample_rate"], t_peak=t_peak, f0=100.0, q=5.0, amplitude=amp)
    rep = REPO_ROOT / "results" / "dingo_official_control" / "comparison_report.json"
    collapsed = None
    if rep.is_file():
        collapsed = bool(json.loads(rep.read_text()).get("poison_collapsed"))
    return {
        "source": "official_control", "cell_id": -1, "family": "sine_gaussian", "held_out": False,
        "severity": 8.0, "asd_policy": "welch", "seed": 0, "detector": det, "t_rel": -1.0,
        "rho_w": _rho(ctx, g, det), "poison_collapsed": collapsed, "gated_recovers": True if collapsed else None,
        "oracle_recovers": None, "gs_label": "", "gs_snr": float("nan"), "rho_w_over_gs_snr": float("nan"),
    }


# ---------------------------------------------------------------------------
# (iii) real excerpts at native loudness
# ---------------------------------------------------------------------------


def real_native(ctx, selected: pd.DataFrame, outdir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from adapt import gravity_spy_io as gs
    from adapt.glitch_excision import analysis_crop_bounds
    from stress_real_glitches import Transplanter, fetch_excerpts

    _, cs0, ce0 = analysis_crop_bounds(duration=ctx["duration"], time_buffer=ctx["time_buffer"], sample_rate=ctx["sample_rate"])
    tp = Transplanter(
        asd_h1=ctx["asds"]["H1"], delta_f=ctx["delta_f"], noise_std=ctx["noise_std"], sample_rate=ctx["sample_rate"],
        f_min=ctx["f_min"], f_max=ctx["f_max"], clean_crop_h1=ctx["td_clean"]["H1"][cs0:ce0],
        rms_inband_h1=ctx["rms_inband"]["H1"],
    )
    ex = fetch_excerpts(selected, sample_rate=ctx["sample_rate"], f_min=ctx["f_min"], f_max=ctx["f_max"], outdir=outdir)
    rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    n = ctx["td_clean"]["H1"].size
    t_peak = (ctx["duration"] - ctx["time_buffer"]) - 1.0
    fetched_ids = set()
    for i, item in sorted(ex.items()):
        r, e = item["row"], item["excerpt"]
        fetched_ids.add(str(r.get("gravityspy_id", "")))
        energy = e.native_energy_w()
        rec = {
            "gravityspy_id": str(r.get("gravityspy_id", "")),
            "gs_label": str(r["ml_label"]),
            "gs_event_time": float(r["event_time"]),
            "gs_snr": float(r.get("snr", float("nan"))),
        }
        if energy <= 0:
            rec["reason"] = (
                f"no measurable whitened energy after taper-aware excess "
                f"(excess={e.excess_energy_w:.3f}, denoised={e.denoised_energy_w:.3f})"
            )
            dropped.append(rec)
            continue
        g0 = tp.colour(e.injection_waveform())
        k = tp.scale_native(g0, energy)
        g = gs.place_series(g0 * k, n_samples=n, sample_rate=ctx["sample_rate"], t_peak=t_peak)
        rho = _rho(ctx, g, "H1")
        if not np.isfinite(rho) or rho <= 0:
            rec["reason"] = f"rho_w={rho} after native scaling (energy={energy:.3f})"
            dropped.append(rec)
            continue
        gs_snr = float(r.get("snr", float("nan")))
        rows.append({
            "source": "real_native", "cell_id": int(i), "family": str(r["ml_label"]), "held_out": True,
            "severity": float("nan"), "asd_policy": "stationary", "seed": 0, "detector": "H1", "t_rel": -1.0,
            "rho_w": rho, "poison_collapsed": None, "gated_recovers": None, "oracle_recovers": None,
            "gs_label": str(r["ml_label"]), "gs_snr": gs_snr,
            "gs_event_time": float(r["event_time"]), "native_snr_w": float(np.sqrt(energy)),
            "rho_w_over_gs_snr": (rho / gs_snr) if (np.isfinite(gs_snr) and gs_snr > 0) else float("nan"),
        })
    fail_path = outdir / "fetch_failures.csv"
    if fail_path.is_file():
        fails = pd.read_csv(fail_path)
        if len(fails):
            for _, fr in fails.iterrows():
                dropped.append({
                    "gravityspy_id": str(fr.get("gravityspy_id", "")),
                    "gs_label": str(fr.get("ml_label", "")),
                    "gs_event_time": float(fr["event_time"]) if "event_time" in fr and pd.notna(fr["event_time"]) else None,
                    "gs_snr": float(fr["snr"]) if "snr" in fr and pd.notna(fr["snr"]) else None,
                    "reason": str(fr.get("error", "fetch failed")),
                })
    for _, r in selected.reset_index(drop=True).iterrows():
        gid = str(r.get("gravityspy_id", ""))
        if gid and gid not in fetched_ids and not any(d.get("gravityspy_id") == gid for d in dropped):
            dropped.append({
                "gravityspy_id": gid, "gs_label": str(r["ml_label"]),
                "gs_event_time": float(r["event_time"]), "gs_snr": float(r.get("snr", float("nan"))),
                "reason": "not in fetch_excerpts output (skipped or failed without a row)",
            })
    return rows, dropped


# ---------------------------------------------------------------------------
# Task 5D: population fraction above a loudness threshold
# ---------------------------------------------------------------------------


def population_fraction(raw_paths: Sequence[Path], *, thr_omicron: float, min_conf: float = 0.95) -> Dict[str, Any]:
    from adapt import gravity_spy_io as gs

    frames = [gs.read_gravity_spy_csv(Path(p)) for p in raw_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ifo"].astype(str).str.upper() == "H1") & (df["ml_confidence"].astype(float) >= min_conf)]
    df = df[~df["ml_label"].astype(str).isin(["No_Glitch", "None_of_the_Above"])]
    snr = df["snr"].astype(float).to_numpy()
    out = {
        "n_triggers": int(len(df)),
        "min_confidence": float(min_conf),
        "omicron_snr_threshold": float(thr_omicron),
        "fraction_above": float(np.mean(snr > thr_omicron)) if len(snr) else None,
        "n_above": int(np.sum(snr > thr_omicron)),
        "snr_quantiles": {q: float(np.quantile(snr, q)) for q in (0.5, 0.9, 0.99, 0.999)} if len(snr) else {},
        "by_label_fraction_above": (
            df.assign(above=snr > thr_omicron).groupby("ml_label")["above"].agg(["mean", "size"]).reset_index()
            .rename(columns={"mean": "fraction_above", "size": "n"}).to_dict(orient="records")
        ),
    }
    return out


# ---------------------------------------------------------------------------


def make_figure(df: pd.DataFrame, figdir: Path, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    syn = df[df["source"] == "synthetic_archive"].copy()
    syn = syn[syn["poison_collapsed"].notna()]
    real = df[df["source"] == "real_native"]
    ctrl = df[df["source"] == "official_control"]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    rng = np.random.default_rng(0)
    cols = {3.0: "#9ecae1", 6.0: "#4292c6", 10.0: "#08306b"}
    for sev, sub in syn.groupby("severity"):
        y = sub["poison_collapsed"].astype(float).to_numpy() + rng.uniform(-0.06, 0.06, len(sub))
        ax.scatter(sub["rho_w"], y, s=9, alpha=0.65, color=cols.get(float(sev), "k"), label=f"synthetic, severity {int(sev)}", zorder=3)
    # binned collapse fraction
    if len(syn):
        edges = np.logspace(np.log10(max(syn["rho_w"].min(), 1.0)), np.log10(syn["rho_w"].max() * 1.01), 9)
        centers, fracs = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (syn["rho_w"] >= a) & (syn["rho_w"] < b)
            if m.sum() >= 5:
                centers.append(np.sqrt(a * b)); fracs.append(syn.loc[m, "poison_collapsed"].astype(float).mean())
        ax.plot(centers, fracs, "-", color="#c0392b", lw=1.2, label="collapse fraction (binned)")
    for _, r in real.iterrows():
        ax.axvline(r["rho_w"], ymin=0.0, ymax=0.18, color="#2a9d8f", lw=0.9, alpha=0.8)
    if len(real):
        ax.plot([], [], color="#2a9d8f", lw=0.9, label=f"real O3 glitches, native (n={len(real)})")
    if len(ctrl):
        ax.axvline(float(ctrl["rho_w"].iloc[0]), color="k", ls=":", lw=0.9, label="official-control SG")
    ax.set_xscale("log")
    ax.set_xlabel(r"whitened optimal SNR $\rho_w$ of injected glitch")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no collapse", "collapse"])
    ax.set_ylim(-0.15, 1.15)
    ax.legend(frameon=False, fontsize=5.8, loc="center left")
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "figS3_loudness.pdf"); fig.savefig(figdir / "figS3_loudness.png", dpi=300)
    fig.savefig(outdir / "figS3_loudness.pdf")
    plt.close(fig)


def population_only(args: argparse.Namespace) -> None:
    """Task 5D from an existing loudness.csv / summary.json (no re-fetch)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = Path(args.outdir)
    df = pd.read_csv(outdir / "loudness.csv")
    summary = json.loads((outdir / "summary.json").read_text())
    real = df[df["source"] == "real_native"]
    ok = real[(real["rho_w"] > 0) & real["gs_snr"].notna() & (real["gs_snr"] > 0)]
    ratio = float(np.median(ok["rho_w"] / ok["gs_snr"])) if len(ok) else None
    raw_paths = [Path(args.raw_dir) / n for n in ("H1_O3a.csv", "H1_O3b.csv") if (Path(args.raw_dir) / n).is_file()]
    thr_rho = float(args.threshold_rho)
    pop = {
        "threshold_rho_w": thr_rho,
        "direct_omicron_axis": population_fraction(raw_paths, thr_omicron=thr_rho),
        "converted_omicron_axis": population_fraction(raw_paths, thr_omicron=thr_rho / ratio) if ratio else None,
        "median_ratio_rho_w_over_omicron_snr": ratio,
        "conversion_note": (
            "rho_w (whitened optimal SNR against the GW170817 H1 ASD) and Omicron SNR are different statistics; "
            "the 'converted' entry divides the rho_w threshold by the median rho_w/Omicron-SNR ratio measured on the "
            "real excerpts at native loudness."
        ),
    }
    if args.threshold_rho_grid is not None:
        pop["grid_threshold_rho_w"] = float(args.threshold_rho_grid)
        pop["direct_omicron_axis_grid"] = population_fraction(raw_paths, thr_omicron=float(args.threshold_rho_grid))
    summary["population_above_threshold"] = pop
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("population fraction above rho_w=%.0f: direct %.4f (n=%d); converted (ratio %.2f -> Omicron %.0f) %s",
                thr_rho, pop["direct_omicron_axis"]["fraction_above"], pop["direct_omicron_axis"]["n_triggers"],
                ratio or float("nan"), thr_rho / ratio if ratio else float("nan"),
                pop["converted_omicron_axis"]["fraction_above"] if ratio else None)


def _annotate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rho_w_over_gs_snr" not in df.columns:
        df["rho_w_over_gs_snr"] = np.nan
    real = (df["source"] == "real_native") & (df["rho_w"] > 0) & df["gs_snr"].notna() & (df["gs_snr"] > 0)
    df.loc[real, "rho_w_over_gs_snr"] = df.loc[real, "rho_w"] / df.loc[real, "gs_snr"]
    return df


def _ratio_by_label(real: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ok = real[(real["rho_w"] > 0) & real["gs_snr"].notna() & (real["gs_snr"] > 0)].copy()
    if "rho_w_over_gs_snr" not in ok.columns:
        ok["rho_w_over_gs_snr"] = ok["rho_w"] / ok["gs_snr"]
    for lab, sub in ok.groupby("gs_label"):
        out[str(lab)] = {
            "n": int(len(sub)),
            "median_rho_w_over_gs_snr": float(sub["rho_w_over_gs_snr"].median()),
        }
    return out


def write_o3_snr_tail(raw_paths: Sequence[Path], dest: Path, *, min_conf: float = 0.95) -> Dict[str, Any]:
    """Task 5D: fraction/count of H1 O3 triggers with gs_snr above high thresholds."""
    from adapt import gravity_spy_io as gs

    frames = [gs.read_gravity_spy_csv(Path(p)) for p in raw_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ifo"].astype(str).str.upper() == "H1") & (df["ml_confidence"].astype(float) >= min_conf)]
    df = df[~df["ml_label"].astype(str).isin(["No_Glitch", "None_of_the_Above"])]
    snr = df["snr"].astype(float)
    thresholds = (300, 500, 1000, 2000)
    top8 = df["ml_label"].value_counts().head(8).index.tolist()

    def _block(mask_df: pd.DataFrame) -> Dict[str, Any]:
        s = mask_df["snr"].astype(float)
        rec: Dict[str, Any] = {"n": int(len(mask_df))}
        for thr in thresholds:
            n_above = int((s >= thr).sum())
            rec[f"snr_ge_{thr}"] = {
                "n": n_above,
                "fraction": float(n_above / len(mask_df)) if len(mask_df) else None,
            }
        return rec

    out: Dict[str, Any] = {
        "min_confidence": float(min_conf),
        "n_triggers": int(len(df)),
        "thresholds": list(thresholds),
        "overall": _block(df),
        "top8_labels": top8,
        "by_label": {str(lab): _block(df[df["ml_label"] == lab]) for lab in top8},
        "snr_quantiles": {str(q): float(np.quantile(snr, q)) for q in (0.5, 0.9, 0.99, 0.999)} if len(snr) else {},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    logger.info("wrote %s (n=%d H1 triggers, ml_confidence>=%.2f)", dest, out["n_triggers"], min_conf)
    return out


def _write_audit_outputs(df: pd.DataFrame, dropped: List[Dict[str, Any]], args: argparse.Namespace, outdir: Path) -> Dict[str, Any]:
    df = _annotate_ratio(df)
    df.to_csv(outdir / "loudness.csv", index=False)
    syn = df[(df["source"] == "synthetic_archive") & df["poison_collapsed"].notna()]
    by_sev = {}
    for sev, sub in syn.groupby("severity"):
        by_sev[str(float(sev))] = {
            "n": int(len(sub)),
            "rho_w_min": float(sub["rho_w"].min()), "rho_w_median": float(sub["rho_w"].median()),
            "rho_w_max": float(sub["rho_w"].max()),
            "poison_collapse_rate": float(sub["poison_collapsed"].astype(float).mean()),
        }
    by_fam = {}
    for fam, sub in syn.groupby("family"):
        by_fam[str(fam)] = {"n": int(len(sub)), "rho_w_min": float(sub["rho_w"].min()),
                            "rho_w_median": float(sub["rho_w"].median()), "rho_w_max": float(sub["rho_w"].max())}
    real = df[df["source"] == "real_native"]
    ok = real[(real["rho_w"] > 0) & real["gs_snr"].notna() & (real["gs_snr"] > 0)]
    ratio = float(np.median(ok["rho_w_over_gs_snr"])) if len(ok) else None
    by_label = _ratio_by_label(real)
    summary: Dict[str, Any] = {
        "definition": "rho_w = sqrt(4 sum |g(f)|^2 / ASD(f)^2 df) over [f_min, f_max], GW170817 analysis ASD of the injected IFO",
        "n_synthetic": int(len(syn)),
        "synthetic_by_severity": by_sev,
        "synthetic_by_family": by_fam,
        "official_control_sg_rho_w": float(df[df["source"] == "official_control"]["rho_w"].iloc[0]) if (df["source"] == "official_control").any() else None,
        "real_native": {
            "n": int(len(real)),
            "rho_w_min": float(real["rho_w"].min()) if len(real) else None,
            "rho_w_median": float(real["rho_w"].median()) if len(real) else None,
            "rho_w_max": float(real["rho_w"].max()) if len(real) else None,
            "median_ratio_rho_w_over_omicron_snr": ratio,
            "median_rho_w_over_gs_snr_by_label": by_label,
            "n_zero_rho_w": int((real["rho_w"] <= 0).sum()) if len(real) else 0,
        },
        "dropped_rows": dropped,
        "extraction_fix": (
            "Four real_native rows previously had rho_w=0 because extract_glitch_excerpt "
            "subtracted n_win * var from a Tukey-tapered window (systematically negative "
            "excess). Excess is now taper-aware; native scale falls back to denoised_energy_w "
            "when excess<=0. Rows that still cannot be recovered are listed in dropped_rows."
        ),
    }
    if args.threshold_rho is not None:
        raw_paths = [Path(args.raw_dir) / n for n in ("H1_O3a.csv", "H1_O3b.csv") if (Path(args.raw_dir) / n).is_file()]
        thr_rho = float(args.threshold_rho)
        thr_omicron_equiv = thr_rho / ratio if ratio else float("nan")
        pop = {
            "threshold_rho_w": thr_rho,
            "direct_omicron_axis": population_fraction(raw_paths, thr_omicron=thr_rho),
            "converted_omicron_axis": population_fraction(raw_paths, thr_omicron=thr_omicron_equiv) if ratio else None,
            "conversion_note": (
                "rho_w (whitened optimal SNR against the GW170817 H1 ASD) and Omicron SNR are different statistics; "
                "the 'converted' entry divides the rho_w threshold by the median rho_w/Omicron-SNR ratio measured on the "
                "real excerpts at native loudness."
            ),
        }
        if args.threshold_rho_grid is not None:
            pop["grid_threshold_rho_w"] = float(args.threshold_rho_grid)
            pop["direct_omicron_axis_grid"] = population_fraction(raw_paths, thr_omicron=float(args.threshold_rho_grid))
        summary["population_above_threshold"] = pop
    raw_paths = [Path(args.raw_dir) / n for n in ("H1_O3a.csv", "H1_O3b.csv") if (Path(args.raw_dir) / n).is_file()]
    if raw_paths:
        summary["o3_snr_tail"] = write_o3_snr_tail(raw_paths, outdir / "o3_snr_tail.json")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    make_figure(df, Path(args.figdir), outdir)
    (outdir / "REPRODUCE.md").write_text(f"""# Reproduce the loudness audit

```bash
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/loudness_audit.py --outdir {outdir} [--threshold-rho <rho_w from collapse_threshold_v1>]
python examples/loudness_audit.py --repair-real --outdir {outdir}
python examples/loudness_audit.py --snr-tail-only --outdir {outdir}
```

Re-synthesises every archived cell of `results/stress_test_excision_v1/results.csv`
from `params_json` + `seed` (identical RNG consumption to `stress_gw170817.inject_spec_into_event`),
the official-control sine-Gaussian, and the cached real Gravity Spy excerpts at native loudness,
and reports the whitened optimal SNR `rho_w` of each injected glitch. No posterior sampling is run.

`--repair-real` keeps synthetic/control rows and re-extracts only the real Gravity Spy
excerpts (taper-aware excess + denoised fallback). Unrecoverable catalogue rows are
recorded in `summary.json` under `dropped_rows`.
""")
    logger.info("summary: %s", json.dumps({k: summary[k] for k in ("official_control_sg_rho_w", "real_native", "dropped_rows") if k in summary}, indent=1, default=str))
    return summary


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ctx = _setup_event(args)

    arch = pd.read_csv(args.archive)
    arch = arch[arch["error"].fillna("") == ""]
    if args.smoke:
        arch = arch.head(6)
    rows = resynthesize_archived(ctx, arch)
    logger.info("re-synthesised %d archived cells", len(rows))
    rows.append(control_sine_gaussian(ctx))
    logger.info("official-control SG rho_w = %.1f", rows[-1]["rho_w"])

    dropped: List[Dict[str, Any]] = []
    if not args.skip_real:
        from adapt import gravity_spy_io as gs

        selected = gs.load_selected_catalogue(Path(args.selected_csv))
        if args.smoke:
            selected = selected.head(2)
        real_rows, dropped = real_native(ctx, selected, outdir)
        rows.extend(real_rows)

    _write_audit_outputs(pd.DataFrame(rows), dropped, args, outdir)


def repair_real(args: argparse.Namespace) -> None:
    """Re-extract real_native rows; keep synthetic/control from existing loudness.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt import gravity_spy_io as gs

    outdir = Path(args.outdir)
    csv_path = outdir / "loudness.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    prev = pd.read_csv(csv_path)
    kept = prev[prev["source"] != "real_native"].copy()
    ctx = _setup_event(args)
    selected = gs.load_selected_catalogue(Path(args.selected_csv))
    if args.smoke:
        selected = selected.head(2)
    real_rows, dropped = real_native(ctx, selected, outdir)
    df = pd.concat([kept, pd.DataFrame(real_rows)], ignore_index=True)
    logger.info("repair-real: kept %d non-real rows, recovered %d real, dropped %d", len(kept), len(real_rows), len(dropped))
    _write_audit_outputs(df, dropped, args, outdir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--selected-csv", type=Path, default=SELECTED)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "gravity_spy" / "raw")
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--threshold-rho", type=float, default=None, help="Task 5D: report population fraction above this rho_w")
    p.add_argument("--threshold-rho-grid", type=float, default=None, help="optional second (grid-point) threshold")
    p.add_argument("--population-only", action="store_true", help="Task 5D only, from existing loudness.csv")
    p.add_argument("--repair-real", action="store_true", help="re-extract real_native rows only")
    p.add_argument("--snr-tail-only", action="store_true", help="write o3_snr_tail.json from cached H1 O3 tables")
    p.add_argument("--skip-real", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    _a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if _a.snr_tail_only:
        raw_paths = [Path(_a.raw_dir) / n for n in ("H1_O3a.csv", "H1_O3b.csv") if (Path(_a.raw_dir) / n).is_file()]
        write_o3_snr_tail(raw_paths, Path(_a.outdir) / "o3_snr_tail.json")
    elif _a.population_only:
        population_only(_a)
    elif _a.repair_real:
        repair_real(_a)
    else:
        run(_a)
