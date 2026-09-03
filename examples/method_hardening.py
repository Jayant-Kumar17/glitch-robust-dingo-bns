#!/usr/bin/env python3
"""Journal method hardening: ablation + oracle gap + runtime.

Produces paper-ready methodology artifacts under
``results/journal_method_hardening_v1/``:

1. 16-cell × 5-arm ablation on GW170817
2. Oracle-vs-detector gap (aggregate real 240-cell + re-run synthetic failures)
3. Runtime / cost table on ablation cells

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python -u examples/method_hardening.py \\
      --num-samples 512 --batch-size 256 --device cpu \\
      --outdir results/journal_method_hardening_v1
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

DEFAULT_OUTDIR = REPO_ROOT / "results" / "journal_method_hardening_v1"
DEFAULT_DETECTOR = (
    REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
)
DEFAULT_REAL_RESULTS = REPO_ROOT / "results" / "stress_test_excision_v1" / "results.csv"
DEFAULT_SYNTH_FAILURES = (
    REPO_ROOT / "results" / "stress_test_synthetic_bns_v1" / "failures.csv"
)
DEFAULT_SYNTH_EVENTS = REPO_ROOT / "results" / "stress_test_synthetic_bns_v1" / "events.csv"

ABLATION_FAMILIES = (
    "sine_gaussian",
    "broadband_burst",
    "scattered_light",
    "ringing",
)
ABLATION_SEVERITIES = (6.0, 10.0)
ARMS = (
    "poison_welch",
    "glitch_orig_asd",
    "gate_welch",
    "adapt_full",
    "fft_replace",
)

logger = logging.getLogger("journal_method_hardening")


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    new_file = not path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def build_ablation_cells(
    *,
    master_seed: int = 0,
    max_cells: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from adapt.glitch_augmentation import HELD_OUT_FAMILIES
    from stress_gw170817 import _sample_family_params

    held = set(HELD_OUT_FAMILIES)
    cells: List[Dict[str, Any]] = []
    cell_id = 0
    for family in ABLATION_FAMILIES:
        for sev in ABLATION_SEVERITIES:
            # Two seeds per (family, sev) → 16 cells total.
            for k in range(2):
                seed = int(master_seed) + 50_000 * cell_id + 17 * k
                rng = np.random.default_rng(seed)
                if family == "sine_gaussian" and cell_id == 0:
                    # Canonical-style SG cell.
                    params = {"f0": 100.0, "q": 5.0}
                    t_rel = -1.0
                else:
                    params = _sample_family_params(family, rng)
                    t_rel = float(rng.uniform(-1.5, -0.3))
                cells.append(
                    {
                        "cell_id": cell_id,
                        "family": family,
                        "held_out": family in held,
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


def aggregate_oracle_gap_real(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    for c in (
        "gated_recovers",
        "oracle_recovers",
        "oracle_only_recovery",
        "detector_fired",
        "held_out",
    ):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    fail = df[df["gated_recovers"] == False].copy()
    out: Dict[str, Any] = {
        "source": str(path),
        "n_total": int(len(df)),
        "n_gated_failures": int(len(fail)),
        "among_failures": {
            "oracle_recovers_rate": float(fail["oracle_recovers"].mean())
            if len(fail)
            else None,
            "oracle_only_recovery_rate": float(fail["oracle_only_recovery"].mean())
            if len(fail) and "oracle_only_recovery" in fail.columns
            else None,
            "detector_fired_rate": float(fail["detector_fired"].mean())
            if len(fail)
            else None,
        },
        "by_family": [],
    }
    if len(fail):
        g = (
            fail.groupby("family")
            .agg(
                n=("gated_recovers", "size"),
                oracle_recovers=("oracle_recovers", "mean"),
                oracle_only=("oracle_only_recovery", "mean")
                if "oracle_only_recovery" in fail.columns
                else ("gated_recovers", "size"),
                detector_fired=("detector_fired", "mean"),
            )
            .reset_index()
        )
        # Fix oracle_only column if we fell back
        if "oracle_only_recovery" in fail.columns:
            rows = []
            for fam, sub in fail.groupby("family"):
                rows.append(
                    {
                        "family": fam,
                        "n": int(len(sub)),
                        "oracle_recovers": float(sub["oracle_recovers"].mean()),
                        "oracle_only_recovery": float(sub["oracle_only_recovery"].mean()),
                        "detector_fired": float(sub["detector_fired"].mean()),
                    }
                )
            out["by_family"] = rows
        else:
            out["by_family"] = g.to_dict(orient="records")
    return out


def write_report(
    outdir: Path,
    ablation_df: pd.DataFrame,
    ablation_summary: Dict[str, Any],
    oracle_real: Dict[str, Any],
    oracle_synth: Dict[str, Any],
    runtime: Dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        logger.exception("matplotlib missing — skip PDF")
        return

    pdf_path = outdir / "method_hardening_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Ablation recovery by arm
        fig, ax = plt.subplots(figsize=(8, 4.5))
        arms = list(ARMS)
        rates = [
            ablation_summary["by_arm"].get(a, {}).get("recovers_like_clean", 0.0) or 0.0
            for a in arms
        ]
        ax.bar(arms, [100 * r for r in rates], color="#1f6f8b", edgecolor="none")
        ax.set_ylabel("Recovers like clean (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Ablation: recovery rate by arm (16 GW170817 cells)")
        plt.xticks(rotation=20, ha="right")
        for i, r in enumerate(rates):
            ax.text(i, 100 * r + 2, f"{100 * r:.0f}%", ha="center", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Median d_L by arm (box-ish via scatter of medians)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if len(ablation_df):
            for i, arm in enumerate(arms):
                sub = ablation_df[ablation_df["arm"] == arm]
                y = sub["med"].astype(float).to_numpy()
                ax.scatter(np.full_like(y, i, dtype=float), y, alpha=0.65, s=22)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=20, ha="right")
            ax.set_ylabel("d_L median (Mpc)")
            ax.set_title("Ablation: per-cell d_L medians by arm")
            ax.axhline(34, color="#888", ls="--", lw=0.8, label="~clean GW170817")
            ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Oracle gap
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        real_fail = oracle_real.get("n_gated_failures") or 0
        real_rate = (oracle_real.get("among_failures") or {}).get(
            "oracle_only_recovery_rate"
        )
        axes[0].bar(
            ["Oracle-only\namong failures"],
            [100 * (real_rate or 0)],
            color="#0b6e4f",
        )
        axes[0].set_ylim(0, 110)
        axes[0].set_ylabel("%")
        axes[0].set_title(f"Real GW170817 stress failures (n={real_fail})")

        synth_counts = oracle_synth.get("classification_counts") or {}
        labels = list(synth_counts.keys()) or ["none"]
        vals = [synth_counts.get(k, 0) for k in labels]
        axes[1].bar(labels, vals, color="#b33a3a")
        axes[1].set_title(
            f"Synthetic failure classes (n={oracle_synth.get('n_failures', 0)})"
        )
        axes[1].tick_params(axis="x", rotation=15)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Runtime table page
        fig = plt.figure(figsize=(8.5, 6))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.axis("off")
        ax.text(0.0, 0.95, "Runtime / cost (ablation cells)", fontsize=14, fontweight="bold")
        lines = ["stage                          mean_s    std_s", "-" * 48]
        for stage, st in (runtime.get("stages") or {}).items():
            lines.append(
                f"{stage:28s} {st.get('mean', float('nan')):8.3f} {st.get('std', float('nan')):8.3f}"
            )
        oh = runtime.get("adapt_overhead_vs_poison_pe")
        if oh is not None:
            lines.append("")
            lines.append(f"ADAPT overhead vs poison PE: {100 * oh:.1f}% of poison PE time")
        ax.text(
            0.0,
            0.85,
            "\n".join(lines),
            fontsize=9,
            family="monospace",
            va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Wrote %s", pdf_path)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.dingo_bns_demo import (
        discover_assets,
        load_event_dataset,
        select_device,
    )
    from stress_gw170817 import (
        _build_spec_stack,
        _detector_gates,
        _dl_ci,
        _gated_recovers,
        _load_detector,
        _oracle_gates,
        _poison_collapsed,
        _sample_dl,
        inject_spec_into_event,
    )
    from adapt.dingo_bns_demo import load_bns_checkpoint

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    logger.info("Device: %s outdir=%s", device, outdir)

    # ---- Oracle gap from existing real stress CSV (no re-PE) ----
    real_csv = Path(args.real_results_csv)
    if real_csv.is_file():
        oracle_real = aggregate_oracle_gap_real(real_csv)
        (outdir / "oracle_gap_real.json").write_text(json.dumps(oracle_real, indent=2))
        logger.info(
            "Real oracle gap: %d failures, oracle_only=%.2f",
            oracle_real["n_gated_failures"],
            (oracle_real["among_failures"] or {}).get("oracle_only_recovery_rate") or 0,
        )
    else:
        oracle_real = {"error": f"missing {real_csv}"}
        (outdir / "oracle_gap_real.json").write_text(json.dumps(oracle_real, indent=2))

    # ---- Setup GW170817 assets for ablation ----
    assets = discover_assets(baseline_ckpt=None, custom_ckpt=None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = dict(assets["fixed_context"])
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

    # Clean reference + threshold calibrate
    t_clean0 = time.time()
    clean_ci = _sample_dl(
        assets,
        event.data,
        settings,
        fixed,
        device=device,
        n=args.num_samples,
        bs=args.batch_size,
    )
    t_clean = time.time() - t_clean0
    logger.info("Clean d_L %s (%.2fs)", clean_ci, t_clean)

    # Calibrate thr on clean STFT crops from first inject of a dummy clean path:
    # use zero-gate threshold from checkpoint + small margin via a no-glitch check
    # on packaged event TD is expensive; use checkpoint threshold + 0.05 like stress.
    thr_event = float(threshold)
    # Better: load clean crops via evaluate path
    try:
        from adapt.dingo_bns_demo import load_event_td_crops

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
        if thr_event >= 0.99:
            thr_event = float(threshold)
    except Exception:
        logger.exception("Threshold calibrate failed; using checkpoint thr")

    cells = build_ablation_cells(master_seed=int(args.seed), max_cells=args.max_cells)
    logger.info("Ablation cells: %d", len(cells))

    abl_path = outdir / "ablation_results.csv"
    if args.overwrite and abl_path.is_file():
        abl_path.unlink()
    abl_fields = [
        "cell_id",
        "family",
        "severity",
        "held_out",
        "arm",
        "lo",
        "med",
        "hi",
        "recovers_like_clean",
        "poison_collapsed",
        "n_gates",
        "detector_fired",
        "t_inject_detect_s",
        "t_rebuild_s",
        "t_pe_s",
        "error",
    ]

    timing_rows: List[Dict[str, float]] = []
    abl_rows: List[Dict[str, Any]] = []

    for cell in cells:
        logger.info(
            "===== Ablation cell %d fam=%s sev=%.0f =====",
            cell["cell_id"],
            cell["family"],
            cell["severity"],
        )
        try:
            t0 = time.time()
            poison, td_stft, meta = inject_spec_into_event(event, assets, cell)
            # Detector
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
            t_inject_detect = time.time() - t0

            # Arm packages
            # 1) poison_welch = poison as-is
            # 2) glitch_orig_asd
            glitch_orig = copy.deepcopy(poison)
            for d, asd in asds_clean.items():
                if d in glitch_orig["asds"]:
                    glitch_orig["asds"][d] = np.asarray(asd, dtype=np.float64).copy()

            # 3) gate_welch — matched delta, keep Welch ASDs from poison
            t_rb0 = time.time()
            gated_welch = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=None,  # keep package ASDs (Welch)
                mode="matched_delta",
            )
            # 4) adapt_full
            adapt = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
                mode="matched_delta",
            )
            # 5) fft_replace
            replaced = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=gates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=asds_clean,
                mode="replace",
            )
            t_rebuild = time.time() - t_rb0

            arm_data = {
                "poison_welch": poison,
                "glitch_orig_asd": glitch_orig,
                "gate_welch": gated_welch.data,
                "adapt_full": adapt.data,
                "fft_replace": replaced.data,
            }

            cell_timing = {
                "t_inject_detect_s": t_inject_detect,
                "t_rebuild_s": t_rebuild,
            }
            for arm, data in arm_data.items():
                t_pe0 = time.time()
                err = ""
                try:
                    ci = _sample_dl(
                        assets,
                        data,
                        settings,
                        fixed,
                        device=device,
                        n=args.num_samples,
                        bs=args.batch_size,
                    )
                    recovers = _gated_recovers(ci, clean_ci)
                    collapsed = _poison_collapsed(ci)
                except Exception as exc:
                    ci = {
                        "lo": float("nan"),
                        "med": float("nan"),
                        "hi": float("nan"),
                        "n": 0,
                    }
                    recovers = False
                    collapsed = False
                    err = repr(exc) or type(exc).__name__
                    logger.exception("Arm %s cell %d failed", arm, cell["cell_id"])
                t_pe = time.time() - t_pe0
                row = {
                    "cell_id": cell["cell_id"],
                    "family": cell["family"],
                    "severity": cell["severity"],
                    "held_out": cell["held_out"],
                    "arm": arm,
                    "lo": ci["lo"],
                    "med": ci["med"],
                    "hi": ci["hi"],
                    "recovers_like_clean": recovers,
                    "poison_collapsed": collapsed,
                    "n_gates": len(gates),
                    "detector_fired": len(gates) > 0,
                    "t_inject_detect_s": round(t_inject_detect, 4),
                    "t_rebuild_s": round(t_rebuild, 4),
                    "t_pe_s": round(t_pe, 4),
                    "error": err,
                }
                append_csv_row(abl_path, row, abl_fields)
                abl_rows.append(row)
                if arm == "poison_welch":
                    cell_timing["t_poison_pe_s"] = t_pe
                if arm == "adapt_full":
                    cell_timing["t_adapt_pe_s"] = t_pe
                logger.info(
                    "  arm=%s med=%.2f recovers=%s (pe=%.2fs)",
                    arm,
                    ci["med"],
                    recovers,
                    t_pe,
                )
            timing_rows.append(cell_timing)
        except Exception as exc:
            logger.exception("Cell %d failed entirely", cell["cell_id"])
            for arm in ARMS:
                append_csv_row(
                    abl_path,
                    {
                        "cell_id": cell["cell_id"],
                        "family": cell["family"],
                        "severity": cell["severity"],
                        "arm": arm,
                        "error": repr(exc) or type(exc).__name__,
                    },
                    abl_fields,
                )

    ablation_df = pd.read_csv(abl_path) if abl_path.is_file() else pd.DataFrame()
    by_arm: Dict[str, Any] = {}
    if len(ablation_df):
        ok = ablation_df[ablation_df["error"].fillna("") == ""].copy()
        ok["recovers_like_clean"] = ok["recovers_like_clean"].astype(bool)
        ok["poison_collapsed"] = ok["poison_collapsed"].astype(bool)
        for arm, sub in ok.groupby("arm"):
            by_arm[str(arm)] = {
                "n": int(len(sub)),
                "recovers_like_clean": float(sub["recovers_like_clean"].mean()),
                "poison_collapsed": float(sub["poison_collapsed"].mean()),
                "median_med_d_L": float(sub["med"].median()),
            }
    ablation_summary = {
        "n_cells": len(cells),
        "clean_reference_d_L": clean_ci,
        "t_clean_pe_s": t_clean,
        "thr_event": thr_event,
        "by_arm": by_arm,
        "arms": list(ARMS),
    }
    (outdir / "ablation_summary.json").write_text(
        json.dumps(ablation_summary, indent=2, default=str)
    )

    # Runtime summary
    def _mean_std(vals: List[float]) -> Dict[str, float]:
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    stages = {
        "clean_pe_once": _mean_std([t_clean]),
        "inject_detect": _mean_std([r.get("t_inject_detect_s", np.nan) for r in timing_rows]),
        "rebuild_all_gated_arms": _mean_std(
            [r.get("t_rebuild_s", np.nan) for r in timing_rows]
        ),
        "poison_pe": _mean_std([r.get("t_poison_pe_s", np.nan) for r in timing_rows]),
        "adapt_pe": _mean_std([r.get("t_adapt_pe_s", np.nan) for r in timing_rows]),
    }
    # Overhead: inject+detect+rebuild + adapt_pe  vs poison_pe alone
    # Per cell: adapt_total = inject_detect + rebuild/3 (amortize 3 gated rebuilds roughly)
    # Simpler: overhead = (inject_detect + rebuild + adapt_pe) / poison_pe - 1
    ratios = []
    for r in timing_rows:
        pp = r.get("t_poison_pe_s")
        ap = r.get("t_adapt_pe_s")
        inj = r.get("t_inject_detect_s", 0)
        rb = r.get("t_rebuild_s", 0)
        if pp and pp > 0 and ap is not None:
            # rebuild timed all 3 gated rebuilds together; attribute 1/3 to adapt
            adapt_total = inj + (rb / 3.0) + ap
            ratios.append(adapt_total / pp)
    runtime = {
        "stages": stages,
        "adapt_total_over_poison_pe_mean": float(np.mean(ratios)) if ratios else None,
        "adapt_overhead_vs_poison_pe": float(np.mean(ratios) - 1.0) if ratios else None,
        "note": (
            "adapt_total ≈ inject_detect + rebuild/3 + adapt_pe; "
            "overhead = adapt_total/poison_pe - 1"
        ),
        "n_cells_timed": len(timing_rows),
    }
    (outdir / "runtime_summary.json").write_text(json.dumps(runtime, indent=2))

    # ---- Synthetic oracle gap re-runs ----
    oracle_synth = run_synthetic_oracle_gap(
        args=args,
        device=device,
        gate_half_s=gate_half_s,
        outdir=outdir,
    )

    write_report(
        outdir,
        ablation_df,
        ablation_summary,
        oracle_real,
        oracle_synth,
        runtime,
    )

    (outdir / "REPRODUCE.md").write_text(
        "\n".join(
            [
                "# Reproduce journal method hardening",
                "",
                "```bash",
                "conda activate adapt_env",
                f"cd {REPO_ROOT}",
                "export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE",
                "python -u examples/method_hardening.py \\",
                f"  --num-samples {args.num_samples} --batch-size {args.batch_size} --device cpu \\",
                f"  --outdir {outdir}",
                "```",
                "",
                "Outputs: ablation_results.csv, ablation_summary.json,",
                "oracle_gap_real.json, oracle_gap_synthetic.csv/json,",
                "runtime_summary.json, method_hardening_report.pdf",
                "",
            ]
        )
    )
    logger.info("Done → %s", outdir)
    logger.info("Ablation by arm: %s", json.dumps(by_arm, indent=2))
    logger.info("Runtime overhead: %s", runtime.get("adapt_overhead_vs_poison_pe"))


def run_synthetic_oracle_gap(
    *,
    args: argparse.Namespace,
    device: torch.device,
    gate_half_s: float,
    outdir: Path,
) -> Dict[str, Any]:
    fail_path = Path(args.synth_failures_csv)
    if not fail_path.is_file():
        out = {"error": f"missing {fail_path}", "n_failures": 0}
        (outdir / "oracle_gap_synthetic.json").write_text(json.dumps(out, indent=2))
        return out

    from adapt.glitch_excision import rebuild_event_from_gated_td
    from dingo.gw.domains import build_domain_from_model_metadata
    from adapt.dingo_bns_demo import (
        discover_assets,
        load_event_dataset,
    )
    from stress_gw170817 import (
        _build_spec_stack,
        _detector_gates,
        _load_detector,
        _oracle_gates,
        _sample_dl,
    )
    from stress_test_synthetic_bns import (
        build_synthetic_event,
        draw_synthetic_theta,
        inject_glitch_on_synthetic,
        _gated_recovers as _gated_recovers_synth,
    )
    from adapt.dingo_bns_demo import build_base_domain_injection, load_bns_checkpoint

    fails = pd.read_csv(fail_path)
    assets = discover_assets(baseline_ckpt=None, custom_ckpt=None)
    ref_event = load_event_dataset(assets)
    settings = dict(ref_event.settings)
    demo_fixed = dict(assets["fixed_context"])
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s", 4096.0))
    delta_f = float(base_domain.delta_f)
    noise_std = float(base_domain.noise_std)
    asds_ref = {d: np.asarray(ref_event.data["asds"][d]).copy() for d in detectors}
    n_base = len(base_domain)
    for d in detectors:
        a = asds_ref[d]
        if len(a) < n_base:
            asds_ref[d] = np.pad(a, (0, n_base - len(a)), constant_values=1.0)
        elif len(a) > n_base:
            asds_ref[d] = a[:n_base]
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    injection, _ = build_base_domain_injection(metadata)

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    model, det_raw = _load_detector(det_path, device)
    threshold = float(det_raw.get("threshold", 0.5))
    norm_stats = det_raw.get("norm_stats")
    stft_kwargs = dict(det_raw.get("stft_kwargs") or {})

    # Load thr_event per event from events.csv if present
    thr_by_event: Dict[int, float] = {}
    ev_path = Path(args.synth_events_csv)
    if ev_path.is_file():
        ev = pd.read_csv(ev_path)
        for _, r in ev.iterrows():
            thr_by_event[int(r["event_id"])] = float(r.get("thr_event", threshold))

    csv_path = outdir / "oracle_gap_synthetic.csv"
    if args.overwrite and csv_path.is_file():
        csv_path.unlink()
    fields = [
        "event_id",
        "cell_id",
        "family",
        "severity",
        "detector_fired",
        "detector_n_gates",
        "gated_med",
        "gated_recovers",
        "oracle_med",
        "oracle_recovers",
        "classification",
        "error",
        "elapsed_s",
    ]

    # Cache rebuilt events
    event_cache: Dict[int, Any] = {}
    rows_out: List[Dict[str, Any]] = []

    for _, fr in fails.iterrows():
        t0 = time.time()
        event_id = int(fr["event_id"])
        cell = {
            "event_id": event_id,
            "cell_id": int(fr["cell_id"]),
            "family": str(fr["family"]),
            "held_out": bool(fr.get("held_out", False)),
            "severity": float(fr["severity"]),
            "asd_policy": str(fr.get("asd_policy", "welch")),
            "seed": int(fr["seed"]),
            "detectors": str(fr.get("detectors", "H1")).split(","),
            "t_rel": float(fr["t_rel"]),
            "params": json.loads(fr["params_json"])
            if isinstance(fr["params_json"], str)
            else dict(fr["params_json"]),
        }
        row: Dict[str, Any] = {
            "event_id": event_id,
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "severity": cell["severity"],
            "gated_med": float(fr.get("gated_med", float("nan"))),
            "gated_recovers": False,
            "error": "",
        }
        try:
            if event_id not in event_cache:
                ev_seed = int(args.seed) + 1_000_003 * int(event_id)
                rng = np.random.default_rng(ev_seed)
                theta = draw_synthetic_theta(injection, rng, fixed_sky=demo_fixed)
                fixed = {
                    "chirp_mass_proxy": float(theta["chirp_mass_proxy"]),
                    "ra": float(demo_fixed["ra"]),
                    "dec": float(demo_fixed["dec"]),
                }
                syn_event, td_clean, truth = build_synthetic_event(
                    injection,
                    theta,
                    asds_ref=asds_ref,
                    settings=settings,
                    detectors=detectors,
                )
                # Clean CI from events.csv if available
                clean_med = float(fr.get("clean_med", float("nan")))
                if ev_path.is_file():
                    ev = pd.read_csv(ev_path)
                    er = ev[ev["event_id"] == event_id]
                    if len(er):
                        clean_ci = {
                            "lo": float(er.iloc[0]["clean_lo"]),
                            "med": float(er.iloc[0]["clean_med"]),
                            "hi": float(er.iloc[0]["clean_hi"]),
                            "n": int(args.num_samples),
                        }
                    else:
                        clean_ci = {
                            "lo": clean_med - 10,
                            "med": clean_med,
                            "hi": clean_med + 10,
                            "n": 0,
                        }
                else:
                    clean_ci = {
                        "lo": clean_med - 10,
                        "med": clean_med,
                        "hi": clean_med + 10,
                        "n": 0,
                    }
                event_cache[event_id] = {
                    "event": syn_event,
                    "td_clean": td_clean,
                    "fixed": fixed,
                    "clean_ci": clean_ci,
                    "asds_clean": {
                        d: np.asarray(syn_event.data["asds"][d]).copy() for d in detectors
                    },
                }

            pack = event_cache[event_id]
            thr_event = float(thr_by_event.get(event_id, threshold))
            if thr_event >= 0.99:
                thr_event = float(threshold)

            poison, td_stft, meta = inject_glitch_on_synthetic(
                pack["event"].data, pack["td_clean"], cell, settings
            )
            spec_g, _ = _build_spec_stack(
                meta["td_stft"],
                asds_clean=pack["asds_clean"],
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
                original_asds=pack["asds_clean"],
            )
            gated_ci = _sample_dl(
                assets,
                gated.data,
                settings,
                pack["fixed"],
                device=device,
                n=args.num_samples,
                bs=args.batch_size,
            )
            ogates = _oracle_gates(meta, cell, gate_half_s)
            oracle = rebuild_event_from_gated_td(
                poison,
                td_by_det=meta["td_full"],
                gates=ogates,
                sample_rate=sample_rate,
                roll_off=roll_off,
                f_max=f_max,
                original_asds=pack["asds_clean"],
            )
            oracle_ci = _sample_dl(
                assets,
                oracle.data,
                settings,
                pack["fixed"],
                device=device,
                n=args.num_samples,
                bs=args.batch_size,
            )
            g_ok = _gated_recovers_synth(gated_ci, pack["clean_ci"])
            o_ok = _gated_recovers_synth(oracle_ci, pack["clean_ci"])
            if o_ok and not g_ok:
                classification = "detector_gap"
            elif (not o_ok) and (not g_ok):
                classification = "gate_insufficient"
            elif g_ok:
                classification = "detector_recovers_on_rerun"
            else:
                classification = "other"
            row.update(
                {
                    "detector_fired": len(gates) > 0,
                    "detector_n_gates": len(gates),
                    "gated_med": gated_ci["med"],
                    "gated_recovers": g_ok,
                    "oracle_med": oracle_ci["med"],
                    "oracle_recovers": o_ok,
                    "classification": classification,
                    "elapsed_s": round(time.time() - t0, 3),
                }
            )
            logger.info(
                "Synth fail e%d c%d → %s (gated=%s oracle=%s)",
                event_id,
                cell["cell_id"],
                classification,
                g_ok,
                o_ok,
            )
        except Exception as exc:
            row["error"] = repr(exc) or type(exc).__name__
            row["classification"] = "error"
            row["elapsed_s"] = round(time.time() - t0, 3)
            logger.exception("Synth oracle cell failed")
        append_csv_row(csv_path, row, fields)
        rows_out.append(row)

    sdf = pd.DataFrame(rows_out)
    counts = (
        sdf["classification"].value_counts().astype(int).to_dict() if len(sdf) else {}
    )
    summary = {
        "n_failures": int(len(sdf)),
        "classification_counts": counts,
        "oracle_recovers_rate": float(sdf["oracle_recovers"].astype(bool).mean())
        if len(sdf) and "oracle_recovers" in sdf.columns
        else None,
        "detector_gap_rate": float((sdf["classification"] == "detector_gap").mean())
        if len(sdf)
        else None,
        "gate_insufficient_rate": float(
            (sdf["classification"] == "gate_insufficient").mean()
        )
        if len(sdf)
        else None,
    }
    (outdir / "oracle_gap_synthetic.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--detector-ckpt", type=str, default=None)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--real-results-csv", type=str, default=str(DEFAULT_REAL_RESULTS))
    p.add_argument(
        "--synth-failures-csv", type=str, default=str(DEFAULT_SYNTH_FAILURES)
    )
    p.add_argument("--synth-events-csv", type=str, default=str(DEFAULT_SYNTH_EVENTS))
    p.add_argument(
        "--skip-synth-oracle",
        action="store_true",
        help="Only aggregate real oracle gap + run ablation",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.skip_synth_oracle:
        # monkeypatch via empty failures path
        args.synth_failures_csv = "/nonexistent_skip.csv"
    run(args)
