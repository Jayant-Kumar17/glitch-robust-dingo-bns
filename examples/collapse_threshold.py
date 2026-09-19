#!/usr/bin/env python3
"""Collapse-threshold sweep on GW170817 (frozen DINGO-BNS).

Synthetic ``sine_gaussian`` and ``broadband_burst`` glitches are injected into
H1 at 8 log-spaced whitened optimal SNRs rho_w in [10, 1e4] (5 seeds each,
stationary ASD) and run through two arms: poisoned and detector-gated
``adapt_full``. Reports the collapse fraction and gated recovery versus rho_w
and the rho_w at which collapse first exceeds 50 %.

Usage::

    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/collapse_threshold.py --outdir results/collapse_threshold_v1
    python examples/collapse_threshold.py --smoke
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
for _p in (REPO_ROOT / "examples", REPO_ROOT / "src", REPO_ROOT / "DINGO-BNS" / "dingo", REPO_ROOT):
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
    _sample_family_params,
    _sha256,
    append_csv_row,
)

logger = logging.getLogger("collapse_threshold")
DEFAULT_OUTDIR = REPO_ROOT / "results" / "collapse_threshold_v1"
DEFAULT_FIGDIR = REPO_ROOT / "paper" / "figures"
FAMILIES = ("sine_gaussian", "broadband_burst")
RHO_GRID = np.logspace(1.0, 4.0, 8)
T_REL_RANGE = (-1.5, -0.3)

FIELDNAMES = [
    "cell_id", "family", "rho_w_target", "rho_w", "seed", "t_rel", "params_json", "amp_scale",
    "poison_lo", "poison_med", "poison_hi", "poison_collapsed",
    "gated_lo", "gated_med", "gated_hi", "gated_recovers",
    "detector_n_gates", "detector_fired", "gates_json", "error", "elapsed_s",
]


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.dingo_bns_demo import discover_assets, load_bns_checkpoint, load_event_dataset, load_event_td_crops, select_device
    from adapt.event_glitch_io import load_full_event_td, td_to_fd_strain
    from adapt.glitch_augmentation import GlitchSpec, synthesize_glitch_td
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.loudness import scale_to_rho
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS
    from dingo.gw.domains import build_domain_from_model_metadata

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    n_samples = 64 if args.smoke else int(args.num_samples)
    n_seeds = 1 if args.smoke else int(args.n_seeds)
    rho_grid = np.asarray(RHO_GRID if not args.smoke else RHO_GRID[[0, 7]], dtype=float)

    assets = discover_assets(baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = assets["fixed_context"]
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"])
    delta_f = float(base_domain.delta_f); noise_std = float(base_domain.noise_std)
    asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    f_min = float(settings.get("f_min", 23.0)); f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    duration = float(settings.get("T", 128.0)); time_buffer = float(settings.get("time_buffer", 2.0))

    td_clean = {}
    for det in detectors:
        td, _, _ = load_full_event_td(assets, settings, det)
        td_clean[det] = td
    n_freq = len(np.asarray(event.data["waveform"]["H1"]))

    det_path = Path(args.detector_ckpt) if args.detector_ckpt else DEFAULT_DETECTOR
    model, det_raw = _load_detector(det_path, device)
    threshold = float(det_raw.get("threshold", 0.5))
    gate_half_s = float(det_raw.get("gate_half_s", args.gate_half_s))
    norm_stats = det_raw.get("norm_stats"); stft_kwargs = dict(det_raw.get("stft_kwargs") or {})

    clean_ci = _sample_dl(assets, event.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
    (outdir / "clean_reference.json").write_text(json.dumps(clean_ci, indent=2))
    clean_crops = load_event_td_crops(assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS)
    trig_idx = int(round((duration - time_buffer) * sample_rate))
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    crop_start = max(0, trig_idx - n_crop // 2)
    spec_c, _ = _build_spec_stack(clean_crops, asds_clean=asds_clean, detectors=detectors, sample_rate=sample_rate,
                                  delta_f=delta_f, noise_std=noise_std, norm_stats=norm_stats, stft_kwargs=stft_kwargs)
    _, clean_probs = _detector_gates(model, spec_c, detectors=detectors, crop_start=crop_start, sample_rate=sample_rate,
                                     threshold=threshold, gate_half_s=gate_half_s, device=device)
    thr_event = float(args.threshold_event) if args.threshold_event is not None else float(max(threshold, float(np.max(clean_probs)) + 0.05))
    logger.info("clean d_L %s thr_event=%.3f", clean_ci, thr_event)

    cfg = {
        "outdir": str(outdir), "seed": int(args.seed), "num_samples": n_samples, "n_seeds": n_seeds,
        "families": list(FAMILIES), "rho_w_grid": [float(x) for x in rho_grid], "asd_policy": "stationary",
        "arms": ["poisoned", "adapt_full"], "t_rel_range": list(T_REL_RANGE), "gate_half_s": gate_half_s,
        "threshold_event": thr_event, "detector_ckpt": str(det_path), "detector_sha256": _sha256(det_path),
        "baseline_ckpt": str(assets["baseline_ckpt"]), "baseline_sha256": _sha256(Path(assets["baseline_ckpt"])),
        "git_commit": _git_commit(), "python": sys.version, "platform": platform.platform(),
        "rho_w_definition": "sqrt(4 sum |g(f)|^2/ASD_H1(f)^2 df) over [f_min,f_max] on the packaged grid",
        "success_criteria": {"poison_collapsed": "d_L hi<15 or lo>90",
                             "gated_recovers": "med in [20,50], CI overlaps clean, |med-clean_med|<=10"},
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    cells: List[Dict[str, Any]] = []
    cid = 0
    for fam in FAMILIES:
        for k in range(n_seeds):
            seed = int(args.seed) + 1000 * k + (0 if fam == "sine_gaussian" else 500)
            rng = np.random.default_rng(seed)
            t_rel = float(rng.uniform(*T_REL_RANGE))
            params = _sample_family_params(fam, rng)
            for rho in rho_grid:
                cells.append({"cell_id": cid, "family": fam, "rho_w_target": float(rho), "seed": seed,
                              "t_rel": t_rel, "params": params})
                cid += 1
    if args.max_cells:
        cells = cells[: int(args.max_cells)]
    logger.info("planned cells: %d", len(cells))

    csv_path = outdir / "results.csv"
    if csv_path.is_file():
        csv_path.unlink()
    t_all = time.time()
    t_trig = duration - time_buffer
    for i, cell in enumerate(cells):
        t0 = time.time()
        row: Dict[str, Any] = {"cell_id": cell["cell_id"], "family": cell["family"], "rho_w_target": cell["rho_w_target"],
                               "seed": cell["seed"], "t_rel": cell["t_rel"], "params_json": json.dumps(cell["params"]), "error": ""}
        try:
            t_peak = t_trig + cell["t_rel"]
            spec = GlitchSpec(family=cell["family"], detectors=["H1"], t_rel=t_peak - 0.5 * duration, severity=1.0,
                              params=cell["params"], asd_policy="stationary", held_out=False)
            # Same waveform realisation for every rho of a given seed (fixed RNG), unit amplitude, then scaled.
            g1 = synthesize_glitch_td(len(td_clean["H1"]), sample_rate, spec, np.random.default_rng(cell["seed"] + 7), rms=1.0)
            g, k, _ = scale_to_rho(g1, cell["rho_w_target"], sample_rate, asd=asds_clean["H1"], delta_f=delta_f,
                                   f_min=f_min, f_max=f_max, roll_off=roll_off)
            row["amp_scale"] = k
            row["rho_w"] = cell["rho_w_target"]
            td_full = {d: td_clean[d].copy() for d in detectors}
            td_full["H1"] = td_clean["H1"] + g
            poison = copy.deepcopy(event.data)
            g_fd = td_to_fd_strain(g, sample_rate, roll_off=roll_off, f_max=f_max)
            g_fd = np.pad(g_fd, (0, max(0, n_freq - len(g_fd))))[:n_freq]
            poison["waveform"]["H1"] = np.asarray(poison["waveform"]["H1"], dtype=np.complex128) + g_fd
            td_stft = {d: td_full[d][crop_start: crop_start + n_crop].copy() for d in detectors}

            poison_ci = _sample_dl(assets, poison, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            row.update({"poison_lo": poison_ci["lo"], "poison_med": poison_ci["med"], "poison_hi": poison_ci["hi"],
                        "poison_collapsed": _poison_collapsed(poison_ci)})
            spec_g, _ = _build_spec_stack(td_stft, asds_clean=asds_clean, detectors=detectors, sample_rate=sample_rate,
                                          delta_f=delta_f, noise_std=noise_std, norm_stats=norm_stats, stft_kwargs=stft_kwargs)
            gates, _ = _detector_gates(model, spec_g, detectors=detectors, crop_start=crop_start, sample_rate=sample_rate,
                                       threshold=thr_event, gate_half_s=gate_half_s, device=device, ifo_whitelist=["H1"])
            row["detector_n_gates"] = len(gates); row["detector_fired"] = bool(gates)
            row["gates_json"] = json.dumps([[round(gg.t_start - t_trig, 3), round(gg.t_end - t_trig, 3)] for gg in gates])
            gated = rebuild_event_from_gated_td(poison, td_by_det=td_full, gates=gates, sample_rate=sample_rate,
                                                roll_off=roll_off, f_max=f_max, original_asds=asds_clean)
            gated_ci = _sample_dl(assets, gated.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            row.update({"gated_lo": gated_ci["lo"], "gated_med": gated_ci["med"], "gated_hi": gated_ci["hi"],
                        "gated_recovers": _gated_recovers(gated_ci, clean_ci)})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("cell %d failed", cell["cell_id"]); traceback.print_exc()
        row["elapsed_s"] = round(time.time() - t0, 3)
        append_csv_row(csv_path, row, FIELDNAMES)
        logger.info("[%d/%d] %s rho_w=%.0f poison_col=%s gated_ok=%s fired=%s (%.1fs)", i + 1, len(cells), cell["family"],
                    cell["rho_w_target"], row.get("poison_collapsed"), row.get("gated_recovers"), row.get("detector_fired"), row["elapsed_s"])
    logger.info("done in %.1f min", (time.time() - t_all) / 60.0)
    write_summary_and_figure(outdir, pd.read_csv(csv_path), clean_ci, cfg, figdir=Path(args.figdir))


def _first_crossing(x: np.ndarray, y: np.ndarray, level: float = 0.5) -> Optional[float]:
    """Log-linear interpolated first rho_w where the fraction crosses ``level``."""
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            lx0, lx1 = np.log10(x[i - 1]), np.log10(x[i])
            f = (level - y[i - 1]) / max(y[i] - y[i - 1], 1e-12)
            return float(10 ** (lx0 + f * (lx1 - lx0)))
    if len(y) and y[0] >= level:
        return float(x[0])
    return None


def write_summary_and_figure(outdir: Path, df: pd.DataFrame, clean_ci, cfg, *, figdir: Path) -> None:
    ok = df[df["error"].fillna("") == ""].copy()
    for c in ("poison_collapsed", "gated_recovers", "detector_fired"):
        ok[c] = ok[c].astype(bool)
    curves: Dict[str, Any] = {}
    thresholds: Dict[str, Any] = {}
    groups = {"all": ok} | {f: ok[ok["family"] == f] for f in FAMILIES}
    for name, sub in groups.items():
        if not len(sub):
            continue
        g = sub.groupby("rho_w_target").agg(n=("poison_collapsed", "size"), collapse=("poison_collapsed", "mean"),
                                            gated_recovery=("gated_recovers", "mean"), detector_fire=("detector_fired", "mean")).reset_index()
        curves[name] = g.to_dict(orient="records")
        x = g["rho_w_target"].to_numpy(); y = g["collapse"].to_numpy()
        # last rho with collapse<0.5 and first rho with collapse>=0.5 (grid points) + interpolated crossing
        above = np.where(y >= 0.5)[0]
        thresholds[name] = {
            "first_grid_rho_w_with_collapse_ge_50pct": float(x[above[0]]) if len(above) else None,
            "last_grid_rho_w_with_collapse_lt_50pct": float(x[above[0] - 1]) if len(above) and above[0] > 0 else None,
            "interpolated_rho_w_at_50pct": _first_crossing(x, y),
            "collapse_at_max_rho": float(y[-1]),
        }
    summary = {
        "n_rows": int(len(df)), "n_ok": int(len(ok)), "n_errors": int((df["error"].fillna("") != "").sum()),
        "clean_reference_d_L": clean_ci, "rho_w_grid": cfg["rho_w_grid"], "n_seeds": cfg["n_seeds"],
        "curves": curves, "collapse_threshold": thresholds,
        "gated_recovery_overall": float(ok["gated_recovers"].mean()) if len(ok) else None,
        "gated_recovery_where_collapsed": float(ok.loc[ok["poison_collapsed"], "gated_recovers"].mean()) if ok["poison_collapsed"].any() else None,
        "config_ref": "config.json",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        styles = {"sine_gaussian": ("o", "-"), "broadband_burst": ("s", "--")}
        for fam in FAMILIES:
            if fam not in curves:
                continue
            g = pd.DataFrame(curves[fam])
            mk, ls = styles[fam]
            ax.plot(g["rho_w_target"], g["collapse"], marker=mk, ls=ls, ms=3.5, color="#c0392b", lw=1.1,
                    label=f"{fam.replace('_', ' ')}: collapsed")
            ax.plot(g["rho_w_target"], g["gated_recovery"], marker=mk, ls=ls, ms=3.5, color="#2a9d8f", lw=1.1,
                    label=f"{fam.replace('_', ' ')}: gated recovered")
        thr = thresholds.get("all", {}).get("interpolated_rho_w_at_50pct")
        if thr:
            ax.axvline(thr, color="#666", ls=":", lw=0.9)
            ax.text(thr * 1.08, 0.05, rf"$\rho_w^{{50\%}}\approx{thr:.0f}$", fontsize=6.5, color="#666")
        ax.axhline(0.5, color="#bbb", lw=0.6)
        ax.set_xscale("log"); ax.set_xlabel(r"injected whitened optimal SNR $\rho_w$")
        ax.set_ylabel("fraction of cells"); ax.set_ylim(-0.03, 1.05)
        ax.legend(frameon=False, fontsize=5.8, loc="center left")
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / "fig9_threshold.pdf"); fig.savefig(figdir / "fig9_threshold.png", dpi=300)
        fig.savefig(outdir / "fig9_threshold.pdf"); plt.close(fig)
    except Exception as e:
        logger.warning("figure failed: %s", e)

    (outdir / "REPRODUCE.md").write_text(f"""# Reproduce the collapse-threshold sweep

```bash
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/collapse_threshold.py --seed {cfg['seed']} --num-samples {cfg['num_samples']} --n-seeds {cfg['n_seeds']} --outdir {outdir}
```

Families: {', '.join(cfg['families'])}; rho_w grid: {', '.join(f'{x:.0f}' for x in cfg['rho_w_grid'])};
stationary ASD; arms poisoned + adapt_full (detector threshold {cfg['threshold_event']:.3f}).
Each seed fixes t_rel and the waveform realisation; only the amplitude is rescaled to hit rho_w.
""")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--detector-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--threshold-event", type=float, default=None)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="2 rho values x 1 seed x 2 families, 64 samples")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run(a)
