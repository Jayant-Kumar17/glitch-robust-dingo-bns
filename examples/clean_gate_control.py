#!/usr/bin/env python3
"""Clean-gate cost control: is a false-positive gate harmless?

Applies the full ``adapt_full`` recipe (Tukey gate -> matched-delta rebuild ->
original ASD) to *clean*, glitch-free GW170817 data at N forced gate
placements and samples the frozen DINGO-BNS posterior for each. This is the
counterpart of ``tier_c_clean_fp`` (detector stays silent on clean data) and
``clean_noop_bit_exact`` (the no-fire path is a bit-exact no-op): it measures
what a spurious gate would cost if the detector *did* fire on clean data.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/clean_gate_control.py --outdir results/clean_gate_control_v1
    python examples/clean_gate_control.py --smoke   # 2 placements / 64 samples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from _repo import REPO_ROOT, resolve_under_repo

from official_control import CORE_PARAMS, js_divergence_1d  # noqa: E402
from stress_gw170817 import _dl_ci, _gated_recovers, _poison_collapsed, append_csv_row  # noqa: E402

DEFAULT_OUTDIR = REPO_ROOT / "results" / "clean_gate_control_v1"
DEFAULT_FIGDIR = REPO_ROOT / "paper" / "figures"
REPORT_PARAMS = ("luminosity_distance", "chirp_mass", "mass_ratio", "theta_jn")
T_REL_RANGE = (-2.0, -0.2)

logger = logging.getLogger("clean_gate_control")


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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sample(assets, event_data, settings, fixed, *, device, n, bs) -> pd.DataFrame:
    from adapt.dingo_bns_demo import run_baseline_sampling

    return run_baseline_sampling(
        assets["baseline_ckpt"], SimpleNamespace(data=event_data, settings=settings), fixed,
        device=device, num_samples=int(n), batch_size=int(bs),
    )


def _inband_power(fd: np.ndarray, delta_f: float, f_min: float, f_max: float) -> float:
    f = np.arange(len(fd)) * float(delta_f)
    m = (f >= float(f_min)) & (f <= float(f_max))
    return float(np.sum(np.abs(np.asarray(fd)[m]) ** 2))


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.dingo_bns_demo import discover_assets, load_bns_checkpoint, load_event_dataset, select_device
    from adapt.event_glitch_io import load_full_event_td
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td
    from dingo.gw.domains import build_domain_from_model_metadata

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    rng = np.random.default_rng(int(args.seed))
    n_samples = 64 if args.smoke else int(args.num_samples)
    n_h1 = 2 if args.smoke else int(args.n_placements)
    n_both = 1 if args.smoke else int(args.n_both)

    assets = discover_assets(baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None)
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fixed = assets["fixed_context"]
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(settings.get("f_s") or metadata["train_settings"]["data"]["window"]["f_s"])
    delta_f = float(base_domain.delta_f)
    asds_clean = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    roll_off = float(settings.get("roll_off", 0.4))
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    gate_half_s = float(args.gate_half_s)

    td_full: Dict[str, np.ndarray] = {}
    for det in ("H1", "L1"):
        td, _, _ = load_full_event_td(assets, settings, det)
        td_full[det] = td

    # ---- clean reference ----
    logger.info("Clean reference PE N=%d", n_samples)
    clean_df = _sample(assets, event.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
    clean_ci = {p: _dl_ci(clean_df[p].to_numpy()) for p in REPORT_PARAMS if p in clean_df.columns}
    (outdir / "clean_reference.json").write_text(json.dumps(clean_ci, indent=2))
    logger.info("Clean d_L %s", clean_ci["luminosity_distance"])
    # Finite-sample JS null: a second independent clean draw of the same size.
    torch.manual_seed(int(args.seed) + 12345)
    clean_df2 = _sample(assets, event.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
    js_null = {p: float(js_divergence_1d(clean_df[p].to_numpy(), clean_df2[p].to_numpy()))
               for p in CORE_PARAMS if p in clean_df.columns}
    torch.manual_seed(int(args.seed))
    logger.info("JS null (clean vs clean, n=%d): %s", n_samples, {k: round(v, 4) for k, v in js_null.items()})

    cfg = {
        "outdir": str(outdir),
        "seed": int(args.seed),
        "num_samples": n_samples,
        "batch_size": int(args.batch_size),
        "gate_half_s": gate_half_s,
        "t_rel_range": list(T_REL_RANGE),
        "n_placements_h1": n_h1,
        "n_placements_h1_l1": n_both,
        "recipe": "Tukey gate (alpha 0.5) -> matched-delta FD rebuild -> original analysis ASD (adapt_full); no glitch injected",
        "baseline_ckpt": str(assets["baseline_ckpt"]),
        "baseline_sha256": _sha256(Path(assets["baseline_ckpt"])),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "criteria": {
            "gated_recovers": "stress_gw170817._gated_recovers: d_L med in [20,50], CI overlaps clean, |med-clean_med|<=10",
            "power_removed_frac": "sum|dX|^2 / sum|X|^2 over [f_min,f_max] on the gated IFO packaged FD strain",
        },
        "js_null_clean_vs_clean_nat": js_null,
        "js_null_max_nat": float(max(js_null.values())) if js_null else None,
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    placements: List[Dict[str, Any]] = []
    for i in range(n_h1):
        placements.append({"cell_id": i, "detectors": ["H1"], "t_rel": float(rng.uniform(*T_REL_RANGE))})
    for j in range(n_both):
        placements.append({"cell_id": n_h1 + j, "detectors": ["H1", "L1"], "t_rel": float(rng.uniform(*T_REL_RANGE))})

    fieldnames = ["cell_id", "detectors", "t_rel", "gate_t_start_rel", "gate_t_end_rel"]
    for p in REPORT_PARAMS:
        fieldnames += [f"{p}_lo", f"{p}_med", f"{p}_hi"]
    fieldnames += ["gated_recovers", "poison_style_collapse", "delta_dL_med"]
    fieldnames += [f"js_{p}" for p in CORE_PARAMS] + ["js_max"]
    fieldnames += ["power_removed_frac_H1", "power_removed_frac_L1", "modified_detectors", "error", "elapsed_s"]
    csv_path = outdir / "results.csv"
    if csv_path.is_file():
        csv_path.unlink()

    t_trig = duration - time_buffer
    for k, pl in enumerate(placements):
        t0 = time.time()
        t_peak = t_trig + pl["t_rel"]
        gates = [GateWindow(detector=d, t_start=t_peak - gate_half_s, t_end=t_peak + gate_half_s, score=1.0)
                 for d in pl["detectors"]]
        row: Dict[str, Any] = {
            "cell_id": pl["cell_id"], "detectors": ",".join(pl["detectors"]), "t_rel": pl["t_rel"],
            "gate_t_start_rel": pl["t_rel"] - gate_half_s, "gate_t_end_rel": pl["t_rel"] + gate_half_s, "error": "",
        }
        try:
            gated = rebuild_event_from_gated_td(
                event.data, td_by_det={d: td_full[d] for d in pl["detectors"]}, gates=gates,
                sample_rate=sample_rate, roll_off=roll_off, f_max=f_max, original_asds=asds_clean, mode="matched_delta",
            )
            row["modified_detectors"] = ",".join(gated.modified_detectors)
            for d in ("H1", "L1"):
                if d in pl["detectors"]:
                    X = np.asarray(event.data["waveform"][d], dtype=np.complex128)
                    dX = np.asarray(gated.data["waveform"][d], dtype=np.complex128) - X
                    row[f"power_removed_frac_{d}"] = _inband_power(dX, delta_f, f_min, f_max) / max(_inband_power(X, delta_f, f_min, f_max), 1e-300)
                else:
                    row[f"power_removed_frac_{d}"] = 0.0
            df = _sample(assets, gated.data, settings, fixed, device=device, n=n_samples, bs=args.batch_size)
            for p in REPORT_PARAMS:
                ci = _dl_ci(df[p].to_numpy())
                row[f"{p}_lo"], row[f"{p}_med"], row[f"{p}_hi"] = ci["lo"], ci["med"], ci["hi"]
            dl_ci = _dl_ci(df["luminosity_distance"].to_numpy())
            row["gated_recovers"] = _gated_recovers(dl_ci, clean_ci["luminosity_distance"])
            row["poison_style_collapse"] = _poison_collapsed(dl_ci)
            row["delta_dL_med"] = float(dl_ci["med"] - clean_ci["luminosity_distance"]["med"])
            js_vals = []
            for p in CORE_PARAMS:
                if p in df.columns and p in clean_df.columns:
                    v = js_divergence_1d(clean_df[p].to_numpy(), df[p].to_numpy())
                    row[f"js_{p}"] = v
                    js_vals.append(v)
            row["js_max"] = float(np.nanmax(js_vals)) if js_vals else float("nan")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("placement %d failed", pl["cell_id"])
        row["elapsed_s"] = round(time.time() - t0, 3)
        append_csv_row(csv_path, row, fieldnames)
        logger.info("[%d/%d] dets=%s t_rel=%.2f d_L med=%.1f (clean %.1f) recovers=%s js_max=%.4f removed=%.3f (%.1fs)",
                    k + 1, len(placements), row["detectors"], pl["t_rel"], row.get("luminosity_distance_med", float("nan")),
                    clean_ci["luminosity_distance"]["med"], row.get("gated_recovers"), row.get("js_max", float("nan")),
                    row.get("power_removed_frac_H1", float("nan")), row["elapsed_s"])

    df = pd.read_csv(csv_path)
    write_summary_and_figure(outdir, df, clean_ci, cfg, figdir=Path(args.figdir))
    logger.info("Clean-gate control complete -> %s", outdir)


def write_summary_and_figure(outdir: Path, df: pd.DataFrame, clean_ci, cfg, *, figdir: Path) -> None:
    ok = df[df["error"].fillna("") == ""].copy()
    ok["gated_recovers"] = ok["gated_recovers"].astype(bool)
    h1 = ok[ok["detectors"] == "H1"]
    both = ok[ok["detectors"] == "H1,L1"]

    def _blk(sub):
        return {
            "n": int(len(sub)),
            "recovery_rate": float(sub["gated_recovers"].mean()) if len(sub) else None,
            "max_js_nat": float(sub["js_max"].max()) if len(sub) else None,
            "mean_abs_delta_dL_med_mpc": float(sub["delta_dL_med"].abs().mean()) if len(sub) else None,
            "max_abs_delta_dL_med_mpc": float(sub["delta_dL_med"].abs().max()) if len(sub) else None,
            "mean_power_removed_frac_H1": float(sub["power_removed_frac_H1"].mean()) if len(sub) else None,
            "collapse_rate": float(sub["poison_style_collapse"].astype(bool).mean()) if len(sub) else None,
        }

    summary = {
        "n_placements": int(len(df)),
        "n_ok": int(len(ok)),
        "n_errors": int((df["error"].fillna("") != "").sum()),
        "clean_reference": clean_ci,
        "recovery_rate": _blk(ok)["recovery_rate"],
        "max_js_nat": _blk(ok)["max_js_nat"],
        "mean_abs_delta_dL_med_mpc": _blk(ok)["mean_abs_delta_dL_med_mpc"],
        "mean_power_removed_frac": _blk(ok)["mean_power_removed_frac_H1"],
        "by_detectors": {"H1": _blk(h1), "H1+L1": _blk(both)},
        "js_by_param_max": {p: float(ok[f"js_{p}"].max()) for p in CORE_PARAMS if f"js_{p}" in ok.columns and len(ok)},
        "js_null_clean_vs_clean_nat": cfg.get("js_null_clean_vs_clean_nat"),
        "js_null_max_nat": cfg.get("js_null_max_nat"),
        "js_excess_over_null_max_nat": (
            float(_blk(ok)["max_js_nat"] - cfg["js_null_max_nat"])
            if len(ok) and cfg.get("js_null_max_nat") is not None else None
        ),
        "note": (
            "Forced Tukey gates (half-width %.2f s) on clean GW170817 data with the full adapt_full recipe; "
            "measures the cost of a false-positive gate. Recovery criterion identical to the stress grids."
            % cfg["gate_half_s"]
        ),
        "config_ref": "config.json",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                             "axes.spines.top": False, "axes.spines.right": False})
        c = clean_ci["luminosity_distance"]
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.axhspan(c["lo"], c["hi"], color="#1f77b4", alpha=0.15, lw=0, label="clean 90% CI")
        ax.axhline(c["med"], color="#1f77b4", lw=1.0, label="clean median")
        for sub, mk, lab in ((h1, "o", "gate on H1"), (both, "s", "gate on H1+L1")):
            if not len(sub):
                continue
            y = sub["luminosity_distance_med"].to_numpy()
            lo = y - sub["luminosity_distance_lo"].to_numpy()
            hi = sub["luminosity_distance_hi"].to_numpy() - y
            ax.errorbar(sub["t_rel"], y, yerr=[lo, hi], fmt=mk, ms=3.5, color="#2a9d8f" if mk == "o" else "#7f7f7f",
                        ecolor="#2a9d8f" if mk == "o" else "#7f7f7f", elinewidth=0.7, capsize=1.5, label=lab)
        ax.set_xlabel("gate centre relative to trigger [s]")
        ax.set_ylabel(r"$d_L$ [Mpc] (median, 90% CI)")
        ax.set_ylim(0, 60)
        ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=2)
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / "figS2_clean_gate_cost.pdf")
        fig.savefig(figdir / "figS2_clean_gate_cost.png", dpi=300)
        fig.savefig(outdir / "figS2_clean_gate_cost.pdf")
        plt.close(fig)
    except Exception as e:
        logger.warning("figure failed: %s", e)

    repro = f"""# Reproduce the clean-gate cost control

Forced `adapt_full` gates (Tukey, half-width {cfg['gate_half_s']} s) on *clean* GW170817
data; N={cfg['n_placements_h1']} placements on H1 with gate centre `t_rel` ~ U{tuple(cfg['t_rel_range'])} s
relative to the trigger, plus {cfg['n_placements_h1_l1']} placements gating H1 and L1 at the same `t_rel`.
Posterior sampled with the frozen official DINGO-BNS ({cfg['num_samples']} samples per placement).

```bash
conda activate adapt_env
cd {REPO_ROOT}
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/clean_gate_control.py --seed {cfg['seed']} --num-samples {cfg['num_samples']} --outdir {outdir}
# smoke: python examples/clean_gate_control.py --smoke --outdir results/clean_gate_control_smoke
```

## Outputs

- `results.csv` -- per placement: d_L / chirp_mass / mass_ratio / theta_jn at 5/50/95 %,
  `gated_recovers` (same criterion as the stress grids), JS divergence vs clean for the five
  core parameters, fraction of in-band packaged-strain power removed by the gate.
- `summary.json` -- recovery rate, max JS, mean |delta d_L median|, mean power removed.
- `paper/figures/figS2_clean_gate_cost.pdf` -- d_L median +/- 90 % CI vs gate centre.
"""
    (outdir / "REPRODUCE.md").write_text(repro)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default=DEFAULT_OUTDIR)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--n-placements", type=int, default=20)
    p.add_argument("--n-both", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="2 H1 + 1 H1/L1 placements, 64 samples")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
