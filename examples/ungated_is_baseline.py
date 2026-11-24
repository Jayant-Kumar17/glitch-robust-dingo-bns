#!/usr/bin/env python3
"""R2: ordinary DINGO-IS on the ungated glitch-contaminated package, all 195 cells."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

from _repo import resolve_under_repo

logger = logging.getLogger("ungated_is")
CLEAN_CI = "results/gated_is_stage3/clean_reference.json"


def build_parser() -> argparse.ArgumentParser:
    from revision_eval import add_runtime_args

    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p, default_outdir="results/revision/ungated_is", default_samples=10_000)
    p.add_argument("--real-dir", type=resolve_under_repo, default="results/stress_real_glitches_v3_final")
    p.add_argument("--clean-ci", type=resolve_under_repo, default=CLEAN_CI)
    return p


def _write(path: Path, rec: dict) -> None:
    import gated_is_stage2 as S2

    path.write_text(json.dumps(rec, indent=2, default=S2._json_default))


def main(argv=None) -> None:
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    from revision_eval import apply_smoke, over_budget
    from stress_gw170817 import _gated_recovers, _poison_collapsed
    from official_control import INFERENCE_PARAMS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    apply_smoke(args, cells=2)
    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    clean_ci = json.loads(Path(args.clean_ci).read_text())["clean_ci"]
    cells = S3.classify_cells(Path(args.real_dir))
    if args.smoke:
        cells = cells.head(int(args.max_cells))
    done = set()
    for f in cells_dir.glob("*.json"):
        done.add(int(json.loads(f.read_text())["cell_id"]))
    t_start = time.time()
    c = S2.setup(args)
    inj = S3.Injector(c, Path(args.real_dir))
    for _, row in cells.iterrows():
        cid = int(row["cell_id"])
        if cid in done:
            continue
        if over_budget(t_start, args.max_hours):
            logger.warning("time budget reached before cell %s", cid)
            break
        dest = cells_dir / f"{cid:03d}.json"
        logger.info("=== ungated %s %s ===", row["group"], cid)
        t0 = time.time()
        rec = {"cell_id": cid, "group": str(row["group"]), "glitch": float(row["gs_event_time"]),
               "label": row["gs_label"], "snr_bin": row["gs_snr_bin"],
               "rho_w_archived": float(row["injected_rho_w"])}
        try:
            poison, meta = inj(row)
            rec["rho_w_rebuilt"] = float(meta["injected_rho_w"])
            rel = abs(rec["rho_w_rebuilt"] - rec["rho_w_archived"]) / max(rec["rho_w_archived"], 1e-12)
            rec["injection_match"] = bool(rel < 1e-6)
            if not rec["injection_match"]:
                rec["error"] = f"injection mismatch: rebuilt {rec['rho_w_rebuilt']:.6g} vs archived {rec['rho_w_archived']:.6g}"
                rec["elapsed_s"] = time.time() - t0
                _write(dest, rec)
                continue
            res, raw = S2.draw(c, poison, int(args.num_samples))
            w = res.samples["weights"].to_numpy(float)
            eff = float(res.sample_efficiency)
            d_L = S2._ci(res.samples["luminosity_distance"].to_numpy(float), w)
            net = S2._ci(raw["luminosity_distance"].to_numpy(float))
            rec.update({
                "n": int(args.num_samples),
                "efficiency": eff,
                "n_eff": eff * int(args.num_samples),
                "d_L": d_L,
                "recovers": bool(_gated_recovers(d_L, clean_ci)),
                "collapsed": bool(_poison_collapsed(d_L)),
                "network_only": {**net, "recovers": bool(_gated_recovers(net, clean_ci)),
                                 "collapsed": bool(_poison_collapsed(net))},
                "elapsed_s": time.time() - t0,
            })
            keep = [p for p in INFERENCE_PARAMS if p in res.samples.columns]
            out = res.samples[keep].copy()
            out["weight"] = w
            out.to_csv(outdir / f"{cid:03d}_samples.csv.gz", index=False)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()
            rec["elapsed_s"] = time.time() - t0
            logger.exception("cell %s failed", cid)
        _write(dest, rec)
    logger.info("ungated IS finished in %.2f h", (time.time() - t_start) / 3600.0)


if __name__ == "__main__":
    main()
