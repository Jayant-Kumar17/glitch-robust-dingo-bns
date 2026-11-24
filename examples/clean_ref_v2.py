#!/usr/bin/env python3
"""R1: one DINGO-IS draw on the clean GW170817 package, seed 20261125, 10^5 samples."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("clean_ref_v2")


def build_parser() -> argparse.ArgumentParser:
    from revision_eval import add_runtime_args

    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p, default_outdir="results/revision/clean_ref_v2", default_samples=100_000)
    p.add_argument("--seed", type=int, default=20261125)
    return p


def main(argv=None) -> None:
    import torch

    import gated_is_stage2 as S2
    from revision_eval import apply_smoke

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    apply_smoke(args, cells=1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "meta.json").is_file() and (outdir / "samples.csv.gz").is_file() and (outdir / "result.hdf5").is_file():
        logger.info("R1 output already present in %s; skipping", outdir)
        return
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    t0 = time.time()
    c = S2.setup(args)
    res, _raw = S2.draw(c, c.event.data, int(args.num_samples))
    w = res.samples["weights"].to_numpy(float)
    n = int(args.num_samples)
    eff = float(res.sample_efficiency)
    meta = {
        "n": n,
        "seed": int(args.seed),
        "efficiency": eff,
        "n_eff": eff * n,
        "log_evidence": None if res.log_evidence is None else float(res.log_evidence),
        "d_L": S2._ci(res.samples["luminosity_distance"].to_numpy(float), w),
        "elapsed_s": time.time() - t0,
    }
    res.samples.to_csv(outdir / "samples.csv.gz", index=False)
    res.to_file(str(outdir / "result.hdf5"))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("R1 efficiency %.6g n_eff %.1f d_L med %.3f", eff, meta["n_eff"], meta["d_L"]["med"])


if __name__ == "__main__":
    main()
