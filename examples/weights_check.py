#!/usr/bin/env python3
"""R6: uniform-grid reweighting with an empty gate versus DINGO's own weights.

Uses the first 10^4 samples of the saved R1 result. Does not draw again.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _repo import resolve_under_repo

logger = logging.getLogger("weights_check")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", type=resolve_under_repo, default="results/revision/clean_ref_v2")
    p.add_argument("--outdir", type=resolve_under_repo, default="results/revision")
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--smoke", action="store_true", help="use every saved sample, at most 300")
    return p


def _neff(w) -> float:
    w = np.asarray(w, float)
    if not np.any(w):
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def main(argv=None) -> None:
    import gated_is_stage2 as S2
    from adapt.gated_likelihood import GatedLikelihood, build_ufd_likelihood, evaluate_components, reweight, theta_frame
    from dingo.gw.result import Result
    from official_control import INFERENCE_PARAMS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    n_take = 300 if args.smoke else int(args.num_samples)
    ref = Path(args.ref)
    res = Result(file_name=str(ref / "result.hdf5"))
    res.samples = res.samples.iloc[:n_take].reset_index(drop=True)
    n = len(res.samples)
    # Stage 2 setup only for the segment settings. The likelihood is the saved result's.
    args.batch_size = n
    args.eps = 1e-3
    args.eps_sensitivity = []
    c = S2.setup(args)
    base = build_ufd_likelihood(res)
    lik = GatedLikelihood(base, {}, sample_rate=c.fs, duration=c.T, time_buffer=c.tb)
    L0, corr = evaluate_components(lik, theta_frame(res), num_processes=int(args.num_processes))
    ours = reweight(res.samples, L0)
    dingo_w = res.samples["weights"].to_numpy(float)
    dingo_w = dingo_w / np.mean(dingo_w)
    lo = np.log(np.maximum(ours["weights"], 1e-300))
    ld = np.log(np.maximum(dingo_w, 1e-300))
    finite = np.isfinite(lo) & np.isfinite(ld)
    pearson = float(np.corrcoef(lo[finite], ld[finite])[0, 1]) if finite.sum() > 2 else float("nan")
    diff = lo[finite] - ld[finite]
    diff = diff - np.mean(diff)
    spread = float(np.std(diff))
    js = S2._js(res.samples, ours["weights"], res.samples, dingo_w, INFERENCE_PARAMS)
    js_max = max(js.values()) if js else float("nan")
    verdict = (f"uniform-grid empty-gate log-weights vs DINGO: Pearson {pearson:.6f}, "
               f"std of the mean-removed difference {spread:.4g}, d_L JS {js.get('luminosity_distance', float('nan')):.4g}, "
               f"max JS {js_max:.4g}")
    out = {
        "n": n,
        "max_abs_empty_gate_correction": float(np.nanmax(np.abs(corr))),
        "n_eff_uniform_grid": _neff(ours["weights"]),
        "n_eff_dingo": _neff(dingo_w),
        "pearson_log_weight": pearson,
        "std_log_weight_difference": spread,
        "js_d_L": js.get("luminosity_distance"),
        "js_max": js_max,
        "verdict": verdict,
    }
    dest = Path(args.outdir) / "weights_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    logger.info("%s", verdict)


if __name__ == "__main__":
    main()
