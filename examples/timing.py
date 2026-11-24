#!/usr/bin/env python3
"""R5: wall times for the network draw, DINGO-IS, and one gated-likelihood pass.

Run this last, on an otherwise idle machine. --smoke is 300 samples and one repeat.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from _repo import resolve_under_repo

logger = logging.getLogger("timing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default="results/revision")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--stage3", type=resolve_under_repo, default="results/gated_is_stage3")
    p.add_argument("--real-dir", type=resolve_under_repo, default="results/stress_real_glitches_v3_final")
    p.add_argument("--smoke", action="store_true", help="300 samples and 1 repeat")
    return p


def _median_gate(stage3: Path) -> dict:
    rows = []
    for f in (stage3 / "cells").glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("gate_total_s") and r.get("gates"):
            rows.append(r)
    if not rows:
        raise RuntimeError(f"no gated cells in {stage3}")
    med = float(np.median([r["gate_total_s"] for r in rows]))
    rows.sort(key=lambda r: (abs(float(r["gate_total_s"]) - med), r["id"]))
    return rows[0]


def _cpu() -> dict:
    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    ncpu = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip())
    return {"cpu": brand, "ncpu": ncpu}


def main(argv=None) -> None:
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    from adapt.gated_likelihood import build_ufd_likelihood, theta_frame
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.localize import excess_gates
    from official_control import run_importance_sampling
    from revision_eval import chunked_gated

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    repeats_net, repeats_is, repeats_gate = (1, 1, 1) if args.smoke else (5, 3, 3)
    if args.smoke:
        args.num_samples, args.batch_size = 300, 300
    args.eps = 1e-3
    args.eps_sensitivity = []
    args.chunk = 20_000
    c = S2.setup(args)
    n = int(args.num_samples)

    net_times = []
    for i in range(repeats_net):
        t0 = time.time()
        c.session.sample_result(c.event.data, c.settings, c.fixed, num_samples=n, batch_size=int(args.batch_size))
        net_times.append(time.time() - t0)
        logger.info("network repeat %d: %.2f s", i, net_times[-1])

    is_times = []
    for i in range(repeats_is):
        fresh = c.session.sample_result(c.event.data, c.settings, c.fixed, num_samples=n, batch_size=int(args.batch_size))
        t0 = time.time()
        run_importance_sampling(fresh, num_processes=int(args.num_processes))
        is_times.append(time.time() - t0)
        logger.info("DINGO-IS repeat %d: %.2f s", i, is_times[-1])

    cell = _median_gate(Path(args.stage3))
    panel = S3.classify_cells(Path(args.real_dir))
    row = panel[panel["cell_id"] == int(cell["cell_id"])].iloc[0]
    inj = S3.Injector(c, Path(args.real_dir))
    poison, meta = inj(row)
    half = float(cell["half"])
    wins, _z = excess_gates({"H1": meta["td_full"]["H1"]}, {"H1": c.asds["H1"]}, c.fs,
                            f_max=c.f_max, pad_s=half, sigma=8.0, detectors=["H1"])
    gated = rebuild_event_from_gated_td(
        poison, td_by_det={"H1": meta["td_full"]["H1"]}, gates=wins, sample_rate=c.fs, roll_off=c.roll_off,
        f_max=c.f_max, original_asds=c.asds, fill="zero", gate_band=(20.0, 23.0),
    )
    res, _raw = S2.draw(c, gated.data, n)
    base = build_ufd_likelihood(res)
    theta = theta_frame(res)
    gate_times = []
    for i in range(repeats_gate):
        t0 = time.time()
        chunked_gated(c, base, wins, theta)
        gate_times.append(time.time() - t0)
        logger.info("gated likelihood repeat %d: %.2f s", i, gate_times[-1])

    out = {
        **_cpu(),
        "num_processes": int(args.num_processes),
        "num_samples": n,
        "network_only_s": {"repeats": net_times, "median": float(np.median(net_times))},
        "dingo_is_s": {"repeats": is_times, "median": float(np.median(is_times))},
        "gated_likelihood_s": {
            "cell": cell["id"], "gate_total_s": float(cell["gate_total_s"]),
            "repeats": gate_times, "median": float(np.median(gate_times)),
        },
    }
    dest = Path(args.outdir) / "timing.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    logger.info("timing medians: network %.2fs, DINGO-IS %.2fs, gated %.2fs",
                out["network_only_s"]["median"], out["dingo_is_s"]["median"], out["gated_likelihood_s"]["median"])


if __name__ == "__main__":
    main()
