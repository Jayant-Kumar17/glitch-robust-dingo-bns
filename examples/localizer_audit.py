#!/usr/bin/env python3
"""False-trigger audit of the whitened-excess localiser.

Three populations, each analysed with the GW170817 analysis ASD of that detector:
  1. the clean GW170817 analysis segment itself (H1, L1, V1),
  2. off-source 128 s segments cut from the same demo frame files (real noise),
  3. stationary Gaussian draws coloured by the analysis ASD.
Thresholds are fixed in advance (6, 8, 10 sigma); nothing is tuned on the event.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _repo import resolve_under_repo

logger = logging.getLogger("localizer_audit")
SIGMAS = (6.0, 8.0, 10.0)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default="results/localizer_audit_v3")
    p.add_argument("--n-gauss", type=int, default=1000)
    p.add_argument("--max-offsource", type=int, default=60, help="max off-source segments per detector")
    p.add_argument("--pad-s", type=float, default=0.1)
    args = p.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from adapt.dingo_bns_demo import discover_assets, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from adapt.localize import excess_gates
    from adapt.packaging import coloured_noise_td

    assets = discover_assets()
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    fs = float(settings.get("f_s", 4096.0))
    T = float(settings.get("T", 128.0))
    tb = float(settings.get("time_buffer", 2.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    n = int(round(T * fs))
    dets = ("H1", "L1", "V1")
    asds = {d: np.asarray(event.data["asds"][d], dtype=np.float64) for d in dets}

    def run(td, det):
        out = {}
        _, zmax = excess_gates({det: td}, {det: asds[det]}, fs, f_max=f_max, pad_s=args.pad_s, sigma=1e9)
        out["zmax"] = zmax[det]
        for s in SIGMAS:
            g, _ = excess_gates({det: td}, {det: asds[det]}, fs, f_max=f_max, pad_s=args.pad_s, sigma=s)
            out[f"fires_{s:g}"] = bool(g)
            out[f"gate_s_{s:g}"] = float(sum(w.t_end - w.t_start for w in g))
        return out

    report = {"sigmas": list(SIGMAS), "pad_s": args.pad_s}

    # 1. the event segment
    report["gw170817_segment"] = {}
    for det in dets:
        td, _, _ = load_full_event_td(assets, settings, det)
        report["gw170817_segment"][det] = run(td, det)
    logger.info("GW170817 segment: %s", report["gw170817_segment"])

    # 2. off-source segments from the same frame files
    report["offsource"] = {}
    trigger = float(assets["trigger_time"])
    try:
        from gwpy.timeseries import TimeSeries

        for det in dets:
            ts = TimeSeries.read(str(assets["strain_gwf"][det]), channel=assets["channels"][det])
            if float(ts.sample_rate.value) != fs:
                ts = ts.resample(fs)
            t0, t1 = float(ts.t0.value), float(ts.t0.value) + float(ts.duration.value)
            rows = []
            start = t0 + 8.0
            while start + T <= t1 - 8.0 and len(rows) < args.max_offsource:
                end = start + T
                if end > trigger - (T - tb) - 4.0 and start < trigger + tb + 4.0:
                    start = trigger + tb + 4.0
                    continue
                x = np.asarray(ts.crop(start, end).value, dtype=np.float64)[:n]
                if x.size == n and np.all(np.isfinite(x)):
                    r = run(x, det)
                    r["gps_start"] = start
                    rows.append(r)
                start = end
            report["offsource"][det] = {
                "n": len(rows),
                **{
                    f"fire_rate_{s:g}": (float(np.mean([r[f"fires_{s:g}"] for r in rows])) if rows else None)
                    for s in SIGMAS
                },
                "zmax_median": float(np.median([r["zmax"] for r in rows])) if rows else None,
                "rows": rows,
            }
            logger.info("off-source %s: n=%d", det, len(rows))
    except Exception as exc:  # frame files may be short; report rather than fail
        logger.exception("off-source audit failed")
        report["offsource_error"] = f"{type(exc).__name__}: {exc}"

    # 3. Gaussian draws
    report["gaussian"] = {}
    for det in ("H1", "L1"):
        fires = {s: 0 for s in SIGMAS}
        zm = []
        for k in range(int(args.n_gauss)):
            x = coloured_noise_td(asds[det], n, fs, seed=10_000 * (1 + dets.index(det)) + k)
            r = run(x, det)
            zm.append(r["zmax"])
            for s in SIGMAS:
                fires[s] += int(r[f"fires_{s:g}"])
        report["gaussian"][det] = {
            "n": int(args.n_gauss),
            **{f"fires_{s:g}": int(fires[s]) for s in SIGMAS},
            "zmax_median": float(np.median(zm)),
            "zmax_max": float(np.max(zm)),
        }
        logger.info("Gaussian %s: %s", det, report["gaussian"][det])

    (outdir / "summary.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", outdir / "summary.json")


if __name__ == "__main__":
    main()
