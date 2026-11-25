#!/usr/bin/env python3
"""In-band leakage of a Tukey gate on clean GW170817 H1 (task C5).

No glitch and no sampler. For each of the first 10 Phase B near-merger
placements and each half-width, compare the packaged strain with and without
the gate, raw and after the 20–23 Hz band limit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from _repo import resolve_under_repo

logger = logging.getLogger("leakage_check")
HALVES = (0.05, 0.1, 0.2, 0.4, 0.8)
F_LO = 23.0


def phase_b_trels(n: int = 10, seed: int = 0) -> list[float]:
    """First ``n`` draws of Stage 2's seed-0 uniform on [-2, -0.2] s."""
    rng = np.random.default_rng(int(seed))
    return [float(rng.uniform(-2.0, -0.2)) for _ in range(int(n))]


def whitened_norm(delta, asd, df: float, f_lo: float, f_hi: float) -> float:
    """rho = sqrt(4 df sum_band |delta|^2 / S_n). Placeholder ASD bins are excluded."""
    d = np.asarray(delta)
    a = np.asarray(asd, dtype=np.float64).ravel()
    n = min(d.size, a.size)
    d, a = d[:n], a[:n]
    freqs = np.arange(n) * float(df)
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi)) & (a > 0.0) & (a < 1e-6)
    if not np.any(band):
        return float("nan")
    sn = a[band] ** 2
    return float(np.sqrt(4.0 * float(df) * np.sum(np.abs(d[band]) ** 2 / sn)))


def _one(x, w, fs, roll_off, f_max, n_freq, tb, asd, df, f_hi) -> float:
    from adapt.packaging import package_fd

    plain = package_fd(x, fs, roll_off=roll_off, f_max=f_max, n_freq=n_freq, time_buffer=tb)
    gated = package_fd(x * w, fs, roll_off=roll_off, f_max=f_max, n_freq=n_freq, time_buffer=tb)
    return whitened_norm(gated - plain, asd, df, F_LO, f_hi)


def run(c, outdir: Path) -> dict:
    from adapt.glitch_excision import GateWindow, tukey_gate_weights
    from adapt.packaging import band_limit_td

    x = np.asarray(c.td["H1"], dtype=np.float64)
    n_freq = len(np.asarray(c.event.data["waveform"]["H1"]))
    asd = np.asarray(c.asds["H1"])
    df = 1.0 / float(c.T)
    cells = []
    for k, t_rel in enumerate(phase_b_trels(10, 0)):
        for h in HALVES:
            t_peak = float(c.t_trig) + t_rel
            wins = [GateWindow("H1", t_peak - h, t_peak + h)]
            w = tukey_gate_weights(c.N, c.fs, wins, t0=0.0)
            rho_raw = _one(x, w, c.fs, c.roll_off, c.f_max, n_freq, c.tb, asd, df, c.f_max)
            xb = band_limit_td(x, c.fs, 20.0, 23.0, roll_off=c.roll_off)
            rho_band = _one(xb, w, c.fs, c.roll_off, c.f_max, n_freq, c.tb, asd, df, c.f_max)
            ratio = float(rho_raw / rho_band) if rho_band and np.isfinite(rho_band) and rho_band != 0 else None
            cells.append({"placement": k, "t_rel": t_rel, "half": h,
                          "rho_raw": rho_raw, "rho_band": rho_band, "ratio": ratio})
            logger.info("placement %d h=%g rho_raw=%.4g rho_band=%.4g", k, h, rho_raw, rho_band)
    summary = []
    for h in HALVES:
        sub = [r for r in cells if r["half"] == h]
        ratios = [r["ratio"] for r in sub if r["ratio"] is not None and np.isfinite(r["ratio"])]
        summary.append({
            "half": h,
            "median_rho_raw": float(np.median([r["rho_raw"] for r in sub])),
            "median_rho_band": float(np.median([r["rho_band"] for r in sub])),
            "median_ratio": float(np.median(ratios)) if ratios else None,
        })
    out = {"placements": phase_b_trels(10, 0), "band_hz": [F_LO, float(c.f_max)],
           "cells": cells, "summary": summary}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "leakage.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    import gated_is_stage2 as S2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default="results/revision")
    p.add_argument("--num-processes", type=int, default=1)
    args = p.parse_args(argv)
    # setup() wants the Stage 2 argument names. No samples are drawn.
    args.num_samples = 1
    args.batch_size = 1
    args.eps = 1e-3
    args.eps_sensitivity = []
    c = S2.setup(args)
    run(c, Path(args.outdir))


if __name__ == "__main__":
    main()
