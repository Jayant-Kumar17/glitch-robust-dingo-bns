"""Shared helpers for the referee-revision drivers. No sampling happens here."""

from __future__ import annotations

import argparse

import numpy as np

from _repo import resolve_under_repo


def add_runtime_args(p: argparse.ArgumentParser, *, default_outdir: str, default_samples: int = 10_000) -> None:
    p.add_argument("--outdir", type=resolve_under_repo, default=default_outdir)
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--num-samples", type=int, default=default_samples)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--chunk", type=int, default=20_000)
    p.add_argument("--smoke", action="store_true", help="300 samples and a two-cell crash test")


def apply_smoke(args, *, cells: int = 2) -> None:
    """Smoke overrides. Callers still set args.eps_sensitivity for stage-2 setup."""
    args.eps_sensitivity = []
    if args.smoke:
        args.num_samples = 300
        args.batch_size = 300
        args.max_cells = cells


def chunked_gated(c, base, wins, theta):
    """L0 + gate correction, in chunks, using the Stage 2 evaluator."""
    import gated_is_stage2 as S2

    ops = S2.gate_ops(c, base, wins, [c.args.eps])[c.args.eps]
    payload = S2._Payload(base, S2.unit_ops(c, base, sorted(ops)), {i: ops[i].idx for i in ops})
    chunk = int(getattr(c.args, "chunk", 20_000))
    parts = []
    for i0 in range(0, len(theta), chunk):
        L0_i, u_i = S2.evaluate(payload, theta.iloc[i0:i0 + chunk], int(c.args.num_processes))
        parts.append(L0_i + S2.corrections(ops, u_i, payload.windows))
        del u_i
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def over_budget(t_start: float, max_hours: float) -> bool:
    import time

    return (time.time() - t_start) / 3600.0 > float(max_hours)
