#!/usr/bin/env python3
"""Smoke: untrained glitch-robust student matches teacher on clean context/log-prob."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "src", REPO / "DINGO-BNS" / "dingo", REPO):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("smoke_glitch_robust")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from train_bns_glitch_robust import (
        DEFAULT_BNS_CKPT,
        PairedGlitchInjectionDataset,
        attach_glitch_robust_embedding,
        calibrate_norm_stats,
        freeze_dingo,
        select_device,
        teacher_context,
    )
    from train_bns_spectrogram import build_flow_wrapper, load_bns_checkpoint

    if not DEFAULT_BNS_CKPT.is_file():
        raise SystemExit(f"Missing baseline ckpt: {DEFAULT_BNS_CKPT}")

    device = select_device()
    raw = load_bns_checkpoint(DEFAULT_BNS_CKPT)
    metadata = raw["metadata"]
    teacher = build_flow_wrapper(raw).float().to(device).eval()
    student = build_flow_wrapper(raw)
    student = attach_glitch_robust_embedding(student)
    student = student.float().to(device)
    freeze_dingo(student)
    student.eval()

    stft_kwargs = {
        "n_time": 32,
        "n_freq": 128,
        "n_fft": 2048,
        "win_length": 1024,
        "hop_length": 256,
    }
    norm_stats = calibrate_norm_stats(
        metadata, n_samples=4, seed=0, stft_kwargs=stft_kwargs
    )
    ds = PairedGlitchInjectionDataset(
        metadata,
        length=2,
        seed=1,
        held_out=False,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        curriculum_severity_max=4.0,
    )
    batch = ds[0]
    sc = batch["strain_clean"].unsqueeze(0).to(device)
    pc = batch["spectrogram_clean"].unsqueeze(0).to(device)
    ec = batch["log_energy_clean"].unsqueeze(0).to(device)
    z = batch["context_z"].unsqueeze(0).to(device)
    y = batch["y"].unsqueeze(0).to(device)

    with torch.no_grad():
        t_ctx = teacher_context(teacher, sc, z)
        s_ctx = student.embedding_net(sc, pc, ec, z)
        ctx_err = (t_ctx - s_ctx).abs().max().item()
        lp_t = float(teacher.log_prob(y, sc, z).item())
        lp_s = float(student.log_prob(y, sc, pc, ec, z).item())
    logger.info("clean context max|Δ|=%.3e  log_prob teacher=%.4f student=%.4f", ctx_err, lp_t, lp_s)
    if ctx_err > 0.05:
        raise SystemExit(f"FAIL: clean context drift {ctx_err}")
    if abs(lp_t - lp_s) > 0.05:
        raise SystemExit(f"FAIL: clean log_prob mismatch {lp_t} vs {lp_s}")
    logger.info("SMOKE OK: untrained custom ≈ baseline on clean")


if __name__ == "__main__":
    main()
