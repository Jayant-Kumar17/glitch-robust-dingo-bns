#!/usr/bin/env python3
"""Shape smoke-test for Spectrogram2DNet + 3-ch mag/coherence STFT context."""

from __future__ import annotations

import sys

import numpy as np
import torch

from adapt.models import Spectrogram2DNet
from adapt.stft_context import (
    build_csd_spectrogram_from_td,
    fd_waveform_to_td_crop,
    log_unit_scale_stft_grid,
)
from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS, compute_complex_stft_grid


def _test_csd_and_stft() -> None:
    rng = np.random.default_rng(0)
    sr = 4096.0
    n = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sr))
    td_h = rng.normal(size=n).astype(np.float64)
    td_l = rng.normal(size=n).astype(np.float64)
    td_v = rng.normal(size=n).astype(np.float64)

    # Complex STFT grid shape.
    z = compute_complex_stft_grid(td_h, sr, n_fft=4096, win_length=2048, hop_length=512)
    if z.shape != (5, 128) or not np.iscomplexobj(z):
        raise AssertionError(f"complex STFT shape/dtype bad: {z.shape} {z.dtype}")

    tensor, energies = build_csd_spectrogram_from_td(
        {"H1": td_h, "L1": td_l, "V1": td_v},
        sr,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        energy_detectors=("H1", "L1", "V1"),
    )
    if tensor.shape != (3, 5, 128):
        raise AssertionError(f"CSD tensor shape {tensor.shape} != (3, 5, 128)")
    if energies.shape != (3,) or not np.isfinite(energies).all():
        raise AssertionError(f"bad log_energy: {energies}")
    # Louder H1 should raise H1 log-energy.
    _, e_loud = build_csd_spectrogram_from_td(
        {"H1": td_h * 10.0, "L1": td_l, "V1": td_v},
        sr,
        energy_detectors=("H1", "L1", "V1"),
    )
    if not (e_loud[0] > energies[0]):
        raise AssertionError("louder H1 should raise log_energy[0]")

    # Legacy helper still works.
    a, ea = log_unit_scale_stft_grid(td_h, sr)
    if a.shape != (5, 128) or not np.isfinite(ea):
        raise AssertionError("legacy log_unit_scale_stft_grid failed")

    # FD → TD crop helper (whitened).
    duration = 4.0
    n_td = int(round(duration * sr))
    n_rfft = n_td // 2 + 1
    fd = (rng.normal(size=n_rfft) + 1j * rng.normal(size=n_rfft)).astype(np.complex128)
    asd = np.ones(n_rfft, dtype=np.float64)
    crop = fd_waveform_to_td_crop(
        fd,
        sample_rate=sr,
        duration=duration,
        trigger_time=0.0,
        asd=asd,
        noise_std=1.0,
        whiten=True,
        analysis_seconds=SPECTROGRAM_ANALYSIS_SECONDS,
    )
    if crop.shape != (n,) or not np.isfinite(crop).all():
        raise AssertionError(f"fd_waveform_to_td_crop bad: {crop.shape}")

    print(
        f"CSD/STFT: OK → tensor {tensor.shape}, energy {energies}, "
        f"complex {z.shape}, td_crop {crop.shape}"
    )


def _test_encoders() -> None:
    strain = torch.randn(2, 3, 3, 3324)
    spectrogram = torch.randn(2, 3, 5, 128)
    log_energy = torch.randn(2, 3)
    for etype in ("cnn_base", "resnet_deep"):
        model = Spectrogram2DNet(encoder_type=etype, in_channels=3)
        model.eval()
        with torch.no_grad():
            out = model(strain, spectrogram, log_energy)
        if out.shape != torch.Size([2, 128]):
            raise AssertionError(f"{etype}: expected (2,128), got {tuple(out.shape)}")
        if not torch.isfinite(out).all():
            raise AssertionError(f"{etype}: non-finite context")
        print(f"encoder {etype}: OK → {tuple(out.shape)}")


def main() -> int:
    try:
        _test_encoders()
        _test_csd_and_stft()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: Spectrogram2DNet 3-ch mag/coherence + STFT context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
