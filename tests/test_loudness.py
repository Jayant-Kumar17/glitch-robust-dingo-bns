"""Tests for the whitened optimal SNR helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_rho_w_matches_analytic_for_white_noise_asd():
    """For a flat ASD, rho_w^2 = 4 * sum |g(f)|^2 df / S_n = 2 * sum g(t)^2 dt / S_n
    (Parseval for the one-sided transform)."""
    from adapt.loudness import scale_to_rho, whitened_optimal_snr

    fs = 4096.0
    T = 8.0
    n = int(T * fs)
    t = np.arange(n) / fs
    delta_f = 1.0 / T
    s_n = 1e-46  # strain^2/Hz
    asd = np.full(int(1600 / delta_f) + 1, np.sqrt(s_n))
    # Sine-Gaussian well inside the band and away from the Tukey roll-off.
    g = 1e-22 * np.exp(-((t - 4.0) ** 2) / (2 * 0.05**2)) * np.cos(2 * np.pi * 200 * (t - 4.0))
    rho = whitened_optimal_snr(g, fs, asd=asd, delta_f=delta_f, f_min=20.0, f_max=1500.0, roll_off=0.4)
    rho_analytic = np.sqrt(2.0 * np.sum(g**2) / fs / s_n)
    assert rho == pytest.approx(rho_analytic, rel=0.02)

    g2, k, rho0 = scale_to_rho(g, 100.0, fs, asd=asd, delta_f=delta_f, f_min=20.0, f_max=1500.0)
    assert rho0 == pytest.approx(rho)
    assert whitened_optimal_snr(g2, fs, asd=asd, delta_f=delta_f, f_min=20.0, f_max=1500.0) == pytest.approx(100.0, rel=1e-6)
    assert k == pytest.approx(100.0 / rho)


def test_rho_w_ignores_out_of_band_power():
    from adapt.loudness import whitened_optimal_snr

    fs = 4096.0
    n = int(8 * fs)
    t = np.arange(n) / fs
    asd = np.full(int(1600 * 8) + 1, 1e-23)
    g_low = 1e-22 * np.sin(2 * np.pi * 5.0 * t) * np.exp(-((t - 4) ** 2) / (2 * 0.5**2))
    assert whitened_optimal_snr(g_low, fs, asd=asd, delta_f=1 / 8, f_min=20.0, f_max=1500.0) < 0.05 * np.sqrt(
        2 * np.sum(g_low**2) / fs / 1e-46
    )
