"""Gated likelihood: exactness against brute force, glitch invariance, and a misplaced-gate control.

If any of these fail, do NOT relax them: the gated importance-sampling results would be invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _setup(fs=256.0, T=8.0, f_max=100.0, f_min=20.0, tb=2.0, seed=0):
    N = int(fs * T)
    n_freq = int(f_max * T) + 1
    f = np.arange(n_freq) / T
    asd = 1e-23 * (1 + (30.0 / np.maximum(f, 1.0)) ** 4) * (1 + (f / 200.0) ** 2)
    asd[f < f_min] = 1.0
    rng = np.random.default_rng(seed)
    return N, n_freq, f, asd, rng, fs, T, tb


def _coeff(r, fs, n_freq, f, tb):
    """Package coefficients S F[r] (no segment window: the gate is far from the edges)."""
    return (np.fft.rfft(r) / fs)[:n_freq] * np.exp(-2j * np.pi * f * tb)


def test_matches_brute_force_marginal():
    """Q - 2*correction equals min over the gated samples of the regularised quadratic form."""
    from adapt.gated_likelihood import GateOperator

    N, n_freq, f, asd, rng, fs, T, tb = _setup()
    ns = 0.7
    eps = 1e-3
    band = asd < 1e-6
    band[0] = False
    f_full = np.arange(N // 2 + 1) / T
    band_full = np.zeros(N // 2 + 1, bool)
    band_full[:n_freq] = band
    a_full = np.ones(N // 2 + 1)
    a_full[:n_freq] = np.where(band, asd * ns, 1.0)
    c_in = np.where(band_full, 1.0 / a_full ** 2, 0.0)
    c_full = np.where(band_full, c_in, eps * np.median(c_in[band_full]))
    c_full[0] = c_full[-1] = 0.0
    I = np.eye(N)
    Cmat = np.fft.rfft(I, axis=0) / fs * np.exp(-2j * np.pi * f_full * tb)[:, None]  # (N/2+1) x N
    Wm = Cmat * np.sqrt(c_full)[:, None]
    B = np.vstack([Wm.real, Wm.imag])
    gate = np.zeros(N, bool)
    gate[700:760] = True
    gate[1300:1310] = True  # two separate gates
    g = np.flatnonzero(gate)
    op = GateOperator(asd, ns, fs, T, tb, gate, eps=eps)
    for trial in range(3):
        r = rng.standard_normal(N) * 3e-22
        rw = (np.fft.rfft(r) / fs)[:n_freq] * np.exp(-2j * np.pi * f * tb) / (asd * ns)
        rw = np.where(band, rw, 0.0)
        Q = float(np.sum(np.abs(rw) ** 2))
        # data: in-band residual, zero out of band
        b_in = np.concatenate([(Wm @ r).real * band_full, (Wm @ r).imag * band_full])
        assert np.isclose(Q, float(b_in @ b_in), rtol=1e-10)
        sol, *_ = np.linalg.lstsq(B[:, g], -b_in, rcond=None)
        Q_ng = float(np.sum((b_in + B[:, g] @ sol) ** 2))
        corr = op.correction(rw)
        assert np.isclose(Q - 2.0 * corr, Q_ng, rtol=1e-7), (Q, corr, Q_ng)


def test_parameter_differences_insensitive_to_eps():
    """What importance sampling uses: differences between models, for several eps."""
    from adapt.gated_likelihood import GateOperator

    N, n_freq, f, asd, rng, fs, T, tb = _setup(fs=1024.0, T=16.0, f_max=400.0, f_min=23.0)
    ns = 1.0
    band = asd < 1e-6
    t = np.arange(N) / fs
    tc = T - tb
    x = np.fft.irfft(
        (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1))
        * np.concatenate([np.where(band, asd, 0.0), np.zeros(N // 2 + 1 - n_freq)]) * np.sqrt(N * fs) / 2, n=N)

    def chirp(scale, phase):
        tau = np.clip(tc - t, 1e-3, None)
        fr = np.clip(40 * (tau / 4) ** (-3 / 8), 0, 350)
        return scale * 3e-22 * (fr / 40) ** (2 / 3) * np.cos(np.cumsum(2 * np.pi * fr / fs) + phase) * (t < tc - 0.002)

    def w(xx):
        return np.where(band, (np.fft.rfft(xx) / fs)[:n_freq] * np.exp(-2j * np.pi * f * tb) / (asd * ns), 0.0)

    d = w(x + chirp(1.0, 0.0))
    models = [w(chirp(1.0, 0.0)), w(chirp(1.05, 0.0)), w(chirp(1.0, 0.4)), w(chirp(0.7, 0.0))]
    gate = np.abs(t - (tc - 0.5)) <= 0.3
    diffs = []
    for eps in (1e-2, 1e-3, 1e-4):
        op = GateOperator(asd, ns, fs, T, tb, gate, eps=eps)
        L = [-0.5 * np.sum(np.abs(d - m) ** 2) + op.correction(d - m) for m in models]
        diffs.append(np.array(L[1:]) - L[0])
    diffs = np.array(diffs)  # rows: eps = 1e-2, 1e-3 (default), 1e-4
    tol = 0.2 + 0.01 * np.abs(diffs[1])
    assert np.all(np.abs(diffs[2] - diffs[1]) < tol), diffs
    assert np.all(np.abs(diffs[0] - diffs[1]) < 2 * tol), diffs


def test_zero_gate_gives_zero_correction():
    from adapt.gated_likelihood import GateOperator

    N, n_freq, f, asd, rng, fs, T, tb = _setup()
    op = GateOperator(asd, 1.0, fs, T, tb, np.zeros(N, bool))
    rw = rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)
    assert op.correction(rw) == 0.0


def _invariance(gate_shift_s, fs=2048.0, T=16.0, f_max=900.0):
    """Return (|delta gated L|, |delta ungated L|) when a loud glitch is added inside the gate."""
    from adapt.gated_likelihood import GateOperator, whittle_log_likelihood
    from adapt.packaging import package_fd

    N, n_freq, f, asd, rng, fs, T, tb = _setup(fs=fs, T=T, f_max=f_max, f_min=23.0)
    ns = 1.0 / np.sqrt(4.0 / T)
    x = np.fft.irfft(
        (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1))
        * np.concatenate([np.where(asd < 1e-6, asd, 0.0), np.zeros(N // 2 + 1 - n_freq)])
        * np.sqrt(N * fs) / 2,
        n=N,
    )
    t = np.arange(N) / fs
    tc = (T - tb) - 1.0
    gate = np.abs(t - tc) <= 0.2
    burst = np.sin(2 * np.pi * 150 * (t - tc)) * np.exp(-((t - tc) / 0.03) ** 2) * gate * 1e-19
    d0 = package_fd(x, fs, roll_off=0.4, f_max=f_max, n_freq=n_freq, time_buffer=tb) / (asd * ns)
    d1 = package_fd(x + burst, fs, roll_off=0.4, f_max=f_max, n_freq=n_freq, time_buffer=tb) / (asd * ns)
    band = asd < 1e-6
    d0, d1 = np.where(band, d0, 0), np.where(band, d1, 0)
    mu = np.zeros(n_freq, complex)  # any model; zero is the hardest case (largest residual)
    shift = int(round(gate_shift_s * fs))
    op = GateOperator(asd, ns, fs, T, tb, np.roll(gate, shift), eps=1e-3)
    L0 = whittle_log_likelihood({"H1": d0}, {"H1": mu})
    L1 = whittle_log_likelihood({"H1": d1}, {"H1": mu})
    G0 = L0 + op.correction(d0 - mu)
    G1 = L1 + op.correction(d1 - mu)
    return abs(G1 - G0), abs(L1 - L0)


def test_glitch_inside_gate_does_not_change_gated_likelihood():
    dg, du = _invariance(0.0)
    assert du > 1e4, du  # the glitch is loud for the ordinary likelihood
    assert dg < 1e-6 * du and dg < 0.05, (dg, du)


def test_misplaced_gate_is_not_invariant():
    # a gate shifted by the cyclic time buffer (the old bug) must fail the invariance check
    dg, du = _invariance(2.0)
    assert dg > 0.1 * du, (dg, du)


def test_reweight_recovers_target():
    from adapt.gated_likelihood import reweight
    import pandas as pd

    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 2.0, 200_000)  # proposal N(0, 2^2)
    log_q = -0.5 * (x / 2.0) ** 2 - np.log(2.0)
    s = pd.DataFrame({"x": x, "log_prob": log_q, "log_prior": np.zeros_like(x)})
    out = reweight(s, -0.5 * (x - 1.0) ** 2)  # target N(1, 1)
    m = np.sum(out["weights"] * x) / np.sum(out["weights"])
    assert abs(m - 1.0) < 0.02
    assert 0.3 < out["efficiency"] < 0.9
