"""Whitened-excess localiser: silent on Gaussian noise and on a BNS-like chirp, fires on a loud glitch at the right time."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FS = 4096.0
T = 32.0
N = int(FS * T)
F_MAX = 1535.3046875


def _asd():
    f = np.arange(int(F_MAX * T) + 1) / T
    pts = np.array([[1, 3e-17], [10, 5e-20], [20, 1e-21], [23, 4e-22], [30, 1e-22], [50, 2e-23],
                    [100, 8e-24], [200, 6e-24], [500, 1e-23], [1000, 2.5e-23], [2048, 6e-23]])
    a = np.exp(np.interp(np.log(np.maximum(f, 1.0)), np.log(pts[:, 0]), np.log(pts[:, 1])))
    a[f < 23.0] = 1.0  # DINGO placeholder
    return a


def _noise(seed):
    from adapt.packaging import coloured_noise_td

    a = _asd().copy()
    f = np.arange(a.size) / T
    full = np.exp(np.interp(np.log(np.maximum(f, 1.0)), np.log([1, 10, 20, 23]), np.log([3e-17, 5e-20, 1e-21, 4e-22])))
    a[f < 23.0] = np.minimum(full[f < 23.0], 9e-7)  # include a seismic wall in the data
    return coloured_noise_td(a, N, FS, seed=seed)


def _whitened_unit_glitch(t_peak, f0=100.0, q=5.0):
    """Sine-Gaussian, coloured so that its whitened energy is easy to scale."""
    from adapt.stft_context import sine_gaussian_glitch

    w = sine_gaussian_glitch(N, FS, t_peak=t_peak, f0=f0, q=q, amplitude=1.0)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(N, 1 / FS)
    a = np.interp(f, np.arange(_asd().size) / T, _asd())
    return np.fft.irfft(W * np.where(f >= 23.0, a, 0.0), n=N)


def test_silent_on_gaussian_noise():
    from adapt.localize import excess_gates

    fired = 0
    for s in range(20):
        gates, zmax = excess_gates({"H1": _noise(s)}, {"H1": _asd()}, FS, f_max=F_MAX, pad_s=0.1, sigma=8.0)
        assert zmax["H1"] < 6.5, zmax
        fired += bool(gates)
    assert fired == 0


def test_fires_on_glitch_at_the_right_time_and_ignores_a_weak_chirp():
    from adapt.localize import excess_gates, whitened_z

    t_peak = T - 2.0 - 1.0
    g = _whitened_unit_glitch(t_peak)
    x = _noise(3)
    # scale the glitch so its whitened peak is ~30 sigma relative to this noise
    z_noise = whitened_z(x, _asd(), FS, f_max=F_MAX)
    z_both = whitened_z(x + g * 1.0, _asd(), FS, f_max=F_MAX)
    amp = 30.0 / max(np.max(np.abs(z_both - z_noise)), 1e-30)
    gates, zmax = excess_gates({"H1": x + amp * g}, {"H1": _asd()}, FS, f_max=F_MAX, pad_s=0.1, sigma=8.0)
    assert len(gates) == 1, gates
    gw = gates[0]
    assert gw.t_start < t_peak < gw.t_end
    assert gw.t_end - gw.t_start < 0.5, (gw.t_start, gw.t_end)

    # A chirp whose whitened per-sample peak is 3 sigma (a real BNS is far below 1) must not trigger.
    t = np.arange(N) / FS
    t_c = T - 2.0
    tau = np.clip(t_c - t, 0.02, None)
    phase = 2 * np.pi * 30.0 * 8.0 * tau ** (5.0 / 8.0) / 5.0  # f ~ 30 Hz * (tau)^(-3/8)
    w = np.where(t < t_c, np.cos(-phase), 0.0) * np.exp(-((t - t_c + 4.0) / 4.0) ** 2)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(N, 1 / FS)
    a = np.interp(f, np.arange(_asd().size) / T, _asd())
    chirp = np.fft.irfft(W * np.where(f >= 23.0, a, 0.0), n=N)
    zc = whitened_z(x + chirp, _asd(), FS, f_max=F_MAX) - z_noise
    scale = 3.0 / max(np.max(np.abs(zc)), 1e-30)
    gates_c, zmax_c = excess_gates({"H1": x + scale * chirp}, {"H1": _asd()}, FS, f_max=F_MAX, pad_s=0.1, sigma=8.0)
    assert gates_c == [], zmax_c


def test_env_switch(monkeypatch):
    from adapt.localize import excess_sigma, localizer_name

    monkeypatch.delenv("ADAPT_LOCALIZER", raising=False)
    assert localizer_name() == "cnn"
    monkeypatch.setenv("ADAPT_LOCALIZER", "excess")
    monkeypatch.setenv("ADAPT_EXCESS_SIGMA", "10")
    assert localizer_name() == "excess" and excess_sigma() == 10.0
