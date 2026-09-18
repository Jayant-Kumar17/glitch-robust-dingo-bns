"""Tests for the Gravity Spy catalogue filter and excerpt scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SELECTED = REPO / "data" / "gravity_spy" / "selected_h1_o3.csv"


def test_cached_selection_has_at_least_20_rows():
    from adapt.gravity_spy_io import SelectionCriteria, load_selected_catalogue, select_glitches

    if not SELECTED.is_file():
        pytest.skip("cached selection CSV not present")
    df = load_selected_catalogue(SELECTED)
    assert len(df) >= 20
    # Re-applying the filter to the cached subset must be idempotent.
    again = select_glitches(df, SelectionCriteria(), min_separation_s=0.0)
    assert len(again) >= 20
    assert set(again["ml_label"]) <= {"Blip", "Scattered_Light", "Whistle", "Koi_Fish"}
    assert (again["ml_confidence"] >= 0.95).all()
    assert ((again["snr"] >= 8.0) & (again["snr"] <= 30.0)).all()
    assert (again["duration"] <= 3.0).all()


def test_normalise_columns_accepts_aliases():
    from adapt.gravity_spy_io import normalise_columns

    df = pd.DataFrame(
        {
            "peak_time": [1.0],
            "peak_time_ns": [0],
            "IFO": ["H1"],
            "Duration": [0.2],
            "SNR": [12.0],
            "label": ["Blip"],
            "confidence": [0.99],
        }
    )
    out = normalise_columns(df)
    for c in ("event_time", "ifo", "duration", "snr", "ml_label", "ml_confidence"):
        assert c in out.columns


def test_normalise_columns_raises_on_missing():
    from adapt.gravity_spy_io import normalise_columns

    with pytest.raises(KeyError):
        normalise_columns(pd.DataFrame({"foo": [1]}))


def _coloured_noise(rng, n, fs):
    """Red-ish Gaussian noise (1/f^2 above 20 Hz) mimicking a seismic wall."""
    white = rng.normal(size=n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.where(f > 0, (20.0 / np.maximum(f, 20.0)) ** 2 + 0.01, 0.0)
    return np.fft.irfft(X * shape, n=n) * 1e-21


def test_whitened_excerpt_recovers_injected_snr():
    from adapt.gravity_spy_io import extract_glitch_excerpt

    fs = 4096.0
    rng = np.random.default_rng(0)
    n = int(8 * fs)
    t = np.arange(n) / fs
    noise = _coloured_noise(rng, n, fs)
    # Sine-Gaussian at 150 Hz, chosen to have whitened SNR ~ 25 in this noise.
    sg = np.exp(-((t - 4.0) ** 2) / (2 * 0.01**2)) * np.cos(2 * np.pi * 150 * (t - 4.0))
    exc0 = extract_glitch_excerpt(noise, sample_rate=fs, duration_s=0.1)
    amp = 25.0 / np.sqrt(np.sum((sg[int(3.5 * fs):int(4.5 * fs)]) ** 2)) * float(np.std(noise[int(3.5 * fs):int(4.5 * fs)]))
    exc = extract_glitch_excerpt(noise + amp * sg * 30, sample_rate=fs, duration_s=0.1)
    assert exc.window_s == pytest.approx(1.0)  # max(0.1, 0.5) * 2
    assert exc.background_rms_w == pytest.approx(1.0, rel=0.1)
    # Glitch cell: clear excess and a non-trivial denoised estimate.
    assert exc.excess_energy_w > 100.0
    assert exc.denoised_energy_w > 0.5 * exc.excess_energy_w
    # Pure-noise window: excess is small relative to the window length and the
    # thresholded estimate carries little energy.
    n_win = exc0.waveform.size
    assert abs(exc0.excess_energy_w) < 0.2 * n_win
    assert exc0.denoised_energy_w < 0.1 * n_win


def test_place_series_and_colour_shapes():
    from adapt.gravity_spy_io import colour_with_asd, place_series

    fs = 4096.0
    w = np.ones(int(fs))
    asd = np.full(2000, 1e-23)
    g = colour_with_asd(w, fs, asd=asd, delta_f=0.5, f_min=23.0, f_max=900.0)
    assert g.shape == w.shape
    s = place_series(g, n_samples=int(4 * fs), sample_rate=fs, t_peak=2.0)
    assert s.shape[0] == int(4 * fs)
    assert np.allclose(s[: int(1.4 * fs)], 0.0)
