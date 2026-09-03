"""Unit tests for detect-and-gate glitch excision."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples", REPO / "DINGO-BNS" / "dingo"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_tukey_gate_zeros_center():
    from adapt.glitch_excision import GateWindow, apply_gates_td, tukey_gate_weights

    fs = 4096.0
    n = int(4.0 * fs)
    t0 = 0.0
    mid = 2.0
    windows = [GateWindow("H1", mid - 0.4, mid + 0.4)]
    w = tukey_gate_weights(n, fs, windows, t0=t0, alpha=0.5)
    i_mid = int(mid * fs)
    assert w[i_mid] < 0.05
    assert w[0] > 0.99 and w[-1] > 0.99
    x = np.ones(n)
    y = apply_gates_td(x, fs, windows, t0=t0)
    assert abs(y[i_mid]) < 0.05
    assert abs(y[10] - 1.0) < 1e-9


def test_mask_to_windows_merge():
    from adapt.glitch_excision import mask_to_windows

    fs = 100.0
    mask = np.zeros(1000, dtype=bool)
    mask[100:150] = True
    mask[160:200] = True  # within merge gap of 0.05s = 5 samples → merge
    wins = mask_to_windows(mask, sample_rate=fs, t0=0.0, merge_gap_s=0.15)
    assert len(wins) == 1
    assert wins[0].t_start <= 1.0
    assert wins[0].t_end >= 2.0


def test_time_bin_mask_expands_to_gate_half():
    from adapt.glitch_excision import time_bin_mask_to_windows

    fs = 4096.0
    n = int(4.0 * fs)
    n_time = 32
    mask = np.zeros(n_time, dtype=bool)
    mask[8] = True  # one bin near early crop
    wins = time_bin_mask_to_windows(
        mask, n_samples=n, sample_rate=fs, t0=0.0, pad_s=0.4
    )
    assert len(wins) >= 1
    assert wins[0].duration >= 0.7  # ~0.8s ± merge


def test_empty_gates_noop_bit_exact():
    from adapt.glitch_excision import apply_excision_to_event_data

    n_freq = 64
    data = {
        "waveform": {
            "H1": (np.arange(n_freq) + 1j * np.arange(n_freq)).astype(np.complex128),
            "L1": np.ones(n_freq, dtype=np.complex128),
        },
        "asds": {
            "H1": np.ones(n_freq) * 1e-23,
            "L1": np.ones(n_freq) * 1e-23,
        },
    }
    res = apply_excision_to_event_data(
        data,
        td_by_det={"H1": np.zeros(128)},
        gates=[],
        sample_rate=4096.0,
        f_max=100.0,
        original_asds=data["asds"],
    )
    assert res.noop
    np.testing.assert_array_equal(res.data["waveform"]["H1"], data["waveform"]["H1"])


def test_rebuild_from_gated_td_replaces_and_restores_asd():
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td, td_to_fd_strain

    fs = 4096.0
    duration = 2.0
    n = int(duration * fs)
    n_freq = n // 2 + 1
    rng = np.random.default_rng(1)
    td = rng.normal(size=n) * 1e-22
    mid = n // 2
    td[mid - 20 : mid + 20] += 5e-20  # loud glitch
    orig_asd = np.linspace(1e-23, 2e-23, n_freq)
    welch = np.ones(n_freq) * 9e-23
    # Mismatched packaged FD (not equal to FFT(td)) — as in real inject path.
    packaged = (rng.normal(size=n_freq) + 1j * rng.normal(size=n_freq)) * 1e-22
    data = {
        "waveform": {
            "H1": packaged.copy(),
            "L1": np.ones(n_freq, dtype=np.complex128),
        },
        "asds": {"H1": welch.copy(), "L1": np.ones(n_freq) * 1e-23},
    }
    gates = [GateWindow("H1", mid / fs - 0.1, mid / fs + 0.1)]
    res = rebuild_event_from_gated_td(
        data,
        td_by_det={"H1": td},
        gates=gates,
        sample_rate=fs,
        f_max=fs / 2,
        original_asds={"H1": orig_asd, "L1": data["asds"]["L1"]},
    )
    assert "H1" in res.modified_detectors
    assert "L1" not in res.modified_detectors
    assert res.meta["rebuild_mode"] == "matched_delta"
    # Matched delta changes FD and restores ASD; L1 untouched.
    rel = float(
        np.linalg.norm(res.data["waveform"]["H1"] - packaged)
        / (np.linalg.norm(packaged) + 1e-30)
    )
    assert rel > 1e-6
    np.testing.assert_array_equal(res.data["waveform"]["L1"], data["waveform"]["L1"])
    np.testing.assert_allclose(res.data["asds"]["H1"], orig_asd)
    assert res.meta["asd_policy"] == "original"
    assert float(res.meta["residual_power_frac"]["H1"]) > 0.01

    # Full-replace ablation differs from matched delta.
    rep = rebuild_event_from_gated_td(
        data,
        td_by_det={"H1": td},
        gates=gates,
        sample_rate=fs,
        f_max=fs / 2,
        original_asds={"H1": orig_asd},
        mode="replace",
    )
    assert rep.meta["rebuild_mode"] == "replace"
    rel_rep = float(
        np.linalg.norm(rep.data["waveform"]["H1"] - res.data["waveform"]["H1"])
        / (np.linalg.norm(res.data["waveform"]["H1"]) + 1e-30)
    )
    assert rel_rep > 0.1
    _ = td_to_fd_strain(td, fs, f_max=fs / 2, n_freq=n_freq)


def test_packaged_base_rebuild_redirect():
    from adapt.glitch_excision import GateWindow, apply_excision_to_event_data

    fs = 512.0
    n = int(1.0 * fs)
    n_freq = n // 2 + 1
    td = np.zeros(n)
    td[n // 2] = 1e-20
    packaged = np.ones(n_freq, dtype=np.complex128) * (1 + 1j) * 1e-22
    data = {
        "waveform": {"H1": packaged.copy()},
        "asds": {"H1": np.ones(n_freq)},
    }
    res = apply_excision_to_event_data(
        data,
        td_by_det={"H1": td},
        gates=[GateWindow("H1", 0.4, 0.6)],
        sample_rate=fs,
        f_max=fs / 2,
        original_asds={"H1": np.full(n_freq, 3.0)},
        packaged_base="rebuild",
    )
    assert res.meta["packaged_base"] == "rebuild_from_gated_td"
    assert res.meta["rebuild_mode"] == "matched_delta"
    np.testing.assert_array_equal(res.data["asds"]["H1"], np.full(n_freq, 3.0))


def test_glitchy_inject_object_gate_restores_asd_unit():
    """Regression without DINGO: inject-like package + matched delta restores ASD."""
    from adapt.glitch_excision import (
        GateWindow,
        apply_gates_td,
        rebuild_event_from_gated_td,
        td_to_fd_strain,
    )
    from adapt.stft_context import sine_gaussian_glitch

    fs = 4096.0
    duration = 4.0
    n = int(duration * fs)
    n_freq = 256
    rng = np.random.default_rng(2)
    td_clean = rng.normal(size=n) * 1e-22
    t_peak = 2.0
    glitch = sine_gaussian_glitch(
        n, fs, t_peak=t_peak, f0=100.0, q=5.0, amplitude=8e-22
    )
    td_g = td_clean + glitch
    # Mimic inject: packaged = clean_FD + FFT(glitch), ASD = fake Welch.
    clean_fd = (rng.normal(size=n_freq) + 1j * rng.normal(size=n_freq)) * 1e-23
    g_fd = td_to_fd_strain(glitch, fs, f_max=fs / 2, n_freq=n_freq)
    packaged = clean_fd + g_fd
    orig_asd = np.linspace(1e-23, 2e-23, n_freq)
    data = {
        "waveform": {"H1": packaged.copy()},
        "asds": {"H1": np.full(n_freq, 5e-22)},
    }
    gates = [GateWindow("H1", t_peak - 0.4, t_peak + 0.4)]
    res = rebuild_event_from_gated_td(
        data,
        td_by_det={"H1": td_g},
        gates=gates,
        sample_rate=fs,
        f_max=fs / 2,
        original_asds={"H1": orig_asd},
    )
    gated_td = apply_gates_td(td_g, fs, gates, t0=0.0)
    fd_gated = td_to_fd_strain(gated_td, fs, f_max=fs / 2, n_freq=n_freq)
    fd_ungated = td_to_fd_strain(td_g, fs, f_max=fs / 2, n_freq=n_freq)
    expect = packaged + (fd_gated - fd_ungated)
    np.testing.assert_allclose(res.data["waveform"]["H1"], expect)
    np.testing.assert_allclose(res.data["asds"]["H1"], orig_asd)
    assert float(res.meta["residual_power_frac"]["H1"]) > 0.05



def test_repackaging_changes_gated_detector_only():
    from adapt.glitch_excision import GateWindow, apply_excision_to_event_data

    fs = 4096.0
    duration = 2.0
    n = int(duration * fs)
    n_freq = n // 2 + 1
    rng = np.random.default_rng(0)
    td = rng.normal(size=n) * 1e-22
    # Put a spike in the middle.
    mid = n // 2
    td[mid - 10 : mid + 10] += 1e-20
    data = {
        "waveform": {
            "H1": np.zeros(n_freq, dtype=np.complex128),
            "L1": np.zeros(n_freq, dtype=np.complex128),
        },
        "asds": {
            "H1": np.ones(n_freq) * 1e-23,
            "L1": np.ones(n_freq) * 1e-23,
        },
    }
    gates = [GateWindow("H1", mid / fs - 0.05, mid / fs + 0.05)]
    res = apply_excision_to_event_data(
        data,
        td_by_det={"H1": td, "L1": td.copy()},
        gates=gates,
        sample_rate=fs,
        f_max=fs / 2,
        original_asds=data["asds"],
        packaged_base="glitchy",
    )
    assert "H1" in res.modified_detectors
    assert "L1" not in res.modified_detectors
    assert np.linalg.norm(res.data["waveform"]["H1"]) > 1e-30
    np.testing.assert_array_equal(res.data["waveform"]["L1"], data["waveform"]["L1"])


def test_support_mask_to_time_bins():
    from adapt.glitch_excision import (
        glitch_support_mask_on_crop,
        support_mask_to_time_bins,
    )

    fs = 4096.0
    n = int(4.0 * fs)
    sm = glitch_support_mask_on_crop(
        n_crop=n,
        sample_rate=fs,
        t_rel=-1.0,
        half_width_s=0.4,
        family="sine_gaussian",
        params={"f0": 100.0, "q": 5.0},
    )
    assert sm.any()
    bins = support_mask_to_time_bins(sm, 32)
    assert bins.shape == (32,)
    assert bins.sum() >= 1


def test_detector_init_silent_and_shapes():
    from adapt.models import GlitchDetectorSTFT

    m = GlitchDetectorSTFT(in_channels=6, n_ifo=3, n_time=32, n_freq=128)
    x = torch.randn(2, 6, 32, 128)
    logits = m(x)
    assert logits.shape == (2, 3, 32)
    probs = torch.sigmoid(logits)
    assert float(probs.mean().detach()) < 0.05  # bias −4 → silent


def test_detector_predict_mask():
    from adapt.models import GlitchDetectorSTFT

    m = GlitchDetectorSTFT(in_channels=6, n_ifo=3, n_time=16, n_freq=64)
    # Force high logits by setting bias.
    with torch.no_grad():
        m.head.bias.fill_(5.0)
    x = torch.zeros(1, 6, 16, 64)
    mask = m.predict_mask(x, threshold=0.5)
    assert mask.shape == (1, 3, 16)
    assert bool(mask.all())
