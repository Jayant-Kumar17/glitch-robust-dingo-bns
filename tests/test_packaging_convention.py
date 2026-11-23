"""The injection, the gate rebuild and the archived package must share one time frame.

``package_fd`` has to equal upstream DINGO ``data_to_domain`` (Tukey window,
rfft / f_s, ``cyclic_time_shift(time_buffer)``). If these tests fail, do NOT
relax them: the convention is wrong and every recovery number is invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal.windows import tukey

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples", REPO / "DINGO-BNS" / "dingo"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FS = 4096.0
T = 16.0
TB = 2.0
N = int(FS * T)


def _toy(seed: int = 0):
    from adapt.stft_context import sine_gaussian_glitch

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    t_peak = (T - TB) - 1.0  # 1 s before the trigger
    g = sine_gaussian_glitch(N, FS, t_peak=t_peak, f0=100.0, q=5.0, amplitude=300.0)
    return x, g, t_peak


def _to_segment_time(fd: np.ndarray) -> np.ndarray:
    """Undo the cyclic shift and return the (windowed) time series the network sees."""
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    return np.fft.irfft(fd * np.exp(+2j * np.pi * f * TB) * FS, n=N)


def test_package_fd_matches_pycbc_cyclic_time_shift():
    """Same operations, same order, as dingo.gw.data.data_preparation.data_to_domain."""
    pycbc_types = pytest.importorskip("pycbc.types")
    from adapt.packaging import package_fd

    x, _, _ = _toy()
    window = tukey(N, alpha=2 * 0.4 / T)
    ts = pycbc_types.TimeSeries(x, delta_t=1.0 / FS)
    ref = np.array((ts * window).to_frequencyseries().cyclic_time_shift(TB))
    got = package_fd(x, FS, roll_off=0.4, f_max=FS / 2, time_buffer=TB)
    assert got.shape == ref.shape
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-10, rel


def test_both_td_to_fd_wrappers_use_the_shift():
    from adapt.event_glitch_io import td_to_fd_strain as inject_fd
    from adapt.glitch_excision import td_to_fd_strain as rebuild_fd
    from adapt.packaging import package_fd

    x, _, _ = _toy()
    ref = package_fd(x, FS, f_max=FS / 2, time_buffer=TB)
    np.testing.assert_allclose(inject_fd(x, FS, f_max=FS / 2), ref)
    np.testing.assert_allclose(rebuild_fd(x, FS, f_max=FS / 2), ref)
    unshifted = package_fd(x, FS, f_max=FS / 2, time_buffer=0.0)
    assert np.linalg.norm(ref - unshifted) / np.linalg.norm(ref) > 1.0


def test_injected_glitch_lands_at_its_nominal_time():
    from adapt.event_glitch_io import td_to_fd_strain

    x, g, t_peak = _toy()
    clean = td_to_fd_strain(x, FS, f_max=FS / 2)
    poison = clean + td_to_fd_strain(g, FS, f_max=FS / 2)
    diff = _to_segment_time(poison) - _to_segment_time(clean)
    t_seen = np.argmax(np.abs(diff)) / FS
    assert abs(t_seen - t_peak) < 0.01, (t_seen, t_peak)


def test_matched_delta_equals_official_repackage_of_gated_strain():
    from adapt.glitch_excision import (
        GateWindow,
        apply_gates_td,
        rebuild_event_from_gated_td,
        td_to_fd_strain,
    )

    x, g, t_peak = _toy()
    n_freq = N // 2 + 1
    clean = td_to_fd_strain(x, FS, f_max=FS / 2)
    poison = {
        "waveform": {"H1": clean + td_to_fd_strain(g, FS, f_max=FS / 2)},
        "asds": {"H1": np.ones(n_freq)},
    }
    gates = [GateWindow("H1", t_peak - 0.4, t_peak + 0.4)]
    res = rebuild_event_from_gated_td(
        poison,
        td_by_det={"H1": x + g},
        gates=gates,
        sample_rate=FS,
        f_max=FS / 2,
        fill="zero",
    )
    honest = td_to_fd_strain(apply_gates_td(x + g, FS, gates), FS, f_max=FS / 2)
    rel = np.linalg.norm(res.data["waveform"]["H1"] - honest) / np.linalg.norm(honest)
    assert rel < 1e-10, rel


def test_gate_is_a_real_hole_in_the_network_frame():
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td, td_to_fd_strain

    x, g, t_peak = _toy()
    n_freq = N // 2 + 1
    clean = td_to_fd_strain(x, FS, f_max=FS / 2)
    poison = {
        "waveform": {"H1": clean + td_to_fd_strain(g, FS, f_max=FS / 2)},
        "asds": {"H1": np.ones(n_freq)},
    }
    gates = [GateWindow("H1", t_peak - 0.4, t_peak + 0.4)]
    res = rebuild_event_from_gated_td(
        poison, td_by_det={"H1": x + g}, gates=gates, sample_rate=FS, f_max=FS / 2, fill="zero"
    )
    y = _to_segment_time(res.data["waveform"]["H1"])
    core = slice(int((t_peak - 0.15) * FS), int((t_peak + 0.15) * FS))  # fully zeroed part
    away = slice(int((t_peak - 2.15) * FS), int((t_peak - 1.85) * FS))  # 2 s earlier, untouched
    rms = lambda s: float(np.sqrt(np.mean(y[s] ** 2)))  # noqa: E731
    assert rms(core) < 1e-6, rms(core)
    assert 0.9 < rms(away) < 1.1, rms(away)


def test_noise_fill_has_the_requested_spectrum_and_removes_the_glitch():
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td, td_to_fd_strain
    from adapt.packaging import coloured_noise_td

    # Spectrum: band-limited ASD with a DINGO-style placeholder (1.0) below f_min.
    n_freq = N // 2 + 1
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    asd = np.full(n_freq, 2e-23)
    asd[f < 23.0] = 1.0
    noise = coloured_noise_td(asd, N, FS, seed=3)
    spec = np.fft.rfft(noise)
    psd = 2.0 * np.abs(spec) ** 2 / (N * FS)
    band = (f >= 23.0) & (f <= FS / 2 - 1)
    assert abs(np.mean(psd[band]) / (2e-23) ** 2 - 1.0) < 0.05
    assert np.all(psd[f < 22.9] < 1e-60)

    # Fill: strain-scale white data with a matching ASD -> variance stays flat, glitch gone.
    scale = 1e-21
    x, g, t_peak = _toy(seed=5)
    x, g = scale * x, scale * g
    asd1 = np.full(n_freq, scale * np.sqrt(2.0 / FS))  # white, std = scale per sample
    clean = td_to_fd_strain(x, FS, f_max=FS / 2)
    poison = {
        "waveform": {"H1": clean + td_to_fd_strain(g, FS, f_max=FS / 2)},
        "asds": {"H1": asd1.copy()},
    }
    gates = [GateWindow("H1", t_peak - 0.4, t_peak + 0.4)]
    res = rebuild_event_from_gated_td(
        poison,
        td_by_det={"H1": x + g},
        gates=gates,
        sample_rate=FS,
        f_max=FS / 2,
        original_asds={"H1": asd1},
        fill="noise",
    )
    assert res.meta["fill"] == "noise"
    y = _to_segment_time(res.data["waveform"]["H1"]) / scale
    core = slice(int((t_peak - 0.15) * FS), int((t_peak + 0.15) * FS))
    taper = slice(int((t_peak - 0.4) * FS), int((t_peak - 0.2) * FS))
    rms = lambda s: float(np.sqrt(np.mean(y[s] ** 2)))  # noqa: E731
    assert 0.9 < rms(core) < 1.1, rms(core)
    assert 0.85 < rms(taper) < 1.15, rms(taper)
    assert np.max(np.abs(y[core])) < 6.0  # the 300-sigma glitch is gone


def _assets_available() -> bool:
    try:
        from adapt.dingo_bns_demo import discover_assets

        assets = discover_assets()
        return Path(assets["baseline_ckpt"]).is_file() and Path(assets["event_hdf5"]).is_file()
    except Exception:
        return False


@pytest.mark.skipif(not _assets_available(), reason="GW170817/DINGO assets missing")
def test_package_fd_reproduces_the_archived_gw170817_package():
    """The decisive check: our transform of the local strain equals the archive."""
    from adapt.dingo_bns_demo import discover_assets, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from adapt.packaging import package_fd

    assets = discover_assets()
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    report = {}
    for det in ("H1", "L1", "V1"):
        td, f_s, time_buffer = load_full_event_td(assets, settings, det)
        archived = np.asarray(event.data["waveform"][det], dtype=np.complex128)
        freqs = np.arange(archived.size) * (f_s / td.size)
        band = (freqs >= f_min) & (freqs <= f_max)
        rel = {}
        for label, tb in (("shifted", time_buffer), ("unshifted", 0.0)):
            ours = package_fd(td, f_s, roll_off=0.4, f_max=f_max, n_freq=archived.size, time_buffer=tb)
            rel[label] = float(np.linalg.norm((ours - archived)[band]) / np.linalg.norm(archived[band]))
        report[det] = rel
    print("relative L2 to archived package:", report)
    for det, rel in report.items():
        assert rel["shifted"] < 1e-2, (det, rel)
        assert rel["unshifted"] > 0.5, (det, rel)


def test_band_limited_gate_does_not_leak_the_seismic_wall():
    """Gating raw strain leaks low-frequency power into the band; gate_band must prevent it."""
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td, td_to_fd_strain

    n_freq = N // 2 + 1
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    rng = np.random.default_rng(11)
    # White in-band noise (unit variance) plus a wall 1e4 times louder below 10 Hz.
    wall = np.where((f > 0.5) & (f < 10.0), 1e4, 0.0)
    spec = (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)) * (1.0 + wall)
    x = np.fft.irfft(spec * np.sqrt(N) / np.sqrt(2.0), n=N)
    clean = td_to_fd_strain(x, FS, f_max=FS / 2)
    data = {"waveform": {"H1": clean.copy()}, "asds": {"H1": np.ones(n_freq)}}
    t_c = (T - TB) - 1.0
    gates = [GateWindow("H1", t_c - 0.1, t_c + 0.1)]
    band = (f >= 23.0) & (f <= 1500.0)

    def inband_change(**kw):
        res = rebuild_event_from_gated_td(
            data, td_by_det={"H1": x}, gates=gates, sample_rate=FS, f_max=FS / 2, fill="zero", **kw
        )
        return float(np.linalg.norm((res.data["waveform"]["H1"] - clean)[band]))

    raw = inband_change()
    limited = inband_change(gate_band=(20.0, 23.0))
    assert raw > 10.0 * limited, (raw, limited)

    # With the band limit, the in-band data inside the gate core are gone.
    res = rebuild_event_from_gated_td(
        data, td_by_det={"H1": x}, gates=gates, sample_rate=FS, f_max=FS / 2,
        fill="zero", gate_band=(20.0, 23.0),
    )
    assert res.meta["gate_band_hz"] == [20.0, 23.0]
    spec_out = res.data["waveform"]["H1"].copy()
    spec_out[f < 30.0] = 0.0
    spec_in = clean.copy()
    spec_in[f < 30.0] = 0.0
    y_out, y_in = _to_segment_time(spec_out), _to_segment_time(spec_in)
    core = slice(int((t_c - 0.03) * FS), int((t_c + 0.03) * FS))
    rms = lambda y: float(np.sqrt(np.mean(y[core] ** 2)))  # noqa: E731
    assert rms(y_out) < 0.1 * rms(y_in), (rms(y_out), rms(y_in))
