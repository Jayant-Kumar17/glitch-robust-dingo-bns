#!/usr/bin/env python3
"""Shared GW170817 strain I/O and glitch-injection helpers for the examples.

No network retraining paths live here — only TD/FD packaging utilities used by
the frozen-DINGO-BNS detect → gate → matched-delta experiments.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

logger = logging.getLogger("event_glitch_io")


def sine_gaussian_glitch(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    f0: float = 100.0,
    q: float = 5.0,
    amplitude: float,
) -> np.ndarray:
    """Sine-Gaussian blip: ``A * exp(-(t-t0)^2/(2σ^2)) * cos(2π f0 (t-t0))``."""
    from adapt.stft_context import sine_gaussian_glitch as _sg

    return _sg(
        n_samples,
        sample_rate,
        t_peak=t_peak,
        f0=f0,
        q=q,
        amplitude=amplitude,
    )


def load_full_event_td(
    assets: Dict[str, Any],
    settings: Dict[str, Any],
    det: str = "H1",
) -> Tuple[np.ndarray, float, float]:
    """Load full analysis-segment TD for ``det`` (length ``T`` at ``f_s``)."""
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    f_s = float(settings.get("f_s", 4096.0))
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    start = trigger - (duration - time_buffer)
    end = trigger + time_buffer
    path = assets["strain_gwf"][det]
    channel = assets["channels"][det]
    logger.info("Loading full TD %s [%s, %s] from %s", det, start, end, path.name)
    ts = TimeSeries.read(str(path), channel=channel)
    ts = ts.crop(start, end)
    if float(ts.sample_rate.value) != f_s:
        ts = ts.resample(f_s)
    td = np.asarray(ts.value, dtype=np.float64)
    n_expected = int(round(duration * f_s))
    if td.size > n_expected:
        td = td[:n_expected]
    elif td.size < n_expected:
        td = np.pad(td, (0, n_expected - td.size))
    return td, f_s, time_buffer


def load_event_td_crops(
    assets: Dict[str, Any],
    *,
    sample_rate: float,
    crop_seconds: float,
) -> Dict[str, np.ndarray]:
    """Load centered TD crops matching the shared STFT analysis window."""
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    half = 0.5 * float(crop_seconds)
    t0 = trigger - half
    t1 = trigger + half
    out: Dict[str, np.ndarray] = {}
    for det, path in assets["strain_gwf"].items():
        channel = assets["channels"][det]
        logger.info("Loading TD %s from %s [%s]", det, path.name, channel)
        ts = TimeSeries.read(str(path), channel=channel)
        ts = ts.crop(t0, t1)
        if float(ts.sample_rate.value) != float(sample_rate):
            ts = ts.resample(sample_rate)
        out[det] = np.asarray(ts.value, dtype=np.float64)
    return out


def td_to_fd_strain(
    td: np.ndarray,
    sample_rate: float,
    *,
    roll_off: float = 0.4,
    f_max: float,
) -> np.ndarray:
    """Tukey-windowed FFT matching demo conditioning."""
    x = np.asarray(td, dtype=np.float64).ravel()
    alpha = 2.0 * float(roll_off) * float(sample_rate) / max(len(x), 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    window = sp_signal.windows.tukey(len(x), alpha=alpha)
    xw = x * window
    fd = np.fft.rfft(xw) / float(sample_rate)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate))
    n_keep = int(np.floor(float(f_max) / (freqs[1] if len(freqs) > 1 else 1.0))) + 1
    n_keep = min(n_keep, len(fd))
    return np.asarray(fd[:n_keep], dtype=np.complex128)


def welch_asd(
    td: np.ndarray,
    sample_rate: float,
    *,
    n_freq: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Welch ASD on the analysis segment; smear transients into a 1-D spectrum."""
    nperseg = int(min(len(td), max(256, int(round(sample_rate * 4.0)))))
    freqs, psd = sp_signal.welch(
        np.asarray(td, dtype=np.float64),
        fs=float(sample_rate),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    target_f = np.linspace(0.0, float(f_max), int(n_freq), dtype=np.float64)
    psd_i = np.interp(target_f, freqs, psd, left=psd[0], right=psd[-1])
    asd = np.sqrt(np.maximum(psd_i, 0.0))
    asd[target_f < float(f_min)] = 1.0
    asd = np.maximum(asd, 1e-30)
    return asd.astype(np.float64)


def inject_h1_glitch_into_event(
    event,
    assets: Dict[str, Any],
    *,
    f0: float = 100.0,
    q: float = 5.0,
    snr_amp_scale: float = 8.0,
    t_rel: float = -1.0,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Return glitchy event ``data`` dict, TD map for STFT, and glitch metadata."""
    settings = dict(event.settings)
    td_h1, f_s, time_buffer = load_full_event_td(assets, settings, "H1")
    duration = float(settings.get("T", 128.0))
    t_peak = (duration - time_buffer) + float(t_rel)
    from adapt.stft_context import inband_rms

    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    rms_broadband = float(np.std(td_h1)) or 1e-22
    rms = inband_rms(td_h1, f_s, f_min=f_min, f_max=f_max)
    amp = snr_amp_scale * rms
    glitch = sine_gaussian_glitch(
        len(td_h1), f_s, t_peak=t_peak, f0=f0, q=q, amplitude=amp
    )
    td_h1_g = td_h1 + glitch

    data = copy.deepcopy(event.data)
    roll_off = float(settings.get("roll_off", 0.4))
    n_freq = len(data["waveform"]["H1"])

    glitch_fd = td_to_fd_strain(glitch, f_s, roll_off=roll_off, f_max=f_max)
    if len(glitch_fd) < n_freq:
        glitch_fd = np.pad(glitch_fd, (0, n_freq - len(glitch_fd)))
    else:
        glitch_fd = glitch_fd[:n_freq]
    data["waveform"]["H1"] = (
        np.asarray(data["waveform"]["H1"], dtype=np.complex128) + glitch_fd
    )
    data["asds"]["H1"] = welch_asd(
        td_h1_g, f_s, n_freq=n_freq, f_min=f_min, f_max=f_max
    )

    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * f_s))
    trig_idx = int(round((duration - time_buffer) * f_s))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    if end > len(td_h1_g):
        end = len(td_h1_g)
        start = end - n_crop
    td_stft = {"H1": td_h1_g[start:end].copy()}
    clean = load_event_td_crops(
        assets, sample_rate=f_s, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
    )
    td_stft["L1"] = clean["L1"]
    td_stft["V1"] = clean["V1"]

    meta = {
        "det": "H1",
        "f0": f0,
        "q": q,
        "t_rel": t_rel,
        "amplitude": amp,
        "rms": rms,
        "rms_inband": rms,
        "rms_broadband": rms_broadband,
        "rms_basis": "inband",
        "t_peak_in_segment": t_peak,
        "td_full": {"H1": td_h1_g},
        "td_clean_full": {"H1": td_h1},
        "sample_rate": float(f_s),
        "duration": float(duration),
        "time_buffer": float(time_buffer),
        "roll_off": float(roll_off),
        "f_min": float(f_min),
        "f_max": float(f_max),
        "asd_policy": "welch",
    }
    logger.info(
        "Injected H1 sine-Gaussian: f0=%.1f Hz Q=%.1f amp=%.3e "
        "(%.1fx in-band RMS; broadband RMS=%.3e) t_rel=%.2fs",
        f0,
        q,
        amp,
        snr_amp_scale,
        rms_broadband,
        t_rel,
    )
    return data, td_stft, meta


def glitch_meta_for_json(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large TD arrays before writing reports."""
    skip = {"td_full", "td_clean_full"}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in skip:
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out
