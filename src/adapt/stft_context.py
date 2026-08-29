"""Shared STFT / coherence spectrogram context builders for BNS models.

Legacy path: 3-channel H1–L1 mag/coherence ``(3, 5, 128)``.
Robust path: 6-channel H1/L1/V1 mag + HL/HV/LV cross mag ``(6, 32, 128)``
with fixed robust channel statistics (not per-sample Z-score).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from adapt.spectrogram_geometry import (
    SPECTROGRAM_ANALYSIS_SECONDS,
    SPECTROGRAM_FREQ_BINS,
    SPECTROGRAM_TIME_STEPS,
    compute_complex_stft_grid,
    compute_stft_spectrogram_grid,
)

LOG_FLOOR = 1e-60
UNIT_SCALE_EPS = 1e-8
ENERGY_EPS = 1e-8
# Floor for ASD interpolation / clamping away from exact zero. Must NOT be used
# as a divisor when the ASD itself is ~0 (design ASD top-bin zeros): that turns
# tiny TD↔FD round-trip leakage into 1e16 whitened spikes and destroys STFT.
WHITEN_EPS = 1e-40
# Relative/absolute mask for "this ASD bin carries no sensitivity".
ASD_ZERO_REL = 1e-12
ASD_ZERO_ABS = 1e-30

# Default STFT geometry used by training CLI (resampled onto 5 x 128).
DEFAULT_N_FFT = 2048
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 256

CSD_CHANNELS = 3
SPECTROGRAM_LAYOUT = "hl_coh_3ch"

# Robust glitch-correction geometry.
ROBUST_N_TIME = 32
ROBUST_N_FREQ = 128
ROBUST_CHANNELS = 6
ROBUST_LAYOUT = "hlv_coh_6ch"
ROBUST_CHANNEL_NAMES = ("H1", "L1", "V1", "HL", "HV", "LV")


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
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    tau = q / (2.0 * np.pi * float(f0))
    envelope = np.exp(-((t - float(t_peak)) ** 2) / (2.0 * tau**2))
    return float(amplitude) * envelope * np.cos(
        2.0 * np.pi * float(f0) * (t - float(t_peak))
    )


def inject_random_glitch(
    td: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator,
    *,
    f0_range: tuple[float, float] = (30.0, 300.0),
    q_range: tuple[float, float] = (3.0, 15.0),
    t_rel_range: tuple[float, float] = (-2.0, 0.5),
    amp_scale_range: tuple[float, float] = (2.0, 10.0),
) -> np.ndarray:
    """Add a randomized sine-Gaussian glitch onto a TD crop (centered trigger)."""
    x = np.asarray(td, dtype=np.float64).ravel().copy()
    n = x.size
    if n < 8:
        return x
    duration = n / float(sample_rate)
    # Crop is centered on trigger → trigger at mid-point.
    t_rel = float(rng.uniform(*t_rel_range))
    t_peak = 0.5 * duration + t_rel
    t_peak = float(np.clip(t_peak, 0.0, max(duration - 1.0 / sample_rate, 0.0)))
    f0 = float(rng.uniform(*f0_range))
    q = float(rng.uniform(*q_range))
    rms = float(np.std(x)) or 1e-22
    amp = float(rng.uniform(*amp_scale_range)) * rms
    return x + sine_gaussian_glitch(
        n, sample_rate, t_peak=t_peak, f0=f0, q=q, amplitude=amp
    )


def crop_td_to_analysis_window(
    td: np.ndarray,
    sample_rate: float,
    *,
    analysis_seconds: float = SPECTROGRAM_ANALYSIS_SECONDS,
) -> np.ndarray:
    """Center-crop or pad ``td`` to exactly ``analysis_seconds * sample_rate``."""
    x = np.asarray(td, dtype=np.float64).ravel()
    n_crop = int(round(float(analysis_seconds) * float(sample_rate)))
    if n_crop < 1:
        raise ValueError(f"analysis window length must be >= 1; got {n_crop}")
    if x.size > n_crop:
        start = (x.size - n_crop) // 2
        x = x[start : start + n_crop]
    elif x.size < n_crop:
        x = np.pad(x, (0, n_crop - x.size))
    return x


def safe_whiten_fd(
    fd: np.ndarray,
    asd: np.ndarray,
    *,
    noise_std: float = 1.0,
) -> np.ndarray:
    """Whiten FD strain, zeroing bins where the ASD has no sensitivity.

    Masking near-zero ASD bins (instead of dividing by ``WHITEN_EPS``) prevents
    TD↔FD round-trip leakage at design-ASD nulls from becoming 1e16 spikes.
    """
    fd_arr = np.asarray(fd, dtype=np.complex128).ravel()
    asd_arr = np.asarray(asd, dtype=np.float64).ravel()
    if asd_arr.shape != fd_arr.shape:
        raise ValueError(f"asd shape {asd_arr.shape} != fd shape {fd_arr.shape}")
    ref = float(np.median(asd_arr[asd_arr > 0.0])) if np.any(asd_arr > 0.0) else 1.0
    dead = asd_arr <= max(float(ASD_ZERO_ABS), float(ASD_ZERO_REL) * ref)
    denom = asd_arr * float(noise_std)
    denom = np.where(dead, 1.0, denom)
    out = fd_arr / denom
    out = np.where(dead, 0.0, out)
    return out


def inband_rms(
    td: np.ndarray,
    sample_rate: float,
    *,
    f_min: float = 23.0,
    f_max: float = 1535.0,
) -> float:
    """RMS of a TD series after retaining only ``[f_min, f_max]``."""
    x = np.asarray(td, dtype=np.float64).ravel()
    if x.size < 8:
        return float(np.std(x) or 1e-22)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    mask = (freqs >= float(f_min)) & (freqs <= float(f_max))
    if not np.any(mask):
        return float(np.std(x) or 1e-22)
    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    xb = np.fft.irfft(Xf, n=x.size)
    return float(np.std(xb) or 1e-22)


def fd_waveform_to_td_crop(
    fd: np.ndarray,
    *,
    sample_rate: float,
    duration: float,
    trigger_time: float = 0.0,
    asd: Optional[np.ndarray] = None,
    noise_std: float = 1.0,
    whiten: bool = True,
    analysis_seconds: float = SPECTROGRAM_ANALYSIS_SECONDS,
) -> np.ndarray:
    """IFFT a uniform FD strain to a trigger-centered analysis-window TD crop.

    Uses the Bilby convention ``td = irfft(fd_full, n=n_td) * sample_rate``.
    When ``whiten`` is True and ``asd`` is provided, divides by
    ``asd * noise_std`` before the IFFT so STFT magnitudes track SNR / ``d_L``.
    """
    fd_arr = np.asarray(fd, dtype=np.complex128).ravel()
    if whiten:
        if asd is None:
            raise ValueError("asd is required when whiten=True")
        fd_arr = safe_whiten_fd(fd_arr, asd, noise_std=noise_std)

    n_td = int(round(float(duration) * float(sample_rate)))
    if n_td < 8:
        raise ValueError(f"n_td too small: {n_td}")
    n_rfft = n_td // 2 + 1
    fd_full = np.zeros(n_rfft, dtype=np.complex128)
    n_copy = min(fd_arr.size, n_rfft)
    fd_full[:n_copy] = fd_arr[:n_copy]
    td = np.fft.irfft(fd_full, n=n_td) * float(sample_rate)

    trig = int(round(float(trigger_time) * float(sample_rate))) % n_td
    td = np.roll(td, n_td // 2 - trig)
    return crop_td_to_analysis_window(
        td, sample_rate, analysis_seconds=analysis_seconds
    )


def injection_waveforms_to_td_map(
    injection_data: Mapping[str, Any],
    detectors: Sequence[str],
    *,
    sample_rate: float,
    duration: float,
    noise_std: float,
    whiten: bool = True,
    analysis_seconds: float = SPECTROGRAM_ANALYSIS_SECONDS,
) -> Dict[str, np.ndarray]:
    """Convert per-IFO FD injection waveforms to trigger-centered TD crops."""
    waveforms = injection_data["waveform"]
    asds = injection_data.get("asds") or {}
    params = injection_data.get("parameters") or {}
    geocent = float(params.get("geocent_time", 0.0))
    out: Dict[str, np.ndarray] = {}
    for det in detectors:
        if det not in waveforms:
            raise KeyError(f"Missing waveform for detector {det}")
        trig = float(params.get(f"{det}_time", geocent))
        asd = asds.get(det) if whiten else None
        out[det] = fd_waveform_to_td_crop(
            waveforms[det],
            sample_rate=sample_rate,
            duration=duration,
            trigger_time=trig,
            asd=asd,
            noise_std=noise_std,
            whiten=whiten,
            analysis_seconds=analysis_seconds,
        )
    return out


def whiten_td_crop_with_asd(
    td: np.ndarray,
    sample_rate: float,
    asd: np.ndarray,
    *,
    delta_f: float,
    noise_std: float,
    taper: bool = True,
    roll_off: float = 0.4,
) -> np.ndarray:
    """ASD-whiten a short TD crop so STFT log-energy matches training scale.

    Training builds STFT from whitened FD→TD injections. Event eval loads raw
    GWF crops; without this step, ``log_energy`` floors near ``log(eps)≈-18``
    and destroys energy conditioning / PE.

    A Tukey taper (default on) suppresses hard-crop spectral leakage from the
    sub-20 Hz seismic wall that otherwise inflates whitened std by ~100x and
    false-triggers the energy gate on clean real events.
    """
    from scipy import signal as sp_signal

    x = np.asarray(td, dtype=np.float64).ravel().copy()
    asd_arr = np.asarray(asd, dtype=np.float64).ravel()
    if x.size < 8:
        return x.copy()
    if taper:
        alpha = 2.0 * float(roll_off) * float(sample_rate) / max(len(x), 1)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        x = x * sp_signal.windows.tukey(len(x), alpha=alpha)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    f_asd = np.arange(asd_arr.size, dtype=np.float64) * float(delta_f)
    asd_i = np.interp(freqs, f_asd, asd_arr, left=asd_arr[0], right=asd_arr[-1])
    ref = float(np.median(asd_i[asd_i > 0.0])) if np.any(asd_i > 0.0) else 1.0
    dead = asd_i <= max(float(ASD_ZERO_ABS), float(ASD_ZERO_REL) * ref)
    denom = asd_i * float(noise_std)
    denom = np.where(dead, 1.0, np.maximum(denom, WHITEN_EPS))
    Xw = X / denom
    Xw = np.where(dead, 0.0, Xw)
    return np.fft.irfft(Xw, n=x.size)


def whiten_td_map_with_asds(
    td_by_det: Mapping[str, np.ndarray],
    asds: Mapping[str, np.ndarray],
    *,
    sample_rate: float,
    delta_f: float,
    noise_std: float,
    detectors: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Whiten each detector crop in ``td_by_det`` with the matching ASD."""
    dets = list(detectors) if detectors is not None else list(td_by_det.keys())
    out: Dict[str, np.ndarray] = {}
    for det in dets:
        if det not in td_by_det:
            raise KeyError(f"Missing TD for detector {det}")
        if det not in asds:
            raise KeyError(f"Missing ASD for detector {det}")
        out[det] = whiten_td_crop_with_asd(
            td_by_det[det],
            sample_rate,
            asds[det],
            delta_f=delta_f,
            noise_std=noise_std,
        )
    return out


def _log10_unit_scale(mag: np.ndarray) -> np.ndarray:
    """log10 + per-channel Z-score for a magnitude grid."""
    grid = np.log10(np.maximum(mag, LOG_FLOOR))
    mu = float(grid.mean())
    sig = float(grid.std())
    return ((grid - mu) / (sig + UNIT_SCALE_EPS)).astype(np.float32)


def log_unit_scale_stft_grid(
    td: np.ndarray,
    sample_rate: float,
    *,
    n_time: int = SPECTROGRAM_TIME_STEPS,
    n_freq: int = SPECTROGRAM_FREQ_BINS,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """STFT magnitude → log-energy + log10/unit-scale grid (legacy helper)."""
    x = crop_td_to_analysis_window(td, sample_rate)
    mag = compute_stft_spectrogram_grid(
        x,
        float(sample_rate),
        n_time=int(n_time),
        n_freq=int(n_freq),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    log_energy = float(np.log(float(np.sum(mag * mag)) + ENERGY_EPS))
    return _log10_unit_scale(mag), log_energy


def build_spectrogram_stack_from_td(
    td_by_det: Mapping[str, np.ndarray],
    detectors: Sequence[str],
    sample_rate: float,
    *,
    n_time: int = SPECTROGRAM_TIME_STEPS,
    n_freq: int = SPECTROGRAM_FREQ_BINS,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy per-IFO stack ``(n_ifo, 1, n_time, n_freq)`` + log-energy."""
    grids = []
    energies = []
    for det in detectors:
        if det not in td_by_det:
            raise KeyError(f"Missing TD strain for detector {det}")
        grid, e = log_unit_scale_stft_grid(
            td_by_det[det],
            sample_rate,
            n_time=n_time,
            n_freq=n_freq,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        grids.append(grid)
        energies.append(e)
    stacked = np.stack(grids, axis=0)[:, None, :, :]
    expected = (len(detectors), 1, int(n_time), int(n_freq))
    if stacked.shape != expected:
        raise RuntimeError(f"spectrogram stack shape {stacked.shape} != {expected}")
    log_energy = np.asarray(energies, dtype=np.float32)
    return stacked, log_energy


def build_csd_spectrogram_from_td(
    td_by_det: Mapping[str, np.ndarray],
    sample_rate: float,
    *,
    n_time: int = SPECTROGRAM_TIME_STEPS,
    n_freq: int = SPECTROGRAM_FREQ_BINS,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    energy_detectors: Sequence[str] = ("H1", "L1", "V1"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build H1–L1 3-channel mag/coherence tensor + 3-IFO log-energy.

    Channels (float32, shape ``(3, n_time, n_freq)``):
      0: log10-unit-scaled ``|S_H1|``
      1: log10-unit-scaled ``|S_L1|``
      2: log10-unit-scaled ``|S_H1 conj(S_L1)|``

    Log-energy is ``(len(energy_detectors),)`` from raw magnitude power before
    normalization (H1/L1 from complex STFT; V1 from magnitude STFT if present).
    """
    if "H1" not in td_by_det or "L1" not in td_by_det:
        raise KeyError("CSD spectrogram requires H1 and L1 TD strains")

    stft_kw = dict(
        n_time=int(n_time),
        n_freq=int(n_freq),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    h1 = crop_td_to_analysis_window(td_by_det["H1"], sample_rate)
    l1 = crop_td_to_analysis_window(td_by_det["L1"], sample_rate)
    s_h1 = compute_complex_stft_grid(h1, float(sample_rate), **stft_kw)
    s_l1 = compute_complex_stft_grid(l1, float(sample_rate), **stft_kw)
    s_coh = s_h1 * np.conj(s_l1)

    mag_h1 = np.abs(s_h1)
    mag_l1 = np.abs(s_l1)
    mag_coh = np.abs(s_coh)

    ch0 = _log10_unit_scale(mag_h1)
    ch1 = _log10_unit_scale(mag_l1)
    ch2 = _log10_unit_scale(mag_coh)

    tensor = np.stack([ch0, ch1, ch2], axis=0)
    expected = (CSD_CHANNELS, int(n_time), int(n_freq))
    if tensor.shape != expected:
        raise RuntimeError(f"CSD tensor shape {tensor.shape} != {expected}")

    energies: list[float] = []
    raw_power = {
        "H1": float(np.sum(mag_h1 * mag_h1)),
        "L1": float(np.sum(mag_l1 * mag_l1)),
    }
    for det in energy_detectors:
        if det in ("H1", "L1"):
            energies.append(float(np.log(raw_power[det] + ENERGY_EPS)))
        elif det in td_by_det:
            _, e = log_unit_scale_stft_grid(
                td_by_det[det], sample_rate, **stft_kw
            )
            energies.append(float(e))
        else:
            energies.append(0.0)

    log_energy = np.asarray(energies, dtype=np.float32)
    # Whitened STFT log-energy is typically O(1–8); clip rare numerical blow-ups
    # so the energy head is not poisoned by ASD near-zeros / loud outliers.
    log_energy = np.clip(log_energy, 0.0, 12.0)
    return tensor, log_energy


def default_stft_config() -> Dict[str, Any]:
    return {
        "n_time": int(SPECTROGRAM_TIME_STEPS),
        "n_freq": int(SPECTROGRAM_FREQ_BINS),
        "n_fft": int(DEFAULT_N_FFT),
        "win_length": int(DEFAULT_WIN_LENGTH),
        "hop_length": int(DEFAULT_HOP_LENGTH),
        "csd_channels": int(CSD_CHANNELS),
        "spectrogram_layout": SPECTROGRAM_LAYOUT,
    }


def default_robust_stft_config() -> Dict[str, Any]:
    return {
        "n_time": int(ROBUST_N_TIME),
        "n_freq": int(ROBUST_N_FREQ),
        "n_fft": int(DEFAULT_N_FFT),
        "win_length": int(DEFAULT_WIN_LENGTH),
        "hop_length": int(DEFAULT_HOP_LENGTH),
        "csd_channels": int(ROBUST_CHANNELS),
        "spectrogram_layout": ROBUST_LAYOUT,
    }


def _log10_grid(mag: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(mag, LOG_FLOOR)).astype(np.float64)


def apply_fixed_channel_norm(
    log_grids: Sequence[np.ndarray],
    stats: Mapping[str, Any],
) -> np.ndarray:
    """Apply stored per-channel median/IQR normalization."""
    med = np.asarray(stats["channel_median"], dtype=np.float64).reshape(-1)
    iqr = np.asarray(stats["channel_iqr"], dtype=np.float64).reshape(-1)
    iqr = np.maximum(iqr, UNIT_SCALE_EPS)
    out = []
    for i, g in enumerate(log_grids):
        out.append(((g - med[i]) / iqr[i]).astype(np.float32))
    return np.stack(out, axis=0)


def standardize_log_energy(
    log_energy: np.ndarray,
    stats: Mapping[str, Any],
) -> np.ndarray:
    """Standardize raw log-energy with stored mean/std (no hard clip)."""
    e = np.asarray(log_energy, dtype=np.float64).ravel()
    mu = np.asarray(stats["energy_mean"], dtype=np.float64).ravel()
    sig = np.asarray(stats["energy_std"], dtype=np.float64).ravel()
    sig = np.maximum(sig, UNIT_SCALE_EPS)
    n = min(e.size, mu.size)
    out = np.zeros_like(e, dtype=np.float32)
    out[:n] = ((e[:n] - mu[:n]) / sig[:n]).astype(np.float32)
    return out


def build_robust_spectrogram_from_td(
    td_by_det: Mapping[str, np.ndarray],
    sample_rate: float,
    *,
    n_time: int = ROBUST_N_TIME,
    n_freq: int = ROBUST_N_FREQ,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    energy_detectors: Sequence[str] = ("H1", "L1", "V1"),
    norm_stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 6-channel HLV mag/coherence tensor + 3-IFO raw log-energy.

    Channels (float32, shape ``(6, n_time, n_freq)``):
      0–2: log10 ``|S_H1|``, ``|S_L1|``, ``|S_V1|``
      3–5: log10 ``|S_H1 conj(S_L1)|``, ``|S_H1 conj(S_V1)|``, ``|S_L1 conj(S_V1)|``

    If ``norm_stats`` is provided, applies fixed robust channel + energy norms.
    Otherwise returns unnormalized log10 grids and raw log-energy (for calibration).
    """
    required = ("H1", "L1", "V1")
    for det in required:
        if det not in td_by_det:
            raise KeyError(f"Robust spectrogram requires {det} TD strain")

    stft_kw = dict(
        n_time=int(n_time),
        n_freq=int(n_freq),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    s = {
        det: compute_complex_stft_grid(
            crop_td_to_analysis_window(td_by_det[det], sample_rate),
            float(sample_rate),
            **stft_kw,
        )
        for det in required
    }
    mags = {
        "H1": np.abs(s["H1"]),
        "L1": np.abs(s["L1"]),
        "V1": np.abs(s["V1"]),
        "HL": np.abs(s["H1"] * np.conj(s["L1"])),
        "HV": np.abs(s["H1"] * np.conj(s["V1"])),
        "LV": np.abs(s["L1"] * np.conj(s["V1"])),
    }
    log_grids = [_log10_grid(mags[name]) for name in ROBUST_CHANNEL_NAMES]

    energies: list[float] = []
    for det in energy_detectors:
        if det in ("H1", "L1", "V1"):
            energies.append(float(np.log(float(np.sum(mags[det] ** 2)) + ENERGY_EPS)))
        else:
            energies.append(0.0)
    log_energy = np.asarray(energies, dtype=np.float32)

    if norm_stats is None:
        tensor = np.stack([g.astype(np.float32) for g in log_grids], axis=0)
    else:
        tensor = apply_fixed_channel_norm(log_grids, norm_stats)
        log_energy = standardize_log_energy(log_energy, norm_stats)

    expected = (ROBUST_CHANNELS, int(n_time), int(n_freq))
    if tensor.shape != expected:
        raise RuntimeError(f"robust tensor shape {tensor.shape} != {expected}")
    return tensor, log_energy


def calibrate_robust_norm_stats(
    raw_tensors: Sequence[np.ndarray],
    raw_energies: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Calibrate fixed robust stats from unnormalized log10 tensors / energies.

    ``raw_tensors`` entries are ``(6, T, F)`` log10 grids (no per-sample Z).
    """
    if not raw_tensors:
        raise ValueError("need at least one tensor to calibrate")
    stacked = np.stack([np.asarray(t, dtype=np.float64) for t in raw_tensors], axis=0)
    # stacked: (N, C, T, F)
    c = stacked.shape[1]
    med = np.empty(c, dtype=np.float64)
    iqr = np.empty(c, dtype=np.float64)
    for i in range(c):
        vals = stacked[:, i].reshape(-1)
        q25, q50, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        med[i] = q50
        iqr[i] = max(q75 - q25, UNIT_SCALE_EPS)
    e_stack = np.stack([np.asarray(e, dtype=np.float64).ravel() for e in raw_energies], axis=0)
    return {
        "channel_median": med.astype(np.float32),
        "channel_iqr": iqr.astype(np.float32),
        "energy_mean": e_stack.mean(axis=0).astype(np.float32),
        "energy_std": np.maximum(e_stack.std(axis=0), UNIT_SCALE_EPS).astype(np.float32),
        "n_calibrate": int(stacked.shape[0]),
        "layout": ROBUST_LAYOUT,
        "csd_channels": int(ROBUST_CHANNELS),
        "n_time": int(stacked.shape[2]),
        "n_freq": int(stacked.shape[3]),
    }
