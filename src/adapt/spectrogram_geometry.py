"""Shared spectrogram / STFT geometry for glitch detection and gating.

Constants and STFT grid helpers used by the frozen-DINGO-BNS front-end.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as sp_signal

SPECTROGRAM_TIME_STEPS = 5
SPECTROGRAM_FREQ_BINS = 128
SPECTROGRAM_ANALYSIS_SECONDS = 4.0
SAMPLE_RATE_HZ = 4096.0
FREQ_LO_HZ = 20.0
FREQ_HI_HZ = 512.0


def _prepare_stft_signal(
    strain: np.ndarray,
    sample_rate: float,
    *,
    n_time: int,
    n_fft: Optional[int],
    win_length: Optional[int],
    hop_length: Optional[int],
) -> tuple[np.ndarray, int, int, int]:
    """Crop/pad TD strain and resolve STFT geometry."""
    x = np.asarray(strain, dtype=np.float64).ravel()
    if x.size < 32:
        x = np.pad(x, (0, 32 - x.size))

    max_samples = int(round(SPECTROGRAM_ANALYSIS_SECONDS * sample_rate))
    if x.size > max_samples:
        start = (x.size - max_samples) // 2
        x = x[start : start + max_samples]

    if win_length is not None:
        nperseg = int(min(max(16, int(win_length)), x.size))
    else:
        nperseg = int(min(max(256, x.size // (n_time + 1)), x.size))

    if hop_length is not None:
        hop = int(max(1, min(int(hop_length), nperseg)))
        noverlap = int(max(0, nperseg - hop))
    else:
        noverlap = int(nperseg * 3 // 4)

    if n_fft is not None:
        nfft = int(max(nperseg, int(n_fft)))
    else:
        nfft = int(max(nperseg, 1024))
    return x, nperseg, noverlap, nfft


def _resample_tf_grid(
    values: np.ndarray,
    freqs: np.ndarray,
    *,
    n_time: int,
    n_freq: int,
    f_lo: float,
    f_hi: float,
) -> np.ndarray:
    """Resample a ``(F_stft, T_stft)`` real grid onto ``(n_time, n_freq)``."""
    if values.size == 0:
        return np.zeros((n_time, n_freq), dtype=np.float64)

    target_f = np.linspace(float(f_lo), float(f_hi), int(n_freq), dtype=np.float64)
    vals_f = np.empty((n_freq, values.shape[1]), dtype=np.float64)
    for t_idx in range(values.shape[1]):
        vals_f[:, t_idx] = np.interp(target_f, freqs, values[:, t_idx])

    if vals_f.shape[1] == 1:
        vals_ft = np.repeat(vals_f, n_time, axis=1)
    else:
        t_src = np.linspace(0.0, 1.0, vals_f.shape[1])
        t_tgt = np.linspace(0.0, 1.0, n_time)
        vals_ft = np.empty((n_freq, n_time), dtype=np.float64)
        for f_idx in range(n_freq):
            vals_ft[f_idx, :] = np.interp(t_tgt, t_src, vals_f[f_idx, :])

    grid = vals_ft.T
    if grid.shape != (n_time, n_freq):
        raise RuntimeError(
            f"spectrogram grid shape {grid.shape} != ({n_time}, {n_freq})"
        )
    return grid


def compute_stft_spectrogram_grid(
    strain: np.ndarray,
    sample_rate: float,
    *,
    n_time: int = SPECTROGRAM_TIME_STEPS,
    n_freq: int = SPECTROGRAM_FREQ_BINS,
    f_lo: float = FREQ_LO_HZ,
    f_hi: float = FREQ_HI_HZ,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """Windowed STFT magnitude resampled to exactly ``(n_time, n_freq)``."""
    x, nperseg, noverlap, nfft = _prepare_stft_signal(
        strain,
        sample_rate,
        n_time=n_time,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    freqs, _times, zxx = sp_signal.stft(
        x,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    return _resample_tf_grid(
        np.abs(zxx),
        freqs,
        n_time=int(n_time),
        n_freq=int(n_freq),
        f_lo=f_lo,
        f_hi=f_hi,
    )


def compute_complex_stft_grid(
    strain: np.ndarray,
    sample_rate: float,
    *,
    n_time: int = SPECTROGRAM_TIME_STEPS,
    n_freq: int = SPECTROGRAM_FREQ_BINS,
    f_lo: float = FREQ_LO_HZ,
    f_hi: float = FREQ_HI_HZ,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """Windowed complex STFT resampled to ``(n_time, n_freq)`` complex128."""
    x, nperseg, noverlap, nfft = _prepare_stft_signal(
        strain,
        sample_rate,
        n_time=n_time,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    freqs, _times, zxx = sp_signal.stft(
        x,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    if zxx.size == 0:
        return np.zeros((n_time, n_freq), dtype=np.complex128)

    real_g = _resample_tf_grid(
        np.real(zxx),
        freqs,
        n_time=int(n_time),
        n_freq=int(n_freq),
        f_lo=f_lo,
        f_hi=f_hi,
    )
    imag_g = _resample_tf_grid(
        np.imag(zxx),
        freqs,
        n_time=int(n_time),
        n_freq=int(n_freq),
        f_lo=f_lo,
        f_hi=f_hi,
    )
    return real_g.astype(np.float64) + 1j * imag_g.astype(np.float64)
