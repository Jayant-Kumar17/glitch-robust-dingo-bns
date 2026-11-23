"""Single source of truth for strain -> frequency-domain packaging.

Reproduces ``dingo.gw.data.data_preparation.data_to_domain`` for one detector:
Tukey window (``alpha = 2 * roll_off / T``), ``rfft / f_s``, then
``cyclic_time_shift(time_buffer)``, i.e. multiplication by
``exp(-2j * pi * f * time_buffer)``.

Every glitch injection and every gate rebuild must go through
``package_fd`` so that injected glitches, gate deltas, and the archived event
package all live in the same time frame as the frozen network.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as sp_signal

DEFAULT_TIME_BUFFER_S = 2.0


def package_fd(
    td: np.ndarray,
    sample_rate: float,
    *,
    roll_off: float = 0.4,
    f_max: float,
    n_freq: Optional[int] = None,
    time_buffer: float = DEFAULT_TIME_BUFFER_S,
) -> np.ndarray:
    """Windowed rFFT of ``td`` in the DINGO event-package convention."""
    x = np.asarray(td, dtype=np.float64).ravel()
    n = len(x)
    alpha = 2.0 * float(roll_off) * float(sample_rate) / max(n, 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    window = sp_signal.windows.tukey(n, alpha=alpha)
    fd = np.fft.rfft(x * window) / float(sample_rate)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    if float(time_buffer) != 0.0:
        fd = fd * np.exp(-2j * np.pi * freqs * float(time_buffer))
    if n_freq is None:
        df = freqs[1] if len(freqs) > 1 else 1.0
        n_keep = min(int(np.floor(float(f_max) / df)) + 1, len(fd))
    else:
        n_keep = int(n_freq)
    if len(fd) < n_keep:
        fd = np.pad(fd, (0, n_keep - len(fd)))
    return np.asarray(fd[:n_keep], dtype=np.complex128)


def coloured_noise_td(
    asd: np.ndarray,
    n_samples: int,
    sample_rate: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Stationary Gaussian noise with one-sided PSD ``asd**2`` on the package grid.

    ``asd[k]`` belongs to frequency ``k * sample_rate / n_samples``. Bins where the
    ASD is a placeholder (``<= 0`` or ``>= 1e-6``, e.g. the 1.0 that DINGO writes
    below ``f_min``) and bins above ``len(asd)`` are left empty, so the noise is
    band-limited to the analysis band.
    """
    n = int(n_samples)
    n_rfft = n // 2 + 1
    a = np.asarray(asd, dtype=np.float64).ravel()
    m = min(a.size, n_rfft)
    valid = (a[:m] > 0.0) & (a[:m] < 1e-6)
    rng = np.random.default_rng(int(seed))
    spec = np.zeros(n_rfft, dtype=np.complex128)
    draw = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    spec[:m] = np.where(valid, draw * a[:m], 0.0) * (np.sqrt(n * float(sample_rate)) / 2.0)
    spec[0] = 0.0
    return np.fft.irfft(spec, n=n)


def band_limit_td(
    td: np.ndarray,
    sample_rate: float,
    f_lo: float,
    f_hi: float,
    *,
    roll_off: float = 0.4,
) -> np.ndarray:
    """Segment-windowed strain with everything below the analysis band removed.

    Zero below ``f_lo``, unity above ``f_hi``, raised-cosine in between. Gating raw
    strain multiplies the seismic wall by the gate taper and leaks it into the
    analysis band; gating this band-limited series does not.
    """
    x = np.asarray(td, dtype=np.float64).ravel()
    n = len(x)
    alpha = float(np.clip(2.0 * float(roll_off) * float(sample_rate) / max(n, 1), 0.0, 1.0))
    spec = np.fft.rfft(x * sp_signal.windows.tukey(n, alpha=alpha))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    ramp = np.clip((freqs - float(f_lo)) / max(float(f_hi) - float(f_lo), 1e-12), 0.0, 1.0)
    return np.fft.irfft(spec * (0.5 - 0.5 * np.cos(np.pi * ramp)), n=n)
