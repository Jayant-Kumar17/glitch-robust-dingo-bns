"""Common loudness scale for injected glitches: whitened optimal SNR.

``rho_w = sqrt( 4 * sum_{f_min<=f<=f_max} |g(f)|^2 / S_n(f) * df )``

with ``g(f)`` the frequency-domain glitch on the DINGO packaging grid
(``td_to_fd_strain``: Tukey-windowed rFFT divided by the sample rate, so units
are strain/Hz) and ``S_n = ASD^2`` the one-sided noise PSD of the GW170817 H1
analysis ASD on the same grid (``k * delta_f``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def whitened_optimal_snr(
    g_td: np.ndarray,
    sample_rate: float,
    *,
    asd: np.ndarray,
    delta_f: float,
    f_min: float,
    f_max: float,
    roll_off: float = 0.4,
    g_fd: Optional[np.ndarray] = None,
) -> float:
    """Whitened optimal SNR of a time-domain glitch against ``asd``.

    ``g_td`` must be on the analysis-segment grid (length ``T * f_s``) so that
    its rFFT lands on the packaged frequency grid. Pass ``g_fd`` to skip the
    transform if the packaged FD glitch is already available.
    """
    from adapt.event_glitch_io import td_to_fd_strain

    asd_arr = np.asarray(asd, dtype=np.float64).ravel()
    if g_fd is None:
        g_fd = td_to_fd_strain(np.asarray(g_td, dtype=np.float64), float(sample_rate), roll_off=roll_off, f_max=f_max)
    g_fd = np.asarray(g_fd, dtype=np.complex128).ravel()
    n = min(g_fd.size, asd_arr.size)
    g_fd, a = g_fd[:n], asd_arr[:n]
    f = np.arange(n, dtype=np.float64) * float(delta_f)
    band = (f >= float(f_min)) & (f <= float(f_max)) & (a > 0) & np.isfinite(a)
    psd = a[band] ** 2
    rho2 = 4.0 * float(np.sum(np.abs(g_fd[band]) ** 2 / psd)) * float(delta_f)
    return float(np.sqrt(max(rho2, 0.0)))


def scale_to_rho(
    g_td: np.ndarray,
    target_rho: float,
    sample_rate: float,
    *,
    asd: np.ndarray,
    delta_f: float,
    f_min: float,
    f_max: float,
    roll_off: float = 0.4,
) -> tuple[np.ndarray, float, float]:
    """Return ``(k * g_td, k, rho_before)`` with ``k`` chosen so that the
    whitened optimal SNR equals ``target_rho``."""
    rho0 = whitened_optimal_snr(g_td, sample_rate, asd=asd, delta_f=delta_f, f_min=f_min, f_max=f_max, roll_off=roll_off)
    k = float(target_rho) / max(rho0, 1e-300)
    return np.asarray(g_td, dtype=np.float64) * k, k, rho0
