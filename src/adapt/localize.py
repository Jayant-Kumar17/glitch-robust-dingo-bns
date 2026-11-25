"""Whitened-excess glitch localiser (a simple, standard alternative to the CNN detector).

Each detector's full analysis-segment strain is band-limited, whitened with the
analysis ASD, and normalised to unit variance with a robust (MAD) scale. Samples
with ``|z| > sigma`` are clustered, and each cluster becomes one gate
``[first - pad_s, last + pad_s]`` in segment time. This is close in spirit to
PyCBC autogating and needs no training.

Driver integration: set ``ADAPT_LOCALIZER=excess`` (default ``cnn``) and
``ADAPT_EXCESS_SIGMA`` (default 8). The pad is the driver's gate half-width.
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

from adapt.glitch_excision import GateWindow

DEFAULT_SIGMA = 8.0


def localizer_name() -> str:
    name = os.environ.get("ADAPT_LOCALIZER", "cnn").strip().lower()
    if name not in ("cnn", "excess"):
        raise ValueError(f"ADAPT_LOCALIZER must be 'cnn' or 'excess', got {name!r}")
    return name


def excess_sigma() -> float:
    return float(os.environ.get("ADAPT_EXCESS_SIGMA", DEFAULT_SIGMA))


def whitened_z(
    td: np.ndarray,
    asd: np.ndarray,
    sample_rate: float,
    *,
    f_max: float,
    band: Tuple[float, float] = (20.0, 23.0),
    roll_off: float = 0.4,
    asd_df: Optional[float] = None,
    edge_s: float = 1.0,
) -> np.ndarray:
    """Unit-variance whitened, band-limited strain. ``asd`` is on a grid of spacing ``asd_df``."""
    x = np.asarray(td, dtype=np.float64).ravel()
    n = x.size
    fs = float(sample_rate)
    alpha = float(np.clip(2.0 * float(roll_off) * fs / max(n, 1), 0.0, 1.0))
    X = np.fft.rfft(x * sp_signal.windows.tukey(n, alpha=alpha))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    a = np.asarray(asd, dtype=np.float64).ravel()
    df = float(asd_df) if asd_df is not None else fs / n
    f_asd = np.arange(a.size) * df
    valid_asd = (a > 0.0) & (a < 1e-6)  # DINGO writes 1.0 below f_min
    a_i = np.interp(f, f_asd, np.where(valid_asd, a, np.nan), left=np.nan, right=np.nan)
    ramp = np.clip((f - band[0]) / max(band[1] - band[0], 1e-12), 0.0, 1.0)
    weight = 0.5 - 0.5 * np.cos(np.pi * ramp)
    ok = np.isfinite(a_i) & (f <= float(f_max)) & (weight > 0)
    Y = np.zeros_like(X)
    Y[ok] = X[ok] * weight[ok] / a_i[ok]
    y = np.fft.irfft(Y, n=n)
    e = int(round(float(edge_s) * fs))
    core = y[e : n - e] if n > 2 * e + 16 else y
    scale = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    z = y / max(scale, 1e-300)
    if e > 0 and n > 2 * e:
        z[:e] = 0.0
        z[n - e :] = 0.0
    return z


def gates_from_z(
    z: np.ndarray,
    sample_rate: float,
    *,
    detector: str,
    sigma: float,
    pad_s: float,
    merge_s: float = 0.05,
) -> List[GateWindow]:
    fs = float(sample_rate)
    idx = np.flatnonzero(np.abs(z) > float(sigma))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > int(round(float(merge_s) * fs)))
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    wins = []
    for s, e in zip(starts, ends):
        peak = float(np.max(np.abs(z[s : e + 1])))
        wins.append([s / fs - float(pad_s), e / fs + float(pad_s), peak])
    merged = [wins[0]]
    for w in wins[1:]:
        if w[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w[1])
            merged[-1][2] = max(merged[-1][2], w[2])
        else:
            merged.append(w)
    t_end = (z.size - 1) / fs
    return [
        GateWindow(detector=detector, t_start=max(0.0, a), t_end=min(t_end, b), score=float(p))
        for a, b, p in merged
    ]


def excess_gates(
    td_by_det: Mapping[str, np.ndarray],
    asds: Mapping[str, np.ndarray],
    sample_rate: float,
    *,
    f_max: float,
    pad_s: float,
    sigma: Optional[float] = None,
    detectors: Optional[Sequence[str]] = None,
    band: Tuple[float, float] = (20.0, 23.0),
    asd_df: Optional[float] = None,
) -> Tuple[List[GateWindow], Dict[str, float]]:
    """Gates for every detector in ``detectors`` (default: all in ``td_by_det``) and max |z| per detector."""
    sig = excess_sigma() if sigma is None else float(sigma)
    gates: List[GateWindow] = []
    zmax: Dict[str, float] = {}
    for det in detectors or list(td_by_det.keys()):
        if det not in td_by_det or det not in asds:
            continue
        z = whitened_z(td_by_det[det], asds[det], sample_rate, f_max=f_max, band=band, asd_df=asd_df)
        zmax[det] = float(np.max(np.abs(z)))
        gates.extend(gates_from_z(z, sample_rate, detector=det, sigma=sig, pad_s=pad_s))
    return gates, zmax


def select_gates(
    cnn_call,
    *,
    td_by_det: Mapping[str, np.ndarray],
    asds: Mapping[str, np.ndarray],
    sample_rate: float,
    f_max: float,
    pad_s: float,
    detectors: Optional[Sequence[str]] = None,
):
    """Return ``(gates, info)``. ``cnn_call()`` must return ``(gates, probs)`` from the CNN detector.

    ``ADAPT_LOCALIZER=excess`` replaces the CNN with :func:`excess_gates`; otherwise the CNN runs.
    """
    if localizer_name() == "excess":
        gates, zmax = excess_gates(
            td_by_det, asds, sample_rate, f_max=f_max, pad_s=pad_s, detectors=detectors
        )
        return gates, {"localizer": "excess", "sigma": excess_sigma(), "zmax": zmax}
    gates, _probs = cnn_call()
    return gates, {"localizer": "cnn"}
