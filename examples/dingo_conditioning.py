"""Official DINGO conditioning of a gated time series.

``data_to_domain`` windows the strain with the 128 s Tukey
(``alpha = 2 * roll_off / T``), Fourier-transforms it, and applies
``cyclic_time_shift(time_buffer)``. The analysis ASD is the original
GW170817 spectrum, not a spectrum estimated on the gated segment.
"""

from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

WINDOW = {"type": "tukey", "roll_off": 0.4, "T": 128.0, "f_s": 4096.0}
F_MIN = 23.0
F_MAX = 1535.3046875
TIME_BUFFER = 2.0
SAMPLE_RATE = 4096.0
BAND_EDGES = (23.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, F_MAX)


def base_frequency_domain(base_domain):
    """Uniform frequency domain used by ``data_to_domain`` (not the multibanded one)."""
    from dingo.gw.domains.frequency_domain import FrequencyDomain

    if type(base_domain) is FrequencyDomain:
        return base_domain
    return FrequencyDomain(
        f_min=F_MIN,
        f_max=F_MAX,
        delta_f=float(base_domain.delta_f),
        window_factor=float(base_domain.window_factor),
    )


def _gates_for(gates: Sequence, detector: str) -> list:
    return [g for g in gates if getattr(g, "detector", None) == detector]


def gated_strain(
    td_by_det: Mapping[str, np.ndarray],
    gates: Sequence,
    sample_rate: float,
) -> Dict[str, np.ndarray]:
    """``x_g(t) = w(t) x(t)`` with the same Tukey gate used by matched-delta."""
    from adapt.glitch_excision import DEFAULT_TUKEY_ALPHA, apply_gates_td

    out: Dict[str, np.ndarray] = {}
    for det, td in td_by_det.items():
        det_gates = _gates_for(gates, det)
        series = np.asarray(td, dtype=np.float64).ravel()
        if det_gates:
            series = apply_gates_td(
                series, sample_rate, det_gates, t0=0.0, alpha=DEFAULT_TUKEY_ALPHA
            )
        out[det] = series
    return out


def inpainted_strain(
    td_by_det: Mapping[str, np.ndarray],
    gates: Sequence,
    sample_rate: float,
) -> Dict[str, np.ndarray]:
    """Replace each gated window by the straight line between its exterior samples."""
    out: Dict[str, np.ndarray] = {}
    for det, td in td_by_det.items():
        x = np.asarray(td, dtype=np.float64).ravel().copy()
        for gate in _gates_for(gates, det):
            i0 = int(np.floor(float(gate.t_start) * float(sample_rate)))
            i1 = int(np.ceil(float(gate.t_end) * float(sample_rate)))
            i0 = max(0, i0)
            i1 = min(x.size, max(i0 + 2, i1))
            left = x[i0 - 1] if i0 > 0 else x[i0]
            right = x[i1] if i1 < x.size else x[i1 - 1]
            n = i1 - i0
            x[i0:i1] = np.linspace(left, right, n + 2)[1:-1]
        out[det] = x
    return out


def data_to_domain_package(
    strain_by_det: Mapping[str, np.ndarray],
    asds: Mapping[str, np.ndarray],
    domain,
    *,
    sample_rate: float = SAMPLE_RATE,
    time_buffer: float = TIME_BUFFER,
) -> dict:
    """Run upstream ``data_to_domain`` and put the original analysis ASD back."""
    from dingo.gw.data.data_preparation import data_to_domain

    raw = {"strain": {}, "psd": {}}
    for det, series in strain_by_det.items():
        raw["strain"][det] = np.asarray(series, dtype=np.float64).ravel()
        asd = np.asarray(asds[det], dtype=np.float64).ravel()
        raw["psd"][det] = np.square(asd)
    packaged = data_to_domain(
        raw,
        {"f_s": float(sample_rate), "time_buffer": float(time_buffer)},
        domain,
        window=WINDOW,
    )
    for det, asd in asds.items():
        if det in packaged["asds"]:
            packaged["asds"][det] = np.asarray(asd, dtype=np.float64).copy()
    return packaged


def splice_package(
    event_data: Mapping,
    packaged: Mapping,
    asds: Mapping[str, np.ndarray],
) -> dict:
    """Copy an event package and replace only the detectors ``packaged`` contains."""
    data = copy.deepcopy(dict(event_data))
    for det, wave in packaged["waveform"].items():
        data["waveform"][det] = np.asarray(wave, dtype=np.complex128)
    for det, asd in asds.items():
        if det in data.get("asds", {}):
            data["asds"][det] = np.asarray(asd, dtype=np.float64).copy()
    return data


def repackage_event(
    event_data: Mapping,
    td_by_det: Mapping[str, np.ndarray],
    gates: Sequence,
    asds: Mapping[str, np.ndarray],
    domain,
    *,
    mode: str,
    sample_rate: float = SAMPLE_RATE,
    time_buffer: float = TIME_BUFFER,
) -> tuple[dict, Dict[str, np.ndarray]]:
    """Gate or inpaint, then run official ``data_to_domain``.

    The archived GW170817 strain is ``data_to_domain`` with ``time_buffer=2``
    applied once to the segment that already ends two seconds after the
    trigger. ``dingo_repackage`` and ``inpaint_repackage`` use that single
    shift. The ``*_aligned`` arms pass ``time_buffer=0`` so the cyclic shift
    is not applied on top of the cropped segment.
    """
    aligned = mode.endswith("_aligned")
    base = mode[: -len("_aligned")] if aligned else mode
    if base == "dingo_repackage":
        td = gated_strain(td_by_det, gates, sample_rate)
    elif base == "inpaint_repackage":
        td = inpainted_strain(td_by_det, gates, sample_rate)
    else:
        raise ValueError(mode)
    packaged = data_to_domain_package(
        td,
        asds,
        domain,
        sample_rate=sample_rate,
        time_buffer=0.0 if aligned else float(time_buffer),
    )
    return splice_package(event_data, packaged, asds), td


def packaging_residual(
    td: np.ndarray,
    packaged_fd: np.ndarray,
    asd: np.ndarray,
    *,
    sample_rate: float = SAMPLE_RATE,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    time_buffer: float = TIME_BUFFER,
) -> dict:
    """``ε(f) = X(f) - F[x](f)`` on the analysis band.

    ``X`` is the ``data_to_domain`` strain. ``F[x]`` is the unwindowed numpy
    Fourier transform of the same time series (``rfft / f_s``), on the same
    frequency grid. The cyclic time shift inside ``data_to_domain`` is therefore
    part of ``ε``. A second, shift-aligned comparison is stored beside it.
    """
    x = np.asarray(td, dtype=np.float64).ravel()
    X = np.asarray(packaged_fd, dtype=np.complex128).ravel()
    # Same Tukey as data_to_domain. Without it, the seismic wall leaks into
    # the analysis band and the residual is not a packaging comparison.
    from dingo.gw.gwutils import get_window

    window = get_window(WINDOW)
    if window.size != x.size:
        raise ValueError(f"analysis window {window.size} != strain {x.size}")
    plain = np.fft.rfft(x * window) / float(sample_rate)
    if plain.size < X.size:
        plain = np.pad(plain, (0, X.size - plain.size))
    plain = plain[: X.size]
    unwindowed = np.fft.rfft(x) / float(sample_rate)
    if unwindowed.size < X.size:
        unwindowed = np.pad(unwindowed, (0, X.size - unwindowed.size))
    unwindowed = unwindowed[: X.size]
    df = float(sample_rate) / float(x.size)
    freqs = np.arange(X.size, dtype=np.float64) * df
    asd_arr = np.asarray(asd, dtype=np.float64).ravel()[: X.size]
    live = (freqs >= float(f_min)) & (freqs <= float(f_max) + 1e-9)
    live = live & (asd_arr > 0.0) & (asd_arr < 1e-6)
    eps = X - plain
    e = eps[live]
    Xb = X[live]
    Fb = plain[live]
    fb = freqs[live]
    Sb = np.square(asd_arr[live])
    rel_l2 = float(np.linalg.norm(e) / (np.linalg.norm(Xb) + 1e-30))
    amplitude_ratio = float(np.linalg.norm(Fb) / (np.linalg.norm(Xb) + 1e-30))
    rel_l2_unwindowed = float(
        np.linalg.norm((X - unwindowed)[live]) / (np.linalg.norm(Xb) + 1e-30)
    )
    rho = float(np.sqrt(np.sum(4.0 * df * np.abs(e) ** 2 / np.maximum(Sb, 1e-300))))
    mag = np.minimum(np.abs(Xb), np.abs(Fb))
    floor = float(np.quantile(mag, 0.5)) if mag.size else 0.0
    use = mag >= max(floor, 1e-30)
    if np.any(use):
        dphi = np.abs(np.angle(Xb[use] * np.conj(Fb[use])))
        max_phase = float(np.max(dphi))
        median_phase = float(np.median(dphi))
    else:
        max_phase = float("nan")
        median_phase = float("nan")

    best_rel = None
    best_phase = None
    best_sign = None
    for sign in (1.0, -1.0):
        shifted = Fb * np.exp(-2j * np.pi * fb * sign * float(time_buffer))
        rel = float(np.linalg.norm(Xb - shifted) / (np.linalg.norm(Xb) + 1e-30))
        if np.any(use):
            phase = float(np.max(np.abs(np.angle(Xb[use] * np.conj(shifted[use])))))
        else:
            phase = float("nan")
        if best_rel is None or rel < best_rel:
            best_rel = rel
            best_phase = phase
            best_sign = sign

    power = 4.0 * df * np.abs(e) ** 2 / np.maximum(Sb, 1e-300)
    total = float(np.sum(power)) + 1e-30
    bands = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        mask = (fb >= lo) & (fb < hi if hi < F_MAX else fb <= hi)
        bands.append(
            {
                "f_lo": lo,
                "f_hi": hi,
                "whitened_power_fraction": float(np.sum(power[mask]) / total),
            }
        )
    return {
        "rel_l2": rel_l2,
        "rel_l2_unwindowed": rel_l2_unwindowed,
        "amplitude_ratio_F_over_X": amplitude_ratio,
        "rho_eps": rho,
        "max_phase_mismatch_rad": max_phase,
        "median_abs_phase_mismatch_rad": median_phase,
        "shift_aligned_rel_l2": best_rel,
        "shift_aligned_max_phase_rad": best_phase,
        "shift_sign_that_minimizes_l2": best_sign,
        "n_bins": int(np.sum(live)),
        "bands": bands,
    }


def residual_record(
    td_by_det: Mapping[str, np.ndarray],
    packaged: Mapping,
    asds: Mapping[str, np.ndarray],
    detectors: Optional[Sequence[str]] = None,
) -> dict:
    dets = list(detectors or td_by_det.keys())
    out = {}
    for det in dets:
        if det not in td_by_det or det not in packaged["waveform"]:
            continue
        out[det] = packaging_residual(
            td_by_det[det], packaged["waveform"][det], asds[det]
        )
    return out
