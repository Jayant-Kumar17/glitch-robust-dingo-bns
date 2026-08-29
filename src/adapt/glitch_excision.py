"""Detect-and-gate glitch excision around frozen DINGO.

Validated on GW170817 + injected H1 sine-Gaussian: Tukey-gating ±0.4 s around
the glitch peak and keeping the *original* (off-source) ASD recovers the clean
``d_L`` posterior when applied as a **matched TD→FD delta on the glitchy
package** (``rebuild_event_from_gated_td``). Embedding repair and imperfect
subtraction do not. Full FFT-replace of packaged H1 destroys demo conditioning
and rails ``d_L``.

Clean path is a bit-exact no-op when no gate is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

# Keep this module free of heavy STFT-builder imports.
SPECTROGRAM_ANALYSIS_SECONDS = 4.0

# Gate half-width that recovered d_L in the motivating ablation.
DEFAULT_GATE_HALF_S = 0.4
DEFAULT_TUKEY_ALPHA = 0.5
DEFAULT_MERGE_GAP_S = 0.05


@dataclass
class GateWindow:
    """One contiguous gate on a single detector."""

    detector: str
    t_start: float  # seconds in the TD segment convention used by the caller
    t_end: float
    score: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))


@dataclass
class ExcisionResult:
    """Output of applying excision to an event data package."""

    data: Dict[str, Any]
    gates: List[GateWindow] = field(default_factory=list)
    modified_detectors: List[str] = field(default_factory=list)
    noop: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


def mask_to_windows(
    mask: np.ndarray,
    *,
    sample_rate: float,
    t0: float = 0.0,
    detector: str = "H1",
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_width_s: float = 0.05,
    pad_s: float = 0.0,
    scores: Optional[np.ndarray] = None,
) -> List[GateWindow]:
    """Convert a boolean sample mask into merged gate windows."""
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0 or not np.any(m):
        return []
    padded = np.concatenate([[False], m, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    raw: List[Tuple[float, float, float]] = []
    for s, e in zip(starts, ends):
        t_s = float(t0) + float(s) / float(sample_rate) - float(pad_s)
        t_e = float(t0) + float(e) / float(sample_rate) + float(pad_s)
        if scores is not None:
            sc = float(np.max(np.asarray(scores).ravel()[s:e]))
        else:
            sc = 1.0
        raw.append((t_s, t_e, sc))
    if not raw:
        return []
    raw.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float, float]] = [raw[0]]
    gap = float(merge_gap_s)
    for t_s, t_e, sc in raw[1:]:
        ps, pe, psc = merged[-1]
        if t_s <= pe + gap:
            merged[-1] = (ps, max(pe, t_e), max(psc, sc))
        else:
            merged.append((t_s, t_e, sc))
    out: List[GateWindow] = []
    for t_s, t_e, sc in merged:
        if (t_e - t_s) < float(min_width_s):
            mid = 0.5 * (t_s + t_e)
            half = 0.5 * float(min_width_s)
            t_s, t_e = mid - half, mid + half
        out.append(GateWindow(detector=detector, t_start=t_s, t_end=t_e, score=sc))
    return out


def time_bin_mask_to_windows(
    bin_mask: np.ndarray,
    *,
    n_samples: int,
    sample_rate: float,
    t0: float = 0.0,
    detector: str = "H1",
    pad_s: float = DEFAULT_GATE_HALF_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    scores: Optional[np.ndarray] = None,
) -> List[GateWindow]:
    """Map STFT time-bin detections onto sample-domain gate windows.

    Bins are assumed uniform across ``[t0, t0 + n_samples/sample_rate)``.
    Each positive bin is expanded by ``pad_s`` on each side (default ±0.4 s,
    the half-width validated to recover GW170817 glitch ``d_L``).
    """
    mask = np.asarray(bin_mask, dtype=bool).ravel()
    n_bins = int(mask.size)
    if n_bins < 1 or not np.any(mask):
        return []
    duration = float(n_samples) / float(sample_rate)
    bin_dur = duration / float(n_bins)
    sample_mask = np.zeros(int(n_samples), dtype=bool)
    score_arr = np.zeros(int(n_samples), dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64).ravel() if scores is not None else None
    for i, hit in enumerate(mask):
        if not hit:
            continue
        # Bin center in segment time.
        t_c = float(t0) + (i + 0.5) * bin_dur
        i0 = int(np.floor((t_c - float(pad_s) - float(t0)) * float(sample_rate)))
        i1 = int(np.ceil((t_c + float(pad_s) - float(t0)) * float(sample_rate)))
        i0 = max(0, i0)
        i1 = min(int(n_samples), max(i0 + 1, i1))
        sample_mask[i0:i1] = True
        val = float(sc[i]) if sc is not None and i < sc.size else 1.0
        score_arr[i0:i1] = np.maximum(score_arr[i0:i1], val)
    return mask_to_windows(
        sample_mask,
        sample_rate=sample_rate,
        t0=t0,
        detector=detector,
        merge_gap_s=merge_gap_s,
        min_width_s=2.0 * float(pad_s) * 0.5,
        pad_s=0.0,
        scores=score_arr,
    )


def tukey_gate_weights(
    n_samples: int,
    sample_rate: float,
    windows: Sequence[GateWindow],
    *,
    t0: float = 0.0,
    alpha: float = DEFAULT_TUKEY_ALPHA,
) -> np.ndarray:
    """Multiplicative TD weights: 1 outside gates, Tukey→0 inside each window."""
    w = np.ones(int(n_samples), dtype=np.float64)
    if not windows:
        return w
    for gw in windows:
        i0 = int(np.floor((float(gw.t_start) - float(t0)) * float(sample_rate)))
        i1 = int(np.ceil((float(gw.t_end) - float(t0)) * float(sample_rate)))
        i0 = max(0, i0)
        i1 = min(int(n_samples), max(i0 + 1, i1))
        n = i1 - i0
        if n <= 1:
            w[i0:i1] = 0.0
            continue
        taper = sp_signal.windows.tukey(n, alpha=float(np.clip(alpha, 0.0, 1.0)))
        # Zero the gated region with tapered edges (weight = 1 - tukey).
        w[i0:i1] = np.minimum(w[i0:i1], 1.0 - taper)
    return w


def apply_gates_td(
    td: np.ndarray,
    sample_rate: float,
    windows: Sequence[GateWindow],
    *,
    t0: float = 0.0,
    alpha: float = DEFAULT_TUKEY_ALPHA,
) -> np.ndarray:
    """Apply Tukey gates to a TD strain series."""
    x = np.asarray(td, dtype=np.float64).ravel().copy()
    det_windows = [g for g in windows]  # caller filters by detector
    if not det_windows:
        return x
    w = tukey_gate_weights(
        x.size, sample_rate, det_windows, t0=t0, alpha=alpha
    )
    return x * w


def td_to_fd_strain(
    td: np.ndarray,
    sample_rate: float,
    *,
    roll_off: float = 0.4,
    f_max: float,
    n_freq: Optional[int] = None,
) -> np.ndarray:
    """Tukey-windowed rFFT matching the GW170817 packaging convention."""
    x = np.asarray(td, dtype=np.float64).ravel()
    alpha = 2.0 * float(roll_off) * float(sample_rate) / max(len(x), 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    window = sp_signal.windows.tukey(len(x), alpha=alpha)
    fd = np.fft.rfft(x * window) / float(sample_rate)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate))
    if n_freq is None:
        n_keep = int(np.floor(float(f_max) / (freqs[1] if len(freqs) > 1 else 1.0))) + 1
        n_keep = min(n_keep, len(fd))
    else:
        n_keep = int(n_freq)
    if len(fd) < n_keep:
        fd = np.pad(fd, (0, n_keep - len(fd)))
    return np.asarray(fd[:n_keep], dtype=np.complex128)


def gate_delta_fd(
    td_clean: np.ndarray,
    td_glitchy: np.ndarray,
    sample_rate: float,
    windows: Sequence[GateWindow],
    *,
    t0: float = 0.0,
    roll_off: float = 0.4,
    f_max: float,
    n_freq: int,
    alpha: float = DEFAULT_TUKEY_ALPHA,
) -> np.ndarray:
    """Self-consistent FD delta for gated glitchy strain vs ungated clean.

    ``delta = FFT(gated_glitchy) - FFT(clean)`` — the convention validated to
    recover clean ``d_L`` when added onto the packaged FD waveform while keeping
    the original ASD.
    """
    gated = apply_gates_td(
        td_glitchy, sample_rate, windows, t0=t0, alpha=alpha
    )
    fd_g = td_to_fd_strain(
        gated, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
    )
    fd_c = td_to_fd_strain(
        td_clean, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
    )
    return fd_g - fd_c


def rebuild_event_from_gated_td(
    event_data: Mapping[str, Any],
    *,
    td_by_det: Mapping[str, np.ndarray],
    gates: Sequence[GateWindow],
    sample_rate: float,
    roll_off: float = 0.4,
    f_max: float,
    t0: float = 0.0,
    original_asds: Optional[Mapping[str, np.ndarray]] = None,
    alpha: float = DEFAULT_TUKEY_ALPHA,
    replace_ungated: bool = False,
    mode: str = "matched_delta",
) -> ExcisionResult:
    """Honest excision on the **glitchy** DINGO package.

    ``mode``:
      - ``"matched_delta"`` (default, recovers ``d_L``): for each gated IFO apply
        ``packaged_FD += FFT(gated_TD) - FFT(ungated_TD)`` using the **same**
        analysis-segment TD that contains the injected glitch, then restore
        ``original_asds``. This is algebraically the working clean-base recipe
        when ``packaged = clean_HDF5 + FFT(glitch)``.
      - ``"replace"``: overwrite ``packaged_FD`` with ``FFT(gated_TD)``. Keeps
        TD↔FD consistent but discards demo HDF5 conditioning and **rails**
        ``d_L`` on GW170817; kept only as an ablation.

    Never feed on-source Welch of the glitchy segment into DINGO when
    ``original_asds`` is provided.

    If ``gates`` is empty and ``replace_ungated`` is False → bit-exact no-op.
    """
    import copy

    data = copy.deepcopy(dict(event_data))
    gates_list = list(gates)
    mode_l = str(mode).lower().strip()
    if mode_l in ("replace", "full_replace", "fft_replace"):
        mode_l = "replace"
    else:
        mode_l = "matched_delta"

    if not gates_list and not replace_ungated:
        if original_asds is not None:
            for det, asd in original_asds.items():
                if det in data.get("asds", {}):
                    data["asds"][det] = np.asarray(asd, dtype=np.float64).copy()
        return ExcisionResult(
            data=data,
            gates=[],
            modified_detectors=[],
            noop=True,
            meta={
                "n_gates": 0,
                "asd_policy": "original" if original_asds is not None else "unchanged",
                "packaged_base": "rebuild_from_gated_td",
                "rebuild_mode": mode_l,
            },
        )

    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))

    by_det: Dict[str, List[GateWindow]] = {}
    for g in gates_list:
        by_det.setdefault(g.detector, []).append(g)

    dets_to_rebuild = set(by_det.keys())
    if replace_ungated:
        dets_to_rebuild |= set(td_by_det.keys())

    modified: List[str] = []
    residual_power: Dict[str, float] = {}
    for det in sorted(dets_to_rebuild):
        if det not in td_by_det or det not in data["waveform"]:
            continue
        td = np.asarray(td_by_det[det], dtype=np.float64).ravel()
        det_gates = by_det.get(det, [])
        if det_gates:
            td_use = apply_gates_td(td, sample_rate, det_gates, t0=t0, alpha=alpha)
        else:
            td_use = td
        fd_gated = td_to_fd_strain(
            td_use, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
        )
        fd_ungated = td_to_fd_strain(
            td, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
        )
        packaged = np.asarray(data["waveform"][det], dtype=np.complex128)
        if mode_l == "replace":
            fd_new = fd_gated
        else:
            # Matched delta: remove the gated-region contribution measured on
            # the same TD series that was used for injection / detection.
            fd_new = packaged + (fd_gated - fd_ungated)
        num = float(np.linalg.norm(fd_ungated - fd_gated))
        den = float(np.linalg.norm(fd_ungated)) + 1e-30
        residual_power[det] = num / den
        data["waveform"][det] = fd_new
        modified.append(det)

    if original_asds is not None:
        for det, asd in original_asds.items():
            if det in data.get("asds", {}):
                data["asds"][det] = np.asarray(asd, dtype=np.float64).copy()

    noop = len(gates_list) == 0 and not replace_ungated
    return ExcisionResult(
        data=data,
        gates=gates_list,
        modified_detectors=modified,
        noop=noop,
        meta={
            "n_gates": len(gates_list),
            "gate_half_default_s": DEFAULT_GATE_HALF_S,
            "tukey_alpha": float(alpha),
            "asd_policy": "original" if original_asds is not None else "unchanged",
            "packaged_base": "rebuild_from_gated_td",
            "rebuild_mode": mode_l,
            "residual_power_frac": residual_power,
            "replace_ungated": bool(replace_ungated),
        },
    )


def apply_excision_to_event_data(
    event_data: Mapping[str, Any],
    *,
    td_by_det: Mapping[str, np.ndarray],
    td_clean_by_det: Optional[Mapping[str, np.ndarray]] = None,
    gates: Sequence[GateWindow],
    sample_rate: float,
    roll_off: float = 0.4,
    f_max: float,
    t0: float = 0.0,
    keep_original_asd: bool = True,
    original_asds: Optional[Mapping[str, np.ndarray]] = None,
    alpha: float = DEFAULT_TUKEY_ALPHA,
    packaged_base: str = "glitchy",
) -> ExcisionResult:
    """Apply excision to an event package.

    Prefer ``rebuild_event_from_gated_td`` (matched delta on the glitchy package)
    for honest eval. ``packaged_base`` in ``{"rebuild","replace","gated_td",
    "matched_delta"}`` redirects there.
    """
    base = str(packaged_base).lower().strip()
    if base in ("rebuild", "gated_td", "matched_delta", "honest"):
        return rebuild_event_from_gated_td(
            event_data,
            td_by_det=td_by_det,
            gates=gates,
            sample_rate=sample_rate,
            roll_off=roll_off,
            f_max=f_max,
            t0=t0,
            original_asds=original_asds if keep_original_asd else None,
            alpha=alpha,
            mode="matched_delta",
        )
    if base in ("replace", "full_replace"):
        return rebuild_event_from_gated_td(
            event_data,
            td_by_det=td_by_det,
            gates=gates,
            sample_rate=sample_rate,
            roll_off=roll_off,
            f_max=f_max,
            t0=t0,
            original_asds=original_asds if keep_original_asd else None,
            alpha=alpha,
            mode="replace",
        )

    import copy

    data = copy.deepcopy(dict(event_data))
    gates_list = list(gates)
    if not gates_list:
        return ExcisionResult(data=data, gates=[], modified_detectors=[], noop=True)

    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))
    modified: List[str] = []

    by_det: Dict[str, List[GateWindow]] = {}
    for g in gates_list:
        by_det.setdefault(g.detector, []).append(g)

    for det, det_gates in by_det.items():
        if det not in td_by_det or det not in data["waveform"]:
            continue
        td_g = np.asarray(td_by_det[det], dtype=np.float64).ravel()
        gated = apply_gates_td(td_g, sample_rate, det_gates, t0=t0, alpha=alpha)
        fd_gated = td_to_fd_strain(
            gated, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
        )
        if base == "clean":
            if td_clean_by_det is None or det not in td_clean_by_det:
                raise ValueError(
                    "packaged_base='clean' requires td_clean_by_det for each gated IFO"
                )
            td_c = np.asarray(td_clean_by_det[det], dtype=np.float64).ravel()
            fd_ref = td_to_fd_strain(
                td_c, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
            )
        else:
            fd_ref = td_to_fd_strain(
                td_g, sample_rate, roll_off=roll_off, f_max=f_max, n_freq=n_freq
            )
        delta = fd_gated - fd_ref
        data["waveform"][det] = (
            np.asarray(data["waveform"][det], dtype=np.complex128) + delta
        )
        modified.append(det)

    if keep_original_asd and original_asds is not None:
        for det, asd in original_asds.items():
            if det in data.get("asds", {}):
                data["asds"][det] = np.asarray(asd, dtype=np.float64).copy()

    return ExcisionResult(
        data=data,
        gates=gates_list,
        modified_detectors=modified,
        noop=len(modified) == 0,
        meta={
            "n_gates": len(gates_list),
            "gate_half_default_s": DEFAULT_GATE_HALF_S,
            "tukey_alpha": float(alpha),
            "asd_policy": "original" if keep_original_asd else "unchanged",
            "packaged_base": base,
        },
    )


def analysis_crop_bounds(
    *,
    duration: float,
    time_buffer: float,
    sample_rate: float,
    crop_seconds: float = SPECTROGRAM_ANALYSIS_SECONDS,
) -> Tuple[int, int, int]:
    """Return ``(trig_idx, crop_start, crop_end)`` for the trigger-centered crop."""
    trig_idx = int(round((float(duration) - float(time_buffer)) * float(sample_rate)))
    n_crop = int(round(float(crop_seconds) * float(sample_rate)))
    half = n_crop // 2
    start = max(0, trig_idx - half)
    end = start + n_crop
    return trig_idx, start, end


def glitch_support_mask_on_crop(
    *,
    n_crop: int,
    sample_rate: float,
    t_rel: float,
    half_width_s: float,
    family: str = "sine_gaussian",
    params: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Boolean sample mask on a trigger-centered analysis crop for a known glitch."""
    params = dict(params or {})
    mid = n_crop // 2
    # Family-specific intrinsic width, then take max with requested half_width.
    if family == "sine_gaussian":
        f0 = float(params.get("f0", 100.0))
        q = float(params.get("q", 5.0))
        tau = q / (2.0 * np.pi * max(f0, 1.0))
        hw = max(float(half_width_s), 3.0 * tau)
    elif family in ("broadband_burst", "whistle", "scattered_light", "narrowband_tone", "ringing"):
        dur = float(params.get("duration", 0.5))
        hw = max(float(half_width_s), 0.5 * dur + 0.05)
    elif family == "glitch_train":
        n_pulses = int(params.get("n_pulses", 3))
        spacing = float(params.get("spacing", 0.15))
        hw = max(float(half_width_s), 0.5 * n_pulses * spacing + 0.1)
    elif family == "double_blip":
        sep = float(params.get("separation", 0.2))
        hw = max(float(half_width_s), 0.5 * sep + 0.15)
    else:
        hw = float(half_width_s)
    peak = mid + int(round(float(t_rel) * float(sample_rate)))
    i0 = max(0, peak - int(round(hw * sample_rate)))
    i1 = min(n_crop, peak + int(round(hw * sample_rate)) + 1)
    mask = np.zeros(n_crop, dtype=bool)
    if i1 > i0:
        mask[i0:i1] = True
    return mask


def support_mask_to_time_bins(
    sample_mask: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Downsample a sample mask to ``n_time`` STFT bins (any-positive pooling)."""
    m = np.asarray(sample_mask, dtype=bool).ravel()
    n = m.size
    if n_time < 1:
        raise ValueError("n_time must be >= 1")
    if n == 0:
        return np.zeros(n_time, dtype=np.float32)
    edges = np.linspace(0, n, n_time + 1).astype(np.int64)
    out = np.zeros(n_time, dtype=np.float32)
    for i in range(n_time):
        if np.any(m[edges[i] : edges[i + 1]]):
            out[i] = 1.0
    return out
