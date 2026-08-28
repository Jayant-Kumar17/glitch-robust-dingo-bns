"""Advanced noise profiling for ADAPT (local + multi-detector hub).

This module is intentionally isolated from the routing / physics stack. It
extracts a "Rich Noise Profile" from real (or fallback synthetic) detector
strain: a Welch PSD for Gaussian/stationary structure, plus windowed
higher-order moments (std, skewness, excess kurtosis) that track
non-Gaussian and non-stationary behaviour such as glitches and slow
environmental drift.

``GlobalNoiseHub`` orchestrates independent ``LocalNoiseTracker`` instances
across a detector network and concatenates their profiles into a single
global vector. Network acquisition prefers real GWOSC strain per detector
and falls back to detector-specific colored noise only for failed sites.

Physical intuition for the non-Gaussian moments:
  - Standard deviation: local noise power / variance tracking over short windows.
  - Skewness: asymmetry of the amplitude distribution (one-sided artifacts).
  - Excess kurtosis: heavy tails / impulsive transients (glitch sensitivity).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

# Matplotlib is imported lazily inside plot helpers so offline feature
# extraction does not pay the Agg/font-cache cost until diagnostics are needed.


@dataclass(frozen=True)
class ObservatorySpec:
    """One site in the ADAPT multi-detector observatory catalog.

    ``gwosc=True`` means GWOSC may host open strain for this IFO code; the
    fetcher will attempt a live download before falling back to simulation.
    ``gwosc=False`` sites (planned / space / next-gen) skip the network call
    and synthesize colored noise from ``psd_name`` immediately.
    """

    code: str
    name: str
    psd_name: str
    gwosc: bool
    family: str


# ~25-site global network: maximize real GWOSC pull (H1/L1/V1/K1/G1), then
# fill the rest with design-sensitivity simulations for planned / next-gen
# / space-based concepts. PSD names are PyCBC callables resolved lazily.
_OBSERVATORY_CATALOG: Tuple[ObservatorySpec, ...] = (
    # ---- Real open-data sites (try GWOSC first) ----
    ObservatorySpec("H1", "LIGO Hanford", "aLIGOZeroDetHighPower", True, "ligo"),
    ObservatorySpec("L1", "LIGO Livingston", "aLIGOZeroDetHighPower", True, "ligo"),
    ObservatorySpec("V1", "Virgo", "AdVDesignSensitivityP1200087", True, "virgo"),
    ObservatorySpec("K1", "KAGRA", "KAGRADesignSensitivityT1600593", True, "kagra"),
    ObservatorySpec("G1", "GEO600", "GEO", True, "geo"),
    # ---- Ground-based network expansions (simulated) ----
    ObservatorySpec(
        "A1", "LIGO-India (Aundha)", "aLIGOAPlusDesignSensitivityT1800042", False, "ligo"
    ),
    ObservatorySpec(
        "H1A", "LIGO Hanford A+", "aLIGOAPlusDesignSensitivityT1800042", False, "ligo"
    ),
    ObservatorySpec(
        "L1A", "LIGO Livingston A+", "aLIGOAPlusDesignSensitivityT1800042", False, "ligo"
    ),
    ObservatorySpec("V1O4", "Virgo O4 design", "AdVO4T1800545", False, "virgo"),
    ObservatorySpec(
        "K1D", "KAGRA design", "KAGRADesignSensitivityT1600593", False, "kagra"
    ),
    ObservatorySpec(
        "K1L", "KAGRA late", "KAGRALateSensitivityT1600593", False, "kagra"
    ),
    ObservatorySpec(
        "NEMO", "NEMO (Australia)", "aLIGOBNSOptimizedSensitivityP1200087", False, "nextgen"
    ),
    # ---- 3G terrestrial concepts ----
    ObservatorySpec(
        "CE1", "Cosmic Explorer 1", "CosmicExplorerP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "CE2", "Cosmic Explorer 2", "CosmicExplorerP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "CEW", "Cosmic Explorer wideband", "CosmicExplorerWidebandP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "CEP",
        "Cosmic Explorer pessimistic",
        "CosmicExplorerPessimisticP1600143",
        False,
        "nextgen",
    ),
    ObservatorySpec(
        "ET1", "Einstein Telescope 1", "EinsteinTelescopeP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "ET2", "Einstein Telescope 2", "EinsteinTelescopeP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "ET3", "Einstein Telescope 3", "EinsteinTelescopeP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "VOY1", "Voyager-class site 1", "CosmicExplorerPessimisticP1600143", False, "nextgen"
    ),
    ObservatorySpec(
        "VOY2", "Voyager-class site 2", "CosmicExplorerPessimisticP1600143", False, "nextgen"
    ),
    # ---- Space / lunar concepts (simulated design proxies) ----
    ObservatorySpec(
        "LGWA", "Lunar Gravitational-Wave Antenna", "CosmicExplorerPessimisticP1600143", False, "space"
    ),
    ObservatorySpec(
        "DEC1", "DECIGO node 1", "CosmicExplorerWidebandP1600143", False, "space"
    ),
    ObservatorySpec(
        "DEC2", "DECIGO node 2", "CosmicExplorerWidebandP1600143", False, "space"
    ),
    ObservatorySpec(
        "DEC3", "DECIGO node 3", "CosmicExplorerWidebandP1600143", False, "space"
    ),
    ObservatorySpec("TAI1", "Taiji node 1", "CosmicExplorerP1600143", False, "space"),
    ObservatorySpec("TQ1", "TianQin", "CosmicExplorerPessimisticP1600143", False, "space"),
    ObservatorySpec("BBO1", "Big Bang Observer 1", "CosmicExplorerWidebandP1600143", False, "space"),
)

_OBSERVATORY_BY_CODE: Dict[str, ObservatorySpec] = {s.code: s for s in _OBSERVATORY_CATALOG}
DEFAULT_NETWORK_DETECTORS: List[str] = [s.code for s in _OBSERVATORY_CATALOG]
GWOSC_DETECTORS: List[str] = [s.code for s in _OBSERVATORY_CATALOG if s.gwosc]


def _normalize_detector(detector: str) -> str:
    """Return a canonical detector label (e.g. ``H1``)."""
    return str(detector).strip().upper()


def list_observatory_catalog() -> List[ObservatorySpec]:
    """Return the full multi-detector observatory catalog (copy-safe list)."""
    return list(_OBSERVATORY_CATALOG)


def get_observatory_spec(detector: str) -> Optional[ObservatorySpec]:
    """Lookup catalog entry for ``detector``, or ``None`` if unknown."""
    return _OBSERVATORY_BY_CODE.get(_normalize_detector(detector))


@dataclass
class NoiseSegment:
    """A contiguous strain segment with acquisition metadata."""

    strain: np.ndarray
    sample_rate: float
    detector: str
    start_gps: Optional[float]
    duration_seconds: float
    used_fallback: bool
    fallback_reason: Optional[str] = None


@dataclass
class RichProfile:
    """Fixed-length rich noise profile and its named sub-features."""

    vector: np.ndarray
    log_psd: np.ndarray
    freqs: np.ndarray
    window_std: np.ndarray
    window_skewness: np.ndarray
    window_kurtosis: np.ndarray


@dataclass
class TrackerState:
    """Read-only snapshot of LocalNoiseTracker history for diagnostics."""

    history: List[RichProfile]
    drift_deltas: List[float]
    latest_profile: Optional[RichProfile]
    freqs: np.ndarray
    sample_rate: float
    window_size_seconds: float


@dataclass
class HubState:
    """Read-only snapshot of GlobalNoiseHub for network diagnostics."""

    detectors: List[str]
    tracker_states: Dict[str, TrackerState]
    latest_drifts: Dict[str, float]
    profile_length: int


def _psd_name_for_detector(detector: str) -> str:
    """Return the PyCBC PSD model name for ``detector`` (catalog or aLIGO default)."""
    spec = get_observatory_spec(detector)
    if spec is not None:
        return spec.psd_name
    return "aLIGOZeroDetHighPower"


def _psd_model_for_detector(detector: str):
    """Return the PyCBC design-sensitivity PSD callable for ``detector``."""
    from pycbc import psd as pycbc_psd

    return getattr(pycbc_psd, _psd_name_for_detector(detector))


def _supports_gwosc_fetch(detector: str) -> bool:
    """True if this IFO should attempt a live GWOSC open-data download."""
    spec = get_observatory_spec(detector)
    # Unknown codes still try GWOSC (user may pass a valid IFO we have not listed).
    if spec is None:
        return True
    return bool(spec.gwosc)


def _synthesize_colored_noise(
    duration_seconds: float,
    sample_rate: float,
    seed: Optional[int] = None,
    f_lower: float = 20.0,
    detector: str = "H1",
) -> np.ndarray:
    """Generate detector-specific design-sensitivity colored Gaussian noise."""
    from pycbc.noise import noise_from_psd

    length = int(round(duration_seconds * sample_rate))
    if length < 2:
        raise ValueError("duration_seconds * sample_rate must yield at least 2 samples")

    delta_t = 1.0 / sample_rate
    # PyCBC PSD frequency series length must be length//2 + 1 for real FFTs.
    delta_f = 1.0 / (length * delta_t)
    psd_model = _psd_model_for_detector(detector)
    psd = psd_model(length // 2 + 1, delta_f, f_lower)
    ts = noise_from_psd(length, delta_t, psd, seed=seed)
    return np.asarray(ts.numpy(), dtype=np.float64)


def fetch_background_strain(
    detector: str,
    start_gps: float,
    duration_seconds: float = 256.0,
    sample_rate: float = 4096.0,
    seed: Optional[int] = None,
    allow_fallback: bool = True,
) -> NoiseSegment:
    """Fetch real GWOSC strain when available, else detector-specific simulation.

    Catalog sites with ``gwosc=False`` skip the network call and synthesize
    colored noise immediately. GWOSC-capable sites (H1/L1/V1/K1/G1) attempt
    a live download first; on failure they fall back per-detector only.

    After a successful GWOSC fetch, the TimeSeries is explicitly resampled
    to ``sample_rate`` so downstream analytics never depend on the archive's
    native rate alone.
    """
    if duration_seconds <= 0 or sample_rate <= 0:
        raise ValueError("duration_seconds and sample_rate must be positive")

    det_key = _normalize_detector(detector)
    end_gps = float(start_gps) + float(duration_seconds)
    expected_length = int(round(duration_seconds * sample_rate))

    # Planned / space / next-gen sites: no open archive — simulate immediately.
    if not _supports_gwosc_fetch(det_key):
        if not allow_fallback:
            raise RuntimeError(
                f"{det_key} has no GWOSC open-data archive; cannot fetch real strain"
            )
        strain = _synthesize_colored_noise(
            duration_seconds,
            sample_rate,
            seed=seed,
            detector=det_key,
        )
        spec = get_observatory_spec(det_key)
        reason = (
            f"NoGWOSCArchive: {spec.name if spec else det_key} "
            f"uses design PSD {_psd_name_for_detector(det_key)}"
        )
        return NoiseSegment(
            strain=strain,
            sample_rate=float(sample_rate),
            detector=det_key,
            start_gps=float(start_gps),
            duration_seconds=float(duration_seconds),
            used_fallback=True,
            fallback_reason=reason,
        )

    try:
        from gwpy.timeseries import TimeSeries

        ts = TimeSeries.fetch_open_data(det_key, float(start_gps), end_gps)
        # Guarantee the configured sample rate regardless of archive native rate.
        if float(ts.sample_rate.value) != float(sample_rate):
            ts = ts.resample(sample_rate)
        else:
            # Still call resample for an explicit, documented contract.
            ts = ts.resample(sample_rate)

        strain = np.asarray(ts.value, dtype=np.float64)
        if len(strain) != expected_length:
            # Crop or pad tiny length mismatches from resampling / GPS rounding.
            if len(strain) > expected_length:
                strain = strain[:expected_length]
            else:
                padded = np.zeros(expected_length, dtype=np.float64)
                padded[: len(strain)] = strain
                strain = padded

        return NoiseSegment(
            strain=strain,
            sample_rate=float(sample_rate),
            detector=det_key,
            start_gps=float(start_gps),
            duration_seconds=float(duration_seconds),
            used_fallback=False,
            fallback_reason=None,
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        strain = _synthesize_colored_noise(
            duration_seconds,
            sample_rate,
            seed=seed,
            detector=det_key,
        )
        return NoiseSegment(
            strain=strain,
            sample_rate=float(sample_rate),
            detector=det_key,
            start_gps=float(start_gps),
            duration_seconds=float(duration_seconds),
            used_fallback=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def fetch_network_strain(
    detectors: Optional[List[str]] = None,
    start_gps: float = 1240559616.0,
    duration_seconds: float = 256.0,
    sample_rate: float = 4096.0,
    seed: Optional[int] = None,
    allow_fallback: bool = True,
) -> Dict[str, NoiseSegment]:
    """Fetch concurrent strain for a detector network (real-to-sim hybrid).

    Defaults to the full observatory catalog (~25 sites). Each detector is
    fetched independently via ``fetch_background_strain``: GWOSC open-data
    sites try a live download first; all others (and any failed fetch) use
    that site's design-sensitivity colored noise. One site's failure never
    blocks the rest of the network.
    """
    if detectors is None:
        detectors = list(DEFAULT_NETWORK_DETECTORS)
    if not detectors:
        raise ValueError("detectors must be a non-empty list")

    network: Dict[str, NoiseSegment] = {}
    for i, det in enumerate(detectors):
        det_key = _normalize_detector(det)
        det_seed = None if seed is None else int(seed) + i
        network[det_key] = fetch_background_strain(
            det_key,
            start_gps=start_gps,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            seed=det_seed,
            allow_fallback=allow_fallback,
        )

    n_real = sum(1 for seg in network.values() if not seg.used_fallback)
    n_sim = len(network) - n_real
    summary_parts = []
    for det_key, seg in network.items():
        label = "simulated" if seg.used_fallback else "real"
        summary_parts.append(f"{det_key}={label}")
    print(
        f"  network acquisition hybrid ({n_real} real / {n_sim} simulated): "
        + ", ".join(summary_parts),
        flush=True,
    )
    return network


class AdvancedNoiseEncoder:
    """Encode a strain segment into a fixed-length Rich Noise Profile.

    Default geometry for a 256 s / 4096 Hz segment with 4 s windows:
      - 128 log-spaced PSD bins from 20 Hz to Nyquist
      - 64 temporal windows × (std, skewness, excess kurtosis)
      - total vector length = 128 + 64*3 = 320
    """

    PSD_FMIN_HZ = 20.0
    PSD_FLOOR = 1e-60

    def __init__(
        self,
        sample_rate: float = 4096.0,
        expected_duration_seconds: float = 256.0,
        psd_bins: int = 128,
        window_size_seconds: float = 4.0,
    ):
        if sample_rate <= 0 or expected_duration_seconds <= 0:
            raise ValueError("sample_rate and expected_duration_seconds must be positive")
        if window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")
        if expected_duration_seconds < window_size_seconds:
            raise ValueError("expected_duration_seconds must be >= window_size_seconds")

        self.sample_rate = float(sample_rate)
        self.expected_duration_seconds = float(expected_duration_seconds)
        self.psd_bins = int(psd_bins)
        self.window_size_seconds = float(window_size_seconds)

        self.expected_samples = int(round(self.expected_duration_seconds * self.sample_rate))
        self.window_samples = int(round(self.window_size_seconds * self.sample_rate))
        self.n_windows = self.expected_samples // self.window_samples
        if self.n_windows < 1:
            raise ValueError("configuration yields zero temporal windows")

        nyquist = self.sample_rate / 2.0
        if self.PSD_FMIN_HZ >= nyquist:
            raise ValueError("PSD_FMIN_HZ must be below Nyquist")
        self.freqs = np.geomspace(self.PSD_FMIN_HZ, nyquist, self.psd_bins)
        self.vector_length = self.psd_bins + 3 * self.n_windows

    def _normalize_strain(self, strain_data: np.ndarray) -> np.ndarray:
        strain = np.asarray(strain_data, dtype=np.float64).ravel()
        if len(strain) == self.expected_samples:
            return strain
        if len(strain) > self.expected_samples:
            return strain[: self.expected_samples].copy()
        out = np.zeros(self.expected_samples, dtype=np.float64)
        out[: len(strain)] = strain
        return out

    def extract_gaussian_features(self, strain_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch PSD -> log10 PSD at fixed log-spaced frequencies (>= 20 Hz)."""
        strain = self._normalize_strain(strain_data)
        nperseg = min(int(self.sample_rate), len(strain))
        freqs, pxx = welch(
            strain,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )
        # Prefer f > 0 for interpolation source; target grid already starts at 20 Hz.
        mask = freqs > 0
        src_f = freqs[mask]
        src_p = np.maximum(pxx[mask], self.PSD_FLOOR)
        interp = np.interp(self.freqs, src_f, src_p, left=src_p[0], right=src_p[-1])
        log_psd = np.log10(np.maximum(interp, self.PSD_FLOOR))
        return self.freqs.copy(), log_psd.astype(np.float64)

    def extract_non_gaussian_features(
        self,
        strain_data: np.ndarray,
        window_size_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Windowed std / skewness / excess kurtosis over the normalized segment.

        Tracking these moments captures non-stationarity (std drift) and
        non-Gaussianity (skew/kurtosis spikes from glitches) that a 1D PSD
        alone cannot represent.
        """
        strain = self._normalize_strain(strain_data)
        if window_size_seconds is None:
            window_samples = self.window_samples
            n_windows = self.n_windows
        else:
            window_samples = int(round(float(window_size_seconds) * self.sample_rate))
            if window_samples < 2:
                raise ValueError("window_size_seconds too small for sample_rate")
            n_windows = len(strain) // window_samples
            if n_windows < 1:
                raise ValueError("strain too short for requested window_size_seconds")

        stds = np.zeros(n_windows, dtype=np.float64)
        skews = np.zeros(n_windows, dtype=np.float64)
        kurts = np.zeros(n_windows, dtype=np.float64)

        for i in range(n_windows):
            start = i * window_samples
            chunk = strain[start : start + window_samples]
            stds[i] = float(np.std(chunk))
            # fisher=True -> excess kurtosis (0 for a perfect Gaussian).
            skews[i] = float(skew(chunk, bias=False))
            kurts[i] = float(kurtosis(chunk, fisher=True, bias=False))

        for arr in (stds, skews, kurts):
            bad = ~np.isfinite(arr)
            arr[bad] = 0.0

        return {"std": stds, "skewness": skews, "kurtosis": kurts}

    def construct_rich_profile(self, strain_data: np.ndarray) -> RichProfile:
        """Concatenate Gaussian PSD features and non-Gaussian moment vectors."""
        freqs, log_psd = self.extract_gaussian_features(strain_data)
        moments = self.extract_non_gaussian_features(strain_data)
        vector = np.concatenate(
            [log_psd, moments["std"], moments["skewness"], moments["kurtosis"]]
        ).astype(np.float64)
        return RichProfile(
            vector=vector,
            log_psd=log_psd,
            freqs=freqs,
            window_std=moments["std"],
            window_skewness=moments["skewness"],
            window_kurtosis=moments["kurtosis"],
        )


def _taper_edges(signal: np.ndarray, sample_rate: float, taper_seconds: float = 0.05) -> np.ndarray:
    """Apply a short Hann fade-in / fade-out to suppress edge discontinuities."""
    out = np.asarray(signal, dtype=np.float64).copy()
    n_taper = int(round(taper_seconds * sample_rate))
    if n_taper < 2 or 2 * n_taper >= len(out):
        return out
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
    out[:n_taper] *= ramp
    out[-n_taper:] *= ramp[::-1]
    return out


def inject_waveform_into_background(
    background: np.ndarray,
    sample_rate: float,
    m1: float = 30.0,
    m2: float = 25.0,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    f_lower: float = 25.0,
    approximant: str = "IMRPhenomD",
    merger_offset: float = 2.0,
    taper_seconds: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """Inject an IMRPhenomD waveform into a background strain array.

    PyCBC time-domain waveforms are coalescence-centered: sample times are
    relative to merger (near t=0), with the inspiral at negative times. This
    function finds the coalescence sample via ``hp.sample_times`` and places
    that sample at ``merger_offset`` seconds before the end of ``background``.
    """
    from pycbc.waveform import get_td_waveform

    background = np.asarray(background, dtype=np.float64)
    if background.ndim != 1:
        raise ValueError("background must be a 1D strain array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    hp, _ = get_td_waveform(
        approximant=approximant,
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=1.0 / sample_rate,
        f_lower=f_lower,
    )
    times = np.asarray(hp.sample_times.numpy(), dtype=np.float64)
    signal = _taper_edges(hp.numpy(), sample_rate, taper_seconds=taper_seconds)

    # Index of coalescence (sample nearest t=0 in the waveform's own epoch).
    coalescence_idx = int(np.argmin(np.abs(times)))

    n_noise = len(background)
    merger_index = n_noise - int(round(merger_offset * sample_rate))
    if merger_index < 0 or merger_index >= n_noise:
        raise ValueError("merger_offset places coalescence outside the background segment")

    # Place coalescence_idx at merger_index in the background.
    start = merger_index - coalescence_idx
    end = start + len(signal)

    # Clip waveform to background bounds while preserving coalescence alignment.
    sig_start = 0
    sig_end = len(signal)
    if start < 0:
        sig_start = -start
        start = 0
    if end > n_noise:
        sig_end -= end - n_noise
        end = n_noise
    if sig_start >= sig_end or start >= end:
        raise ValueError("waveform has no overlap with the background after clipping")

    injected = background.copy()
    placed = signal[sig_start:sig_end]
    injected[start:end] += placed

    metadata = {
        "m1": m1,
        "m2": m2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "approximant": approximant,
        "f_lower": f_lower,
        "merger_offset": merger_offset,
        "coalescence_idx_in_waveform": coalescence_idx,
        "merger_index_in_background": merger_index,
        "start_idx": start,
        "end_idx": end,
        "n_placed_samples": int(end - start),
        "peak_abs_signal": float(np.max(np.abs(placed))) if len(placed) else 0.0,
    }
    return injected, metadata


class LocalNoiseTracker:
    """Sliding-window history of rich profiles with environmental drift."""

    def __init__(self, encoder: AdvancedNoiseEncoder, history_size: int = 10):
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        self.encoder = encoder
        self.history_size = int(history_size)
        self._history: Deque[RichProfile] = deque(maxlen=self.history_size)
        self._drift_deltas: Deque[float] = deque(maxlen=self.history_size)

    def update_profile(self, new_strain: np.ndarray) -> float:
        """Append a new rich profile and return Euclidean drift vs history mean."""
        profile = self.encoder.construct_rich_profile(new_strain)
        if len(self._history) == 0:
            drift = 0.0
        else:
            mean_vec = np.mean([p.vector for p in self._history], axis=0)
            drift = float(np.linalg.norm(profile.vector - mean_vec))
        self._history.append(profile)
        self._drift_deltas.append(drift)
        return drift

    @property
    def state(self) -> TrackerState:
        return TrackerState(
            history=list(self._history),
            drift_deltas=list(self._drift_deltas),
            latest_profile=self._history[-1] if self._history else None,
            freqs=self.encoder.freqs.copy(),
            sample_rate=self.encoder.sample_rate,
            window_size_seconds=self.encoder.window_size_seconds,
        )


def plot_rich_profile(tracker_state: TrackerState, output_dir: str = "results") -> str:
    """Save a multi-panel vector PDF of PSD + moment drift diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not tracker_state.history:
        raise ValueError("tracker_state.history is empty; call update_profile first")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"rich_noise_profile_{timestamp}.pdf")

    latest = tracker_state.latest_profile
    n_hist = len(tracker_state.history)
    n_windows = len(latest.window_skewness)

    skew_mat = np.vstack([p.window_skewness for p in tracker_state.history])
    kurt_mat = np.vstack([p.window_kurtosis for p in tracker_state.history])

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.semilogx(tracker_state.freqs, latest.log_psd, color="#1f77b4", lw=1.5)
    ax.set_title("Baseline log-PSD (Welch, >= 20 Hz)", fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\log_{10} S(f)$")

    ax = axes[0, 1]
    im = ax.imshow(
        skew_mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        extent=[0, n_windows * tracker_state.window_size_seconds, 0, n_hist],
    )
    ax.set_title("Skewness history (windows × updates)", fontweight="bold")
    ax.set_xlabel("Time in segment [s]")
    ax.set_ylabel("Tracker update index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(
        kurt_mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=[0, n_windows * tracker_state.window_size_seconds, 0, n_hist],
    )
    ax.set_title("Excess kurtosis history (glitch tails)", fontweight="bold")
    ax.set_xlabel("Time in segment [s]")
    ax.set_ylabel("Tracker update index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    deltas = np.asarray(tracker_state.drift_deltas, dtype=np.float64)
    ax.plot(np.arange(len(deltas)), deltas, "o-", color="#d62728", lw=1.5)
    ax.set_title("Environmental drift delta", fontweight="bold")
    ax.set_xlabel("Tracker update index")
    ax.set_ylabel(r"$\|v_t - \langle v \rangle_{\mathrm{hist}}\|_2$")

    fig.suptitle("ADAPT Rich Noise Profile Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


class GlobalNoiseHub:
    """Multi-detector orchestrator: independent local trackers + global vector.

    Defaults to the full observatory catalog (~25 sites). Each detector gets
    its own ``AdvancedNoiseEncoder`` / ``LocalNoiseTracker`` with identical
    feature geometry so concatenated global profiles have a stable length
    ``N * n_detectors``.
    """

    def __init__(
        self,
        detectors: Optional[List[str]] = None,
        *,
        sample_rate: float = 4096.0,
        expected_duration_seconds: float = 256.0,
        history_size: int = 10,
        psd_bins: int = 128,
        window_size_seconds: float = 4.0,
    ):
        if detectors is None:
            detectors = list(DEFAULT_NETWORK_DETECTORS)
        if not detectors:
            raise ValueError("detectors must be a non-empty list")

        # Preserve caller order; normalize labels and reject duplicates.
        ordered: List[str] = []
        seen = set()
        for det in detectors:
            key = _normalize_detector(det)
            if key in seen:
                raise ValueError(f"duplicate detector in hub: {key}")
            seen.add(key)
            ordered.append(key)

        self.detectors = ordered
        self._trackers: Dict[str, LocalNoiseTracker] = {}
        self._encoders: Dict[str, AdvancedNoiseEncoder] = {}
        self._latest_drifts: Dict[str, float] = {d: 0.0 for d in ordered}

        for det in ordered:
            encoder = AdvancedNoiseEncoder(
                sample_rate=sample_rate,
                expected_duration_seconds=expected_duration_seconds,
                psd_bins=psd_bins,
                window_size_seconds=window_size_seconds,
            )
            self._encoders[det] = encoder
            self._trackers[det] = LocalNoiseTracker(encoder, history_size=history_size)

        # Shared single-detector profile length N (identical encoder geometry).
        self._profile_length = self._encoders[ordered[0]].vector_length

    @property
    def profile_length(self) -> int:
        """Length N of one detector's rich profile vector."""
        return self._profile_length

    @property
    def global_profile_length(self) -> int:
        """Expected length of ``get_global_profile()`` = N × n_detectors."""
        return self._profile_length * len(self.detectors)

    def update_network(self, network_strain_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update every local tracker with its strain array; return drift map."""
        if not isinstance(network_strain_dict, dict) or not network_strain_dict:
            raise ValueError("network_strain_dict must be a non-empty dict")

        normalized = {_normalize_detector(k): v for k, v in network_strain_dict.items()}
        missing = [d for d in self.detectors if d not in normalized]
        if missing:
            raise ValueError(f"network_strain_dict missing detectors: {missing}")
        unknown = [k for k in normalized if k not in self._trackers]
        if unknown:
            raise ValueError(f"network_strain_dict has unknown detectors: {unknown}")

        lengths = {k: len(np.asarray(normalized[k]).ravel()) for k in self.detectors}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"all detector streams must share the same length; got {lengths}"
            )

        drifts: Dict[str, float] = {}
        for det in self.detectors:
            drifts[det] = self._trackers[det].update_profile(normalized[det])
            self._latest_drifts[det] = drifts[det]
        return drifts

    def get_global_profile(self) -> np.ndarray:
        """Concatenate latest rich-profile vectors in hub detector order."""
        vectors = []
        for det in self.detectors:
            latest = self._trackers[det].state.latest_profile
            if latest is None:
                raise ValueError(
                    f"detector {det} has no profile yet; call update_network first"
                )
            vectors.append(latest.vector)
        return np.concatenate(vectors).astype(np.float64)

    @property
    def state(self) -> HubState:
        return HubState(
            detectors=list(self.detectors),
            tracker_states={d: self._trackers[d].state for d in self.detectors},
            latest_drifts=dict(self._latest_drifts),
            profile_length=self._profile_length,
        )


def plot_network_diagnostics(hub_state: HubState, output_dir: str = "results") -> str:
    """Stacked environmental-drift panels for all active detectors.

    Writes ``global_network_profile_YYYYMMDD_HHMMSS.pdf`` under ``output_dir``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hub_state.detectors:
        raise ValueError("hub_state.detectors is empty")

    for det in hub_state.detectors:
        ts = hub_state.tracker_states.get(det)
        if ts is None or not ts.history:
            raise ValueError(
                f"detector {det} has empty tracker history; call update_network first"
            )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"global_network_profile_{timestamp}.pdf")

    n_det = len(hub_state.detectors)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, axes = plt.subplots(
        n_det,
        1,
        figsize=(10, max(2.8 * n_det, 4.0)),
        sharex=True,
        squeeze=False,
    )

    for i, det in enumerate(hub_state.detectors):
        ax = axes[i, 0]
        deltas = np.asarray(
            hub_state.tracker_states[det].drift_deltas, dtype=np.float64
        )
        color = colors[i % len(colors)]
        ax.plot(np.arange(len(deltas)), deltas, "o-", color=color, lw=1.5, label=det)
        ax.set_ylabel(det, fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Tracker update index")
    fig.suptitle(
        "Environmental Drift Deltas — Network Audit",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path
