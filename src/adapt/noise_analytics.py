"""Single-detector advanced noise profiling for ADAPT (local encoder layer).

This module is intentionally isolated from the routing / physics stack. It
extracts a "Rich Noise Profile" from real (or fallback synthetic) detector
strain: a Welch PSD for Gaussian/stationary structure, plus windowed
higher-order moments (std, skewness, excess kurtosis) that track
non-Gaussian and non-stationary behaviour such as glitches and slow
environmental drift.

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

# Matplotlib is imported lazily inside plot_rich_profile so offline feature
# extraction does not pay the Agg/font-cache cost until diagnostics are needed.


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


def _synthesize_colored_noise(
    duration_seconds: float,
    sample_rate: float,
    seed: Optional[int] = None,
    f_lower: float = 20.0,
) -> np.ndarray:
    """Generate Advanced LIGO design-sensitivity colored Gaussian noise."""
    from pycbc.noise import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower

    length = int(round(duration_seconds * sample_rate))
    if length < 2:
        raise ValueError("duration_seconds * sample_rate must yield at least 2 samples")

    delta_t = 1.0 / sample_rate
    # PyCBC PSD frequency series length must be length//2 + 1 for real FFTs.
    delta_f = 1.0 / (length * delta_t)
    psd = aLIGOZeroDetHighPower(length // 2 + 1, delta_f, f_lower)
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
    """Fetch real GWOSC strain, or synthesize colored noise on failure.

    After a successful GWOSC fetch, the TimeSeries is explicitly resampled
    to ``sample_rate`` so downstream analytics never depend on the archive's
    native rate alone.
    """
    if duration_seconds <= 0 or sample_rate <= 0:
        raise ValueError("duration_seconds and sample_rate must be positive")

    end_gps = float(start_gps) + float(duration_seconds)
    expected_length = int(round(duration_seconds * sample_rate))

    try:
        from gwpy.timeseries import TimeSeries

        ts = TimeSeries.fetch_open_data(detector, float(start_gps), end_gps)
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
            detector=detector,
            start_gps=float(start_gps),
            duration_seconds=float(duration_seconds),
            used_fallback=False,
            fallback_reason=None,
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        strain = _synthesize_colored_noise(duration_seconds, sample_rate, seed=seed)
        return NoiseSegment(
            strain=strain,
            sample_rate=float(sample_rate),
            detector=detector,
            start_gps=float(start_gps),
            duration_seconds=float(duration_seconds),
            used_fallback=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


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
