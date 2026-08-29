"""Shared broad glitch families and paired clean/glitch FD+STFT conditioning.

Train and eval use the same injector so the residual corrector sees the failure
mode measured at evaluation: corrupted FD strain (+ optional Welch ASD) and
matching STFT crops derived from the same TD realization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

from adapt.stft_context import (
    SPECTROGRAM_ANALYSIS_SECONDS,
    crop_td_to_analysis_window,
    sine_gaussian_glitch,
)

HELD_IN_FAMILIES = (
    "sine_gaussian",
    "broadband_burst",
    "whistle",
    "scattered_light",
    "glitch_train",
)
HELD_OUT_FAMILIES = (
    "ringing",
    "double_blip",
    "narrowband_tone",
)


@dataclass
class GlitchSpec:
    """One concrete glitch realization (possibly multi-detector)."""

    family: str
    detectors: List[str]
    t_rel: float
    severity: float
    params: Dict[str, Any] = field(default_factory=dict)
    asd_policy: str = "stationary"  # or "welch"
    held_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clip_peak(duration: float, t_rel: float, sample_rate: float) -> float:
    # Crop is trigger-centered: trigger at mid-point.
    t_peak = 0.5 * duration + float(t_rel)
    return float(np.clip(t_peak, 0.0, max(duration - 1.0 / sample_rate, 0.0)))


def broadband_burst(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    duration: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian-enveloped white-noise burst."""
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    sigma = max(float(duration) / 2.355, 1.0 / sample_rate)
    env = np.exp(-0.5 * ((t - float(t_peak)) / sigma) ** 2)
    noise = rng.normal(size=n_samples)
    return float(amplitude) * env * noise


def whistle_glitch(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    duration: float,
    f0: float,
    f1: float,
    amplitude: float,
) -> np.ndarray:
    """Linear chirp (whistle) with Tukey envelope."""
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    half = 0.5 * float(duration)
    mask = (t >= t_peak - half) & (t <= t_peak + half)
    out = np.zeros(n_samples, dtype=np.float64)
    if not np.any(mask):
        return out
    tt = t[mask] - (t_peak - half)
    phase = 2.0 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / max(duration, 1e-6) * tt**2)
    n_win = int(mask.sum())
    env = sp_signal.windows.tukey(n_win, alpha=0.5)
    out[mask] = float(amplitude) * env * np.cos(phase)
    return out


def scattered_light_arch(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    duration: float,
    f_min: float,
    f_max: float,
    amplitude: float,
) -> np.ndarray:
    """Arch-like frequency sweep (scattered-light toy model)."""
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    half = 0.5 * float(duration)
    mask = (t >= t_peak - half) & (t <= t_peak + half)
    out = np.zeros(n_samples, dtype=np.float64)
    if not np.any(mask):
        return out
    u = (t[mask] - (t_peak - half)) / max(duration, 1e-6)
    # Parabolic arch in frequency: f_min → f_max → f_min
    f = f_min + (f_max - f_min) * 4.0 * u * (1.0 - u)
    phase = 2.0 * np.pi * np.cumsum(f) / float(sample_rate)
    env = np.sin(np.pi * u) ** 2
    out[mask] = float(amplitude) * env * np.cos(phase)
    return out


def glitch_train(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    n_pulses: int,
    spacing: float,
    f0: float,
    q: float,
    amplitude: float,
) -> np.ndarray:
    """Short train of sine-Gaussian blips."""
    out = np.zeros(n_samples, dtype=np.float64)
    start = t_peak - 0.5 * (n_pulses - 1) * spacing
    for i in range(int(n_pulses)):
        tp = start + i * spacing
        out += sine_gaussian_glitch(
            n_samples, sample_rate, t_peak=tp, f0=f0, q=q, amplitude=amplitude
        )
    return out


def ringing_glitch(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    f0: float,
    decay: float,
    amplitude: float,
) -> np.ndarray:
    """Held-out: causal exponential ringdown after a step."""
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    out = np.zeros(n_samples, dtype=np.float64)
    mask = t >= t_peak
    tt = t[mask] - t_peak
    out[mask] = (
        float(amplitude)
        * np.exp(-tt / max(decay, 1e-3))
        * np.sin(2.0 * np.pi * f0 * tt)
    )
    return out


def double_blip(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    sep: float,
    f0: float,
    q: float,
    amplitude: float,
) -> np.ndarray:
    """Held-out: two nearby sine-Gaussians."""
    return sine_gaussian_glitch(
        n_samples, sample_rate, t_peak=t_peak - 0.5 * sep, f0=f0, q=q, amplitude=amplitude
    ) + sine_gaussian_glitch(
        n_samples, sample_rate, t_peak=t_peak + 0.5 * sep, f0=f0 * 1.15, q=q, amplitude=0.8 * amplitude
    )


def narrowband_tone(
    n_samples: int,
    sample_rate: float,
    *,
    t_peak: float,
    duration: float,
    f0: float,
    amplitude: float,
) -> np.ndarray:
    """Held-out: nearly monochromatic tone burst."""
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    half = 0.5 * float(duration)
    mask = (t >= t_peak - half) & (t <= t_peak + half)
    out = np.zeros(n_samples, dtype=np.float64)
    if not np.any(mask):
        return out
    n_win = int(mask.sum())
    env = sp_signal.windows.tukey(n_win, alpha=0.25)
    out[mask] = float(amplitude) * env * np.cos(2.0 * np.pi * f0 * t[mask])
    return out


def sample_glitch_spec(
    rng: np.random.Generator,
    *,
    detectors: Sequence[str] = ("H1", "L1", "V1"),
    held_out: bool = False,
    severity_range: Tuple[float, float] = (2.0, 12.0),
    curriculum_severity_max: Optional[float] = None,
    asd_policy: Optional[str] = None,
) -> GlitchSpec:
    """Sample a broad glitch family for training or held-out validation."""
    family_pool = HELD_OUT_FAMILIES if held_out else HELD_IN_FAMILIES
    family = str(rng.choice(family_pool))
    sev_hi = float(severity_range[1])
    if curriculum_severity_max is not None:
        sev_hi = min(sev_hi, float(curriculum_severity_max))
    severity = float(rng.uniform(float(severity_range[0]), max(sev_hi, float(severity_range[0]) + 1e-3)))
    t_rel = float(rng.uniform(-2.0, 0.5))

    # Mostly single-detector; occasional coincident H1+L1.
    if float(rng.random()) < 0.15 and len(detectors) >= 2:
        dets = [str(d) for d in rng.choice(["H1", "L1"], size=2, replace=False)]
    else:
        # Prefer H1/L1 (visible in HL image); V1 only ~10%.
        if float(rng.random()) < 0.1 and "V1" in detectors:
            dets = ["V1"]
        else:
            dets = [str(rng.choice([d for d in detectors if d != "V1"] or list(detectors)))]

    if asd_policy is None:
        # Prefer Welch: it is the real-event failure mode (FD poisoned, STFT
        # must still see the glitch via stationary whitening).
        asd_policy = "welch" if float(rng.random()) < 0.75 else "stationary"

    params: Dict[str, Any] = {}
    if family == "sine_gaussian":
        params = {
            "f0": float(rng.uniform(30.0, 300.0)),
            "q": float(rng.uniform(3.0, 15.0)),
        }
    elif family == "broadband_burst":
        params = {"duration": float(rng.uniform(0.02, 0.25))}
    elif family == "whistle":
        f0 = float(rng.uniform(40.0, 200.0))
        params = {
            "duration": float(rng.uniform(0.2, 1.5)),
            "f0": f0,
            "f1": float(rng.uniform(f0 + 20.0, min(500.0, f0 + 250.0))),
        }
    elif family == "scattered_light":
        params = {
            "duration": float(rng.uniform(0.5, 2.5)),
            "f_min": float(rng.uniform(20.0, 40.0)),
            "f_max": float(rng.uniform(60.0, 120.0)),
        }
    elif family == "glitch_train":
        params = {
            "n_pulses": int(rng.integers(2, 6)),
            "spacing": float(rng.uniform(0.05, 0.25)),
            "f0": float(rng.uniform(40.0, 200.0)),
            "q": float(rng.uniform(3.0, 10.0)),
        }
    elif family == "ringing":
        params = {
            "f0": float(rng.uniform(50.0, 250.0)),
            "decay": float(rng.uniform(0.05, 0.4)),
        }
    elif family == "double_blip":
        params = {
            "sep": float(rng.uniform(0.05, 0.3)),
            "f0": float(rng.uniform(40.0, 200.0)),
            "q": float(rng.uniform(3.0, 12.0)),
        }
    elif family == "narrowband_tone":
        params = {
            "duration": float(rng.uniform(0.3, 1.5)),
            "f0": float(rng.uniform(50.0, 300.0)),
        }
    else:
        raise ValueError(f"Unknown glitch family {family}")

    return GlitchSpec(
        family=family,
        detectors=dets,
        t_rel=t_rel,
        severity=severity,
        params=params,
        asd_policy=str(asd_policy),
        held_out=bool(held_out),
    )


def synthesize_glitch_td(
    n_samples: int,
    sample_rate: float,
    spec: GlitchSpec,
    rng: np.random.Generator,
    *,
    rms: float,
) -> np.ndarray:
    """Synthesize a TD glitch waveform for one detector from ``spec``."""
    duration = n_samples / float(sample_rate)
    t_peak = _clip_peak(duration, spec.t_rel, sample_rate)
    amp = float(spec.severity) * max(float(rms), 1e-30)
    p = spec.params
    fam = spec.family
    if fam == "sine_gaussian":
        return sine_gaussian_glitch(
            n_samples, sample_rate, t_peak=t_peak, f0=p["f0"], q=p["q"], amplitude=amp
        )
    if fam == "broadband_burst":
        return broadband_burst(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            duration=p["duration"],
            amplitude=amp,
            rng=rng,
        )
    if fam == "whistle":
        return whistle_glitch(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            duration=p["duration"],
            f0=p["f0"],
            f1=p["f1"],
            amplitude=amp,
        )
    if fam == "scattered_light":
        return scattered_light_arch(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            duration=p["duration"],
            f_min=p["f_min"],
            f_max=p["f_max"],
            amplitude=amp,
        )
    if fam == "glitch_train":
        return glitch_train(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            n_pulses=p["n_pulses"],
            spacing=p["spacing"],
            f0=p["f0"],
            q=p["q"],
            amplitude=amp,
        )
    if fam == "ringing":
        return ringing_glitch(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            f0=p["f0"],
            decay=p["decay"],
            amplitude=amp,
        )
    if fam == "double_blip":
        return double_blip(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            sep=p["sep"],
            f0=p["f0"],
            q=p["q"],
            amplitude=amp,
        )
    if fam == "narrowband_tone":
        return narrowband_tone(
            n_samples,
            sample_rate,
            t_peak=t_peak,
            duration=p["duration"],
            f0=p["f0"],
            amplitude=amp,
        )
    raise ValueError(f"Unknown family {fam}")


def td_to_fd(
    td: np.ndarray,
    sample_rate: float,
    *,
    n_freq: int,
    roll_off: float = 0.4,
) -> np.ndarray:
    """Tukey-windowed rFFT → length ``n_freq`` complex FD (Bilby-like /fs)."""
    x = np.asarray(td, dtype=np.float64).ravel()
    alpha = 2.0 * float(roll_off) * float(sample_rate) / max(len(x), 1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    window = sp_signal.windows.tukey(len(x), alpha=alpha)
    fd = np.fft.rfft(x * window) / float(sample_rate)
    if len(fd) < n_freq:
        fd = np.pad(fd, (0, n_freq - len(fd)))
    else:
        fd = fd[:n_freq]
    return np.asarray(fd, dtype=np.complex128)


def welch_asd_on_grid(
    td: np.ndarray,
    sample_rate: float,
    *,
    n_freq: int,
    delta_f: float,
    f_min: float,
) -> np.ndarray:
    """Welch ASD interpolated onto ``k * delta_f`` for ``k=0..n_freq-1``."""
    nperseg = int(min(len(td), max(256, int(round(sample_rate * 4.0)))))
    freqs, psd = sp_signal.welch(
        np.asarray(td, dtype=np.float64),
        fs=float(sample_rate),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    target_f = np.arange(int(n_freq), dtype=np.float64) * float(delta_f)
    psd_i = np.interp(target_f, freqs, psd, left=psd[0], right=psd[-1])
    asd = np.sqrt(np.maximum(psd_i, 0.0))
    asd[target_f < float(f_min)] = 1.0
    return np.maximum(asd, 1e-30).astype(np.float64)


def apply_glitch_to_td_map(
    td_map: Mapping[str, np.ndarray],
    sample_rate: float,
    spec: GlitchSpec,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return (glitched TD map, per-detector glitch waveforms)."""
    out = {k: np.asarray(v, dtype=np.float64).ravel().copy() for k, v in td_map.items()}
    glitch_waveforms: Dict[str, np.ndarray] = {}
    for det in spec.detectors:
        if det not in out:
            continue
        x = out[det]
        rms = float(np.std(x)) or 1e-22
        g = synthesize_glitch_td(x.size, sample_rate, spec, rng, rms=rms)
        out[det] = x + g
        glitch_waveforms[det] = g
    return out, glitch_waveforms


def corrupt_injection_fd_with_glitch(
    injection_data: Mapping[str, Any],
    *,
    sample_rate: float,
    duration: float,
    spec: GlitchSpec,
    rng: np.random.Generator,
    roll_off: float = 0.4,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """Corrupt FD waveforms (+ optional Welch ASD) using TD glitches on IFFT crops.

    Returns ``(glitched_injection, glitched_td_map_unwhitened, meta)``.
    STFT crops are trigger-centered analysis windows from the same TD.
    """
    waveforms = injection_data["waveform"]
    asds = injection_data["asds"]
    params = injection_data.get("parameters") or {}
    geocent = float(params.get("geocent_time", 0.0))
    detectors = list(waveforms.keys())

    # Unwhitened TD full-segment for glitch injection (IFFT of FD).
    n_td = int(round(float(duration) * float(sample_rate)))
    n_rfft = n_td // 2 + 1
    td_full: Dict[str, np.ndarray] = {}
    for det in detectors:
        fd = np.asarray(waveforms[det], dtype=np.complex128).ravel()
        fd_full = np.zeros(n_rfft, dtype=np.complex128)
        n_copy = min(fd.size, n_rfft)
        fd_full[:n_copy] = fd[:n_copy]
        td = np.fft.irfft(fd_full, n=n_td) * float(sample_rate)
        trig = float(params.get(f"{det}_time", geocent))
        trig_idx = int(round(trig * sample_rate)) % n_td
        td_full[det] = np.roll(td, n_td // 2 - trig_idx)

    glitched_full, glitch_waves = apply_glitch_to_td_map(
        td_full, sample_rate, spec, rng
    )

    # Rebuild FD from glitched TD (same length as original FD).
    out_wave: Dict[str, Any] = {}
    out_asd: Dict[str, Any] = {}
    n_freq = len(np.asarray(next(iter(waveforms.values()))))
    delta_f = 1.0 / float(duration)
    # Infer f_min from first nonzero-ish ASD bin if available.
    f_min = 20.0
    for det in detectors:
        asd0 = np.asarray(asds[det], dtype=np.float64)
        nz = np.where(asd0 > 1.0 + 1e-12)[0]
        if nz.size:
            f_min = float(nz[0] * delta_f)
            break

    glitched_dets = set(glitch_waves.keys())
    for det in detectors:
        # Keep original FD for detectors without a glitch. Round-tripping every
        # IFO through TD↔FD dumps leakage into design-ASD null bins and, after
        # whitening, detonates L1/V1 STFTs even when only H1 was glitched.
        if det not in glitched_dets:
            out_wave[det] = np.asarray(waveforms[det], dtype=np.complex128).copy()
            out_asd[det] = np.asarray(asds[det], dtype=np.float64).copy()
            continue
        td = glitched_full[det]
        # Roll back so FD time convention matches original (undo center roll).
        trig = float(params.get(f"{det}_time", geocent))
        trig_idx = int(round(trig * sample_rate)) % n_td
        td_unrolled = np.roll(td, trig_idx - n_td // 2)
        fd_new = td_to_fd(td_unrolled, sample_rate, n_freq=n_freq, roll_off=roll_off)
        out_wave[det] = fd_new
        if spec.asd_policy == "welch":
            out_asd[det] = welch_asd_on_grid(
                td_unrolled,
                sample_rate,
                n_freq=n_freq,
                delta_f=delta_f,
                f_min=f_min,
            )
        else:
            out_asd[det] = np.asarray(asds[det], dtype=np.float64).copy()

    # Analysis-window crops (still trigger-centered after roll in td_full).
    td_crop = {
        det: crop_td_to_analysis_window(glitched_full[det], sample_rate)
        for det in detectors
    }
    glitched_inj = {
        **dict(injection_data),
        "waveform": out_wave,
        "asds": out_asd,
    }
    meta = {
        **spec.to_dict(),
        "f_min": f_min,
        "delta_f": delta_f,
        "n_freq": n_freq,
    }
    return glitched_inj, td_crop, meta


def make_fixed_eval_glitch(
    *,
    det: str = "H1",
    family: str = "sine_gaussian",
    severity: float = 8.0,
    t_rel: float = -1.0,
    asd_policy: str = "welch",
    **params: Any,
) -> GlitchSpec:
    """Canonical eval glitch (H1 sine-Gaussian by default)."""
    if family == "sine_gaussian" and not params:
        params = {"f0": 100.0, "q": 5.0}
    return GlitchSpec(
        family=family,
        detectors=[det],
        t_rel=float(t_rel),
        severity=float(severity),
        params=dict(params),
        asd_policy=asd_policy,
        held_out=False,
    )


def sample_glitch_spec_with_hard_eval(
    rng: np.random.Generator,
    *,
    detectors: Sequence[str] = ("H1", "L1", "V1"),
    held_out: bool = False,
    curriculum_severity_max: Optional[float] = None,
    hard_eval_frac: float = 0.25,
) -> GlitchSpec:
    """Sample glitches, with a fraction locked to the GW170817-style H1 SG.

    The hard eval twin matches ``event_glitch_io.py``: H1
    sine-Gaussian near ``t_rel=-1``, Welch ASD on FD, severity in a loud band.
    """
    if (not held_out) and float(rng.random()) < float(hard_eval_frac):
        sev_hi = 12.0
        if curriculum_severity_max is not None:
            sev_hi = min(sev_hi, float(curriculum_severity_max))
        sev_lo = min(6.0, sev_hi)
        return make_fixed_eval_glitch(
            det="H1",
            family="sine_gaussian",
            severity=float(rng.uniform(sev_lo, max(sev_lo + 1e-3, sev_hi))),
            t_rel=float(rng.uniform(-1.2, -0.8)),
            asd_policy="welch",
            f0=float(rng.uniform(80.0, 120.0)),
            q=float(rng.uniform(4.0, 8.0)),
        )
    return sample_glitch_spec(
        rng,
        detectors=detectors,
        held_out=held_out,
        curriculum_severity_max=curriculum_severity_max,
    )


# STFT must be whitened with the *clean/stationary* ASD even when FD packaging
# uses a Welch-contaminated ASD. Otherwise the glitch is absorbed into the ASD
# and the corrector never sees the transient (the GW170817 failure mode).
STFT_WHITEN_ASD_POLICY = "stationary_clean"


def stft_whitening_asds(
    clean_asds: Mapping[str, Any],
    glitch_asds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """ASDs used to whiten TD before STFT (always clean/stationary)."""
    del glitch_asds  # intentionally unused — never whiten STFT with Welch ASD
    return {k: np.asarray(v, dtype=np.float64) for k, v in clean_asds.items()}
