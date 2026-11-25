"""Gated Gaussian likelihood for importance sampling a frozen DINGO-BNS proposal.

The standard (Whittle) log-likelihood in DINGO's whitened conventions is

    log L(theta) = -1/2 * sum_k |d_w,k - mu_w,k(theta)|^2 ,

with d_w, mu_w the whitened frequency-domain data and signal on the uniform
frequency grid of the event package (DINGO adds a theta-independent constant).
Write this as -1/2 * r^T P r for the time-domain residual r and the (band-limited,
circulant) noise precision P. Treating the samples g inside a gate as missing
(equivalently: marginalising a glitch with a flat prior supported in the gate)
gives the Schur-complement form

    log L_gated(theta) = log L(theta) + 1/2 * u_g^T (P_gg)^+ u_g ,   u = P r ,

which is exactly independent of the data inside the gate. This is the same
quantity as gating-and-inpainting (Zackay et al. 2021; PyCBC GatedGaussianNoise),
written as a correction to DINGO's own likelihood.

Conventions handled here (the source of the earlier matched-delta bug):
  * the package is S F[x], S = exp(-2 pi i f time_buffer) (DINGO data_to_domain);
    we undo S before going to the time domain, so gate times are segment times
    (t = 0 at the first sample of the T-second analysis segment);
  * whitening by asd * noise_std is taken from the base likelihood object.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

DEFAULT_EPS = 1e-3


class GateOperator:
    """Precomputed pieces of the gate correction for one detector.

    Regularisation. The package only holds data in the analysis band, so a glitch's
    content outside the band is unconstrained and the flat-prior marginal is
    ill-conditioned (it slowly absorbs more and more noise as the out-of-band weight
    goes to zero). We therefore model the out-of-band residual as zero with a weak
    precision ``eps`` times the median in-band precision. Likelihood *differences*
    between parameter values are insensitive to ``eps`` over 1e-1 ... 1e-6 (checked
    in tests and reported by the Stage 2 driver), which is all importance sampling uses.

    Parameters
    ----------
    asd : ASD on the package grid (length n_freq, f_k = k / T). Placeholder values
        (>= 1e-6, e.g. DINGO's 1.0 below f_min) mark bins outside the band.
    noise_std : DINGO domain noise_std (whitened = strain / (asd * noise_std)).
    sample_rate, duration : time-domain sampling of the analysis segment (N = T fs).
    time_buffer : the cyclic shift DINGO applied when packaging (seconds).
    gate_mask : boolean array of length N, True for gated (missing) samples.
    eps : out-of-band regularisation (see above).
    """

    def __init__(
        self,
        asd: np.ndarray,
        noise_std: float,
        sample_rate: float,
        duration: float,
        time_buffer: float,
        gate_mask: np.ndarray,
        *,
        eps: float = DEFAULT_EPS,
        band: Optional[np.ndarray] = None,
    ):
        from scipy.linalg import cho_factor

        self.fs = float(sample_rate)
        self.T = float(duration)
        self.N = int(round(self.T * self.fs))
        if self.N % 2:
            raise ValueError("segment length must be even")
        a = np.asarray(asd, dtype=np.float64).ravel()
        self.n_freq = a.size
        n_rfft = self.N // 2 + 1
        if self.n_freq > n_rfft:
            raise ValueError("ASD longer than the rfft grid of the segment")
        if band is None:
            band = (a > 0.0) & (a < 1e-6)
        band = np.asarray(band, dtype=bool).copy()
        band[0] = False
        if self.n_freq == n_rfft:
            band[-1] = False  # never use the Nyquist bin
        self.band = band
        a_w = a * float(noise_std)  # whitening factor: d_w = d / a_w
        scale = float(np.median(a_w[band]))
        self.a = np.where(band, a_w / scale, 1.0)  # normalised; the correction is scale free
        f = np.arange(self.n_freq) / self.T
        self.unshift = np.exp(2j * np.pi * f * float(time_buffer))
        self.eps = float(eps)
        gm = np.asarray(gate_mask, dtype=bool).ravel()
        if gm.size != self.N:
            raise ValueError(f"gate mask has {gm.size} samples, expected {self.N}")
        self.idx = np.flatnonzero(gm)
        self.M = int(self.idx.size)
        if self.M == 0:
            self.chol = None
            return
        c_full = np.zeros(n_rfft)
        c_in = 1.0 / self.a[band] ** 2
        c_full[1:-1] = self.eps * float(np.median(c_in))  # weak out-of-band precision
        c_full[: self.n_freq][band] = c_in
        c_full[0] = 0.0
        c_full[-1] = 0.0
        kernel = np.fft.irfft(c_full, n=self.N)  # circulant precision (normalised units)
        lag = np.abs(self.idx[:, None] - self.idx[None, :])
        lag = np.minimum(lag, self.N - lag)
        P_gg = kernel[lag]
        self.chol = cho_factor(P_gg, lower=True, check_finite=False)
        d = np.diag(self.chol[0])
        self.cond_est = float((d.max() / d.min()) ** 2)

    def u_tilde(self, r_w: np.ndarray) -> np.ndarray:
        """Time-domain u~ = irfft(r_w / a) (band only, shift undone), full length N."""
        R = np.zeros(self.N // 2 + 1, dtype=np.complex128)
        rw = np.asarray(r_w, dtype=np.complex128)[: self.n_freq]
        R[: self.n_freq] = np.where(self.band, rw / self.a, 0.0) * self.unshift
        return np.fft.irfft(R, n=self.N)

    def correction_from_ug(self, u_g: np.ndarray) -> float:
        if self.M == 0:
            return 0.0
        from scipy.linalg import solve_triangular

        y = solve_triangular(self.chol[0], np.asarray(u_g, dtype=np.float64), lower=True, check_finite=False)
        # Factor N/4 = (1/2) * (N/2); the N/2 converts numpy's irfft normalisation of
        # u~ and of the kernel into the Whittle quadratic form (brute-force tested).
        return 0.25 * self.N * float(y @ y)

    def correction(self, r_w: np.ndarray) -> float:
        if self.M == 0:
            return 0.0
        return self.correction_from_ug(self.u_tilde(r_w)[self.idx])


def whittle_log_likelihood(d_w: Mapping[str, np.ndarray], mu_w: Mapping[str, np.ndarray]) -> float:
    """DINGO's log L without the theta-independent constant: -1/2 sum |d_w - mu_w|^2."""
    return float(-0.5 * sum(np.sum(np.abs(np.asarray(d_w[k]) - np.asarray(mu_w[k])) ** 2) for k in d_w))


def gate_mask_from_windows(
    windows: Sequence, detector: str, n_samples: int, sample_rate: float, *, alpha: Optional[float] = None
) -> np.ndarray:
    """True wherever the repo's Tukey gate weight is below 1 (whole window, tapers included)."""
    from adapt.glitch_excision import DEFAULT_TUKEY_ALPHA, tukey_gate_weights

    w = tukey_gate_weights(
        int(n_samples),
        float(sample_rate),
        [g for g in windows if getattr(g, "detector", None) == detector],
        t0=0.0,
        alpha=DEFAULT_TUKEY_ALPHA if alpha is None else float(alpha),
    )
    m = w < 1.0 - 1e-12
    # include the window edges themselves (weight exactly 1 at the Tukey endpoints)
    for g in windows:
        if getattr(g, "detector", None) != detector:
            continue
        i0 = max(0, int(np.floor(float(g.t_start) * sample_rate)))
        i1 = min(int(n_samples), int(np.ceil(float(g.t_end) * sample_rate)))
        m[i0:i1] = True
    return m


# ---------------------------------------------------------------------------
# DINGO integration
# ---------------------------------------------------------------------------


class GatedLikelihood:
    """Gated log-likelihood on top of a uniform-frequency DINGO likelihood.

    ``base`` must be a ``StationaryGaussianGWLikelihood`` built WITHOUT decimation
    and WITHOUT phase heterodyning (use :func:`build_ufd_likelihood`), so that its
    ``whitened_strains`` and ``signal(theta)['waveform']`` live on the uniform grid.
    """

    def __init__(
        self,
        base,
        gate_masks: Mapping[str, np.ndarray],
        *,
        sample_rate: float,
        duration: float,
        time_buffer: float,
        eps: float = DEFAULT_EPS,
    ):
        self.base = base
        self.d_w = {k: np.asarray(v, dtype=np.complex128) for k, v in base.whitened_strains.items()}
        ns = float(base.data_domain.noise_std)
        self.ops: Dict[str, GateOperator] = {}
        for ifo, mask in gate_masks.items():
            if ifo not in self.d_w or not np.any(mask):
                continue
            self.ops[ifo] = GateOperator(
                np.asarray(base.asd[ifo]), ns, sample_rate, duration, time_buffer, mask, eps=eps
            )

    def components(self, theta: dict):
        mu = self.base.signal(theta)["waveform"]
        L0 = whittle_log_likelihood(self.d_w, mu)
        corr = 0.0
        for ifo, op in self.ops.items():
            corr += op.correction(self.d_w[ifo] - np.asarray(mu[ifo]))
        return L0, corr

    def log_likelihood(self, theta: dict) -> float:
        L0, corr = self.components(theta)
        return L0 + corr


_WORKER: dict = {}


def _worker_init(payload):
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _WORKER["lik"] = payload


def _worker_eval(theta):
    try:
        return _WORKER["lik"].components(theta)
    except Exception:
        return (float("nan"), float("nan"))


def evaluate_components(lik: GatedLikelihood, theta_df, num_processes: int = 1):
    """(L0, correction) for every row of ``theta_df``; the likelihood is sent once per worker."""
    rows = [r.to_dict() for _, r in theta_df.iterrows()]
    if int(num_processes) <= 1:
        out = []
        for th in rows:
            try:
                out.append(lik.components(th))
            except Exception:
                out.append((float("nan"), float("nan")))
    else:
        import multiprocessing as mp

        with mp.Pool(int(num_processes), initializer=_worker_init, initargs=(lik,)) as pool:
            out = pool.map(_worker_eval, rows, chunksize=max(1, len(rows) // (8 * int(num_processes))))
    arr = np.asarray(out, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def build_ufd_likelihood(result):
    """Uniform-frequency, non-heterodyned likelihood on the result's own event data."""
    result._build_likelihood(decimate=False, phase_heterodyning=False)
    lik = result.likelihood
    n = len(next(iter(lik.whitened_strains.values())))
    n_pkg = len(next(iter(result.context["waveform"].values())))
    if n != n_pkg:
        raise RuntimeError(f"likelihood grid has {n} bins but the event package has {n_pkg}")
    return lik


def theta_frame(result):
    """The parameter columns DINGO itself passes to the likelihood."""
    from bilby.core.prior import Constraint

    keys = [k for k, v in result.prior.items() if not isinstance(v, Constraint)]
    return result.samples[keys]


def reweight(samples, log_likelihood_new: np.ndarray) -> dict:
    """Importance weights for a new target from DINGO's stored proposal/prior columns."""
    s = samples
    for col in ("log_prob", "log_prior"):
        if col not in s.columns:
            raise KeyError(f"samples lack {col}; run DINGO importance sampling first")
    log_w = s["log_prior"].to_numpy(float) + np.asarray(log_likelihood_new, float) - s["log_prob"].to_numpy(float)
    if "delta_log_prob_target" in s.columns:
        log_w = log_w + s["delta_log_prob_target"].to_numpy(float)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    n = log_w.size
    if not np.any(np.isfinite(log_w)):
        return {"weights": np.zeros(n), "n_eff": 0.0, "efficiency": 0.0, "log_evidence": float("-inf")}
    m = np.max(log_w)
    w = np.exp(log_w - m)
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    log_z = float(m + np.log(np.mean(w)))
    w = w / np.mean(w)
    return {"weights": w, "n_eff": n_eff, "efficiency": n_eff / n, "log_evidence": log_z}


def env_eps() -> float:
    return float(os.environ.get("ADAPT_GATED_EPS", DEFAULT_EPS))
