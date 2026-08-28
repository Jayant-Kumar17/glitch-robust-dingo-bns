#!/usr/bin/env python3
"""Native unmodified DINGO-T1 baseline verification.

Loads pristine ``models_checkpoint/dingo_t1.pt`` via the official ``dingo-t1``
branch APIs (``build_model_from_kwargs`` + ``GWSampler``). No Cross-Docking,
no ADAPT ``DingoT1Network``, and the checkpoint file is never modified.

Track A — synthetic IMRPhenomXPHM injection with known parameters.
Track B — real GW150914 H1/L1 GWOSC strain with native Tukey/MFD prep.

Writes:
  results/baseline_verification_synthetic.png
  results/baseline_verification_gw150914.png
  results/baseline_verification_{YYYYMMDD_HHMMSS}.pdf
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from pycbc.psd import AdvVirgo, aLIGOZeroDetHighPower

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.domains import build_domain_from_model_metadata
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.injection import Injection

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "models_checkpoint" / "dingo_t1.pt"
RESULTS_DIR = REPO_ROOT / "results"

# Synthetic injection (detector-frame) truth.
SYNTH_M1 = 35.0
SYNTH_M2 = 30.0
SYNTH_DL = 500.0
SYNTH_THETA_JN = 0.4
SYNTH_RA = 1.0
SYNTH_DEC = -0.5
SYNTH_PSI = 0.5

# GW150914 (GWTC-1 published medians; source-frame masses / luminosity distance).
GW150914_GPS = 1126259462.4
GW150914_PUBLISHED = {
    "mass_1": 35.6,
    "mass_2": 30.6,
    "luminosity_distance": 410.0,
    "chirp_mass": None,  # filled below
    "mass_ratio": None,
}
_mc_pub = (35.6 * 30.6) ** 0.6 / (35.6 + 30.6) ** 0.2
_q_pub = 30.6 / 35.6
GW150914_PUBLISHED["chirp_mass"] = float(_mc_pub)
GW150914_PUBLISHED["mass_ratio"] = float(_q_pub)

MC_RAE_PASS = 0.05  # 5%
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 500
DEFAULT_TIME_BUFFER = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chirp_mass_q_from_m1_m2(m1: float, m2: float) -> Tuple[float, float]:
    q = m2 / m1
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return float(mc), float(q)


def m1_m2_from_chirp_mass_q(mc: float, q: float) -> Tuple[float, float]:
    """Convert chirp mass + mass ratio (q=m2/m1 <= 1) → component masses."""
    m1 = mc * ((1.0 + q) ** 0.2) / (q**0.6)
    m2 = q * m1
    return float(m1), float(m2)


def add_component_masses(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    m1s, m2s = [], []
    for mc, q in zip(out["chirp_mass"].to_numpy(), out["mass_ratio"].to_numpy()):
        m1, m2 = m1_m2_from_chirp_mass_q(float(mc), float(q))
        m1s.append(m1)
        m2s.append(m2)
    out["mass_1"] = np.asarray(m1s, dtype=np.float64)
    out["mass_2"] = np.asarray(m2s, dtype=np.float64)
    return out


def relative_abs_error(true: float, est: float) -> float:
    if true == 0.0:
        return float("inf")
    return abs(est - true) / abs(true)


def credible_interval(
    values: np.ndarray, level: float = 0.9
) -> Tuple[float, float, float]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    q_lo, q_med, q_hi = np.quantile(values, [lo, 0.5, hi])
    return float(q_lo), float(q_med), float(q_hi)


def summarize_samples(
    samples: pd.DataFrame, keys: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        if k not in samples.columns:
            continue
        lo, med, hi = credible_interval(samples[k].to_numpy(dtype=np.float64))
        summary[k] = {"q05": lo, "median": med, "q95": hi}
    return summary


def design_asd_on_domain(det: str, base_domain, mfd_domain) -> np.ndarray:
    """Stationary design ASD on MFD (decimated from base UFD)."""
    n = len(base_domain)
    delta_f = float(base_domain.delta_f)
    f_min = float(base_domain.f_min)
    if det == "V1":
        psd = AdvVirgo(n, delta_f, f_min)
    else:
        psd = aLIGOZeroDetHighPower(n, delta_f, f_min)
    asd = np.sqrt(np.asarray(psd, dtype=np.float64))
    asd = base_domain.update_data(asd, low_value=1.0)
    if hasattr(mfd_domain, "decimate"):
        asd = mfd_domain.decimate(asd)
    return np.asarray(asd, dtype=np.float64)


def load_native_model(ckpt: Path, device: str = "cpu"):
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    logger.info("Loading native model from %s on %s", ckpt, device)
    pm = build_model_from_kwargs(filename=str(ckpt), device=device, print_output=True)
    # Sanity: full network present (embedding + flow).
    n_params = sum(p.numel() for p in pm.network.parameters())
    logger.info(
        "Loaded %s epoch=%s network=%s n_params=%d",
        type(pm).__name__,
        pm.epoch,
        type(pm.network).__name__,
        n_params,
    )
    return pm


def run_sampler(
    pm,
    context: Dict[str, Any],
    *,
    num_samples: int,
    batch_size: int,
    detectors: Optional[List[str]] = None,
    event_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    sampler = GWSampler(model=pm)
    if detectors is not None:
        sampler.detectors = list(detectors)
    if event_metadata is not None:
        sampler.event_metadata = dict(event_metadata)
    sampler.context = context
    logger.info(
        "Sampling n=%d batch=%d detectors=%s",
        num_samples,
        batch_size,
        sampler.detectors,
    )
    with torch.no_grad():
        sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
    samples = add_component_masses(sampler.samples)
    return samples


# ---------------------------------------------------------------------------
# Track A — synthetic
# ---------------------------------------------------------------------------


def build_synthetic_context(pm, seed: int = 7) -> Tuple[Dict[str, Any], Dict[str, float]]:
    meta = pm.metadata
    mfd = build_domain_from_model_metadata(meta, base=False)
    base = build_domain_from_model_metadata(meta, base=True)

    mc, q = chirp_mass_q_from_m1_m2(SYNTH_M1, SYNTH_M2)
    truth = {
        "mass_1": SYNTH_M1,
        "mass_2": SYNTH_M2,
        "chirp_mass": mc,
        "mass_ratio": q,
        "luminosity_distance": SYNTH_DL,
        "theta_jn": SYNTH_THETA_JN,
        "ra": SYNTH_RA,
        "dec": SYNTH_DEC,
        "psi": SYNTH_PSI,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "geocent_time": 0.0,
        "phase": 0.0,
    }

    rng = np.random.default_rng(seed)
    # Slight sky/time jitter not needed — fixed truth for recovery test.
    _ = rng  # silence unused if we keep fixed theta

    inj = Injection.from_posterior_model_metadata(meta)
    detectors = list(meta["train_settings"]["data"]["detectors"])
    inj.asd = {det: design_asd_on_domain(det, base, mfd) for det in detectors}
    context = inj.injection(truth)
    return context, truth


def run_track_a(
    pm, *, num_samples: int, batch_size: int, seed: int
) -> Dict[str, Any]:
    print("\n" + "=" * 72, flush=True)
    print("TRACK A — Synthetic IMRPhenomXPHM injection", flush=True)
    print("=" * 72, flush=True)
    context, truth = build_synthetic_context(pm, seed=seed)
    samples = run_sampler(pm, context, num_samples=num_samples, batch_size=batch_size)

    keys = ["chirp_mass", "mass_ratio", "mass_1", "mass_2", "luminosity_distance"]
    summary = summarize_samples(samples, keys)

    print(
        f"{'Parameter':<22} {'True':>10} {'Median':>10} {'RAE':>10} {'90% CI':>24}",
        flush=True,
    )
    print("-" * 80, flush=True)
    rae: Dict[str, float] = {}
    for k in keys:
        true_v = float(truth[k])
        med = summary[k]["median"]
        err = relative_abs_error(true_v, med)
        rae[k] = err
        ci = f"[{summary[k]['q05']:.4g}, {summary[k]['q95']:.4g}]"
        print(
            f"{k:<22} {true_v:10.4f} {med:10.4f} {err:10.4%} {ci:>24}",
            flush=True,
        )

    mc_ok = rae["chirp_mass"] < MC_RAE_PASS
    status = "PASS" if mc_ok else "FAIL"
    print(
        f"\nChirp-mass RAE = {rae['chirp_mass']:.4%} "
        f"(threshold {MC_RAE_PASS:.0%}) → {status}",
        flush=True,
    )
    return {
        "track": "A",
        "name": "synthetic",
        "truth": truth,
        "samples": samples,
        "summary": summary,
        "rae": rae,
        "mc_rae_pass": mc_ok,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Track B — GW150914
# ---------------------------------------------------------------------------


def _read_hdf5_strain(path: Path) -> Tuple[np.ndarray, float, float]:
    """Return (strain, sample_rate_hz, gps_start) from a GWOSC event HDF5."""
    import h5py

    with h5py.File(path, "r") as f:
        ds = f["strain/Strain"]
        strain = np.asarray(ds[()], dtype=np.float64)
        dx = float(ds.attrs["Xspacing"])
        gps_start = float(ds.attrs["Xstart"])
        sample_rate = 1.0 / dx
    return strain, sample_rate, gps_start


def _td_window_to_fd(
    strain_td: np.ndarray,
    *,
    sample_rate: float,
    window_dict: Dict[str, Any],
    time_buffer: float,
) -> np.ndarray:
    """Tukey-window + FFT + cyclic shift so coalescence sits at t=0."""
    from dingo.gw.gwutils import get_window

    window = get_window(window_dict)
    if len(window) != len(strain_td):
        raise ValueError(
            f"window length {len(window)} != strain length {len(strain_td)}"
        )
    # Use pycbc FrequencySeries path consistent with dingo download helpers.
    from pycbc.types import TimeSeries as PycbcTimeSeries

    ts = PycbcTimeSeries(strain_td * window, delta_t=1.0 / sample_rate)
    fs = ts.to_frequencyseries()
    fs = fs.cyclic_time_shift(time_buffer)
    return np.asarray(fs, dtype=np.complex128)


def _welch_asd_from_td(
    strain_td: np.ndarray,
    *,
    sample_rate: float,
    segment_seconds: float,
    window_dict: Dict[str, Any],
) -> np.ndarray:
    """Median Welch ASD over non-overlapping Tukey segments inside ``strain_td``."""
    from dingo.gw.gwutils import get_window
    import pycbc.psd
    from pycbc.types import TimeSeries as PycbcTimeSeries

    seg_len = int(round(segment_seconds * sample_rate))
    window = get_window(window_dict)
    if len(window) != seg_len:
        # Rebuild window for the segment length we actually have.
        wcfg = dict(window_dict)
        wcfg["T"] = float(seg_len / sample_rate)
        window = get_window(wcfg)

    n_seg = len(strain_td) // seg_len
    if n_seg < 1:
        raise ValueError("strain too short for Welch ASD")

    cropped = strain_td[: n_seg * seg_len]
    ts = PycbcTimeSeries(cropped, delta_t=1.0 / sample_rate)
    psd = pycbc.psd.estimate.welch(
        ts,
        seg_len=seg_len,
        seg_stride=seg_len,
        window=window,
        avg_method="median",
    )
    return np.sqrt(np.asarray(psd, dtype=np.float64))


def build_gw150914_context(
    pm,
    time_buffer: float = DEFAULT_TIME_BUFFER,
    num_segments_psd: int = 32,  # unused; kept for CLI compatibility
) -> Dict[str, Any]:
    """Build GWSampler context from small GWOSC event HDF5 cuts (not the 4096s archive)."""
    del num_segments_psd  # CLI flag retained; event cuts are only ~32 s.
    from adapt.gwosc_events import fetch_event_strain

    meta = pm.metadata
    window = dict(meta["train_settings"]["data"]["window"])
    base = build_domain_from_model_metadata(meta, base=True)
    detectors = ["H1", "L1"]
    T = float(window["T"])
    f_s = float(window["f_s"])
    n_analysis = int(round(T * f_s))

    cache_dir = REPO_ROOT / "data" / "gwosc" / "baseline_gw150914"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Fetching GW150914 32s event HDF5 (H1/L1) via GWOSC event API; "
        f"analysis window T={T}s buffer={time_buffer}s f_s={f_s}",
        flush=True,
    )

    context: Dict[str, Any] = {"waveform": {}, "asds": {}}
    for det in detectors:
        dest = cache_dir / f"GW150914_{det}_32s_4096Hz.hdf5"
        if not dest.is_file():
            fetch_event_strain(
                "GW150914",
                "GWTC-1-confident",
                det,
                version=3,
                sample_rate=4096,
                duration=32,
                dest_path=str(dest),
            )
        else:
            print(f"  Using cached {dest.name}", flush=True)

        strain, sr, gps_start = _read_hdf5_strain(dest)
        if abs(sr - f_s) > 1e-6:
            raise RuntimeError(f"{det}: sample rate {sr} != expected {f_s}")

        # Analysis segment: [GPS + buffer - T, GPS + buffer)
        t0 = float(GW150914_GPS) + float(time_buffer) - T
        i0 = int(round((t0 - gps_start) * sr))
        i1 = i0 + n_analysis
        if i0 < 0 or i1 > len(strain):
            raise RuntimeError(
                f"{det}: analysis window [{i0}:{i1}] outside strain length {len(strain)} "
                f"(gps_start={gps_start})"
            )
        segment = strain[i0:i1]
        fd = _td_window_to_fd(
            segment, sample_rate=sr, window_dict=window, time_buffer=float(time_buffer)
        )
        context["waveform"][det] = base.update_data(fd)

        # Prefer Welch ASD from the full 32s cut; fall back to design ASD.
        try:
            asd_raw = _welch_asd_from_td(
                strain, sample_rate=sr, segment_seconds=T, window_dict=window
            )
            asd = base.update_data(asd_raw, low_value=1.0)
            asd_src = "welch_event_hdf5"
        except Exception as exc:
            logger.warning("%s Welch ASD failed (%s); using design ASD", det, exc)
            n = len(base)
            delta_f = float(base.delta_f)
            psd = aLIGOZeroDetHighPower(n, delta_f, float(base.f_min))
            asd = base.update_data(
                np.sqrt(np.asarray(psd, dtype=np.float64)), low_value=1.0
            )
            asd_src = "design_aligo"

        context["asds"][det] = asd
        logger.info(
            "%s waveform len=%d asd len=%d source=%s",
            det,
            len(context["waveform"][det]),
            len(context["asds"][det]),
            asd_src,
        )
        print(
            f"  {det}: waveform={len(context['waveform'][det])} "
            f"asd={len(context['asds'][det])} ({asd_src})",
            flush=True,
        )

    return context


def run_track_b(
    pm,
    *,
    num_samples: int,
    batch_size: int,
    time_buffer: float,
    num_segments_psd: int = 32,
) -> Dict[str, Any]:
    print("\n" + "=" * 72, flush=True)
    print("TRACK B — Real GWOSC event GW150914", flush=True)
    print("=" * 72, flush=True)
    context = build_gw150914_context(
        pm, time_buffer=time_buffer, num_segments_psd=num_segments_psd
    )
    samples = run_sampler(
        pm,
        context,
        num_samples=num_samples,
        batch_size=batch_size,
        detectors=["H1", "L1"],
        event_metadata={"time_event": float(GW150914_GPS)},
    )

    keys = ["chirp_mass", "mass_ratio", "mass_1", "mass_2", "luminosity_distance"]
    summary = summarize_samples(samples, keys)
    published = GW150914_PUBLISHED

    print(
        f"{'Parameter':<22} {'Published':>10} {'Median':>10} "
        f"{'In 90% CI':>10} {'90% CI':>24}",
        flush=True,
    )
    print("-" * 82, flush=True)
    inside: Dict[str, bool] = {}
    for k in keys:
        pub = float(published[k])
        lo, med, hi = summary[k]["q05"], summary[k]["median"], summary[k]["q95"]
        ok = lo <= pub <= hi
        inside[k] = ok
        ci = f"[{lo:.4g}, {hi:.4g}]"
        print(
            f"{k:<22} {pub:10.4f} {med:10.4f} "
            f"{'YES' if ok else 'NO':>10} {ci:>24}",
            flush=True,
        )

    core = ["mass_1", "mass_2", "luminosity_distance"]
    all_ok = all(inside[k] for k in core)
    status = "PASS" if all_ok else "FAIL"
    print(
        f"\nPublished m1, m2, dL inside 90% CI → {status} "
        f"({sum(inside[k] for k in core)}/{len(core)} core params)",
        flush=True,
    )
    return {
        "track": "B",
        "name": "gw150914",
        "published": published,
        "samples": samples,
        "summary": summary,
        "inside_90": inside,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Plots / PDF
# ---------------------------------------------------------------------------


def _corner_like(
    ax_grid,
    samples: pd.DataFrame,
    params: Sequence[str],
    truths: Optional[Dict[str, float]] = None,
    truth_label: str = "truth",
):
    n = len(params)
    for i, yi in enumerate(params):
        for j, xj in enumerate(params):
            ax = ax_grid[i, j]
            if j > i:
                ax.axis("off")
                continue
            x = samples[xj].to_numpy()
            if i == j:
                ax.hist(x, bins=40, color="#4682B4", alpha=0.85, density=True)
                if truths and xj in truths:
                    ax.axvline(truths[xj], color="crimson", lw=1.5, label=truth_label)
                ax.set_yticks([])
            else:
                y = samples[yi].to_numpy()
                ax.scatter(x, y, s=3, alpha=0.15, c="#4682B4", rasterized=True)
                if truths and xj in truths and yi in truths:
                    ax.axvline(truths[xj], color="crimson", lw=1.0)
                    ax.axhline(truths[yi], color="crimson", lw=1.0)
            if i == n - 1:
                ax.set_xlabel(xj.replace("_", " "))
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(yi.replace("_", " "))
            elif j == 0 and i == j:
                ax.set_ylabel("density")
            else:
                ax.set_yticklabels([])


def save_track_png(
    result: Dict[str, Any], out_path: Path, title: str
) -> None:
    params = ["chirp_mass", "mass_1", "mass_2", "luminosity_distance"]
    samples = result["samples"]
    truths = result.get("truth") or result.get("published")
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    _corner_like(axes, samples, params, truths=truths)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def save_verification_pdf(
    track_a: Dict[str, Any],
    track_b: Dict[str, Any],
    out_path: Path,
    ckpt: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        # Page 1 — summary tables
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].axis("off")
        axes[1].axis("off")

        def _table(ax, title, rows, col_labels):
            ax.set_title(title, loc="left", fontsize=11, pad=8)
            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "(no rows — track skipped or failed)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return
            tbl = ax.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.35)

        rows_a = []
        for k, err in track_a["rae"].items():
            s = track_a["summary"][k]
            rows_a.append(
                [
                    k,
                    f"{track_a['truth'][k]:.4g}",
                    f"{s['median']:.4g}",
                    f"{err:.2%}",
                    f"[{s['q05']:.4g}, {s['q95']:.4g}]",
                ]
            )
        _table(
            axes[0],
            f"Track A — Synthetic  [{track_a['status']}]",
            rows_a,
            ["param", "true", "median", "RAE", "90% CI"],
        )

        rows_b = []
        for k, ok in track_b["inside_90"].items():
            s = track_b["summary"][k]
            rows_b.append(
                [
                    k,
                    f"{track_b['published'][k]:.4g}",
                    f"{s['median']:.4g}",
                    "YES" if ok else "NO",
                    f"[{s['q05']:.4g}, {s['q95']:.4g}]",
                ]
            )
        _table(
            axes[1],
            f"Track B — GW150914  [{track_b['status']}]",
            rows_b,
            ["param", "published", "median", "in 90% CI", "90% CI"],
        )
        fig.suptitle(
            f"DINGO-T1 native baseline\n{ckpt.name}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2/3 — corner-like plots (skip empty sample frames)
        for result, title in [
            (track_a, "Track A — Synthetic posterior"),
            (track_b, "Track B — GW150914 posterior"),
        ]:
            samples = result.get("samples")
            if samples is None or len(samples) == 0:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis("off")
                ax.set_title(title)
                ax.text(
                    0.5,
                    0.5,
                    f"No samples ({result.get('status', '?')})",
                    ha="center",
                    va="center",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue
            params = ["chirp_mass", "mass_1", "mass_2", "luminosity_distance"]
            n = len(params)
            fig, axes = plt.subplots(n, n, figsize=(8.5, 8.5))
            truths = result.get("truth") or result.get("published")
            _corner_like(axes, samples, params, truths=truths)
            fig.suptitle(title, fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info("Wrote verification PDF: %s", out_path)
    print(f"\nWrote verification PDF: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native unmodified DINGO-T1 baseline verification"
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-buffer", type=float, default=DEFAULT_TIME_BUFFER)
    parser.add_argument(
        "--psd-segments",
        type=int,
        default=32,
        help="Number of Tukey windows for Welch PSD (default 32 ≈ 256 s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the posterior model (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--skip-track-b",
        action="store_true",
        help="Skip GWOSC download track (synthetic only)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Unmodified DINGO-T1 Baseline Verification")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples: {args.num_samples}  batch: {args.batch_size}  device: {args.device}")

    pm = load_native_model(args.ckpt, device=args.device)

    track_a = run_track_a(
        pm,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    png_a = RESULTS_DIR / "baseline_verification_synthetic.png"
    save_track_png(
        track_a,
        png_a,
        f"Track A synthetic [{track_a['status']}] — Mc RAE={track_a['rae']['chirp_mass']:.2%}",
    )

    if args.skip_track_b:
        track_b = {
            "track": "B",
            "name": "gw150914",
            "published": GW150914_PUBLISHED,
            "samples": track_a["samples"].iloc[:0].copy(),
            "summary": {},
            "inside_90": {},
            "status": "SKIPPED",
        }
        png_b = RESULTS_DIR / "baseline_verification_gw150914.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Track B skipped", ha="center", va="center")
        fig.savefig(png_b, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        try:
            track_b = run_track_b(
                pm,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                time_buffer=args.time_buffer,
                num_segments_psd=args.psd_segments,
            )
            png_b = RESULTS_DIR / "baseline_verification_gw150914.png"
            save_track_png(
                track_b,
                png_b,
                f"Track B GW150914 [{track_b['status']}]",
            )
        except Exception as exc:
            logger.exception("Track B failed: %s", exc)
            print(f"\nTrack B FAILED with error: {exc}", flush=True)
            track_b = {
                "track": "B",
                "name": "gw150914",
                "published": GW150914_PUBLISHED,
                "samples": track_a["samples"].iloc[:0].copy(),
                "summary": {},
                "inside_90": {},
                "status": "ERROR",
                "error": str(exc),
            }
            png_b = RESULTS_DIR / "baseline_verification_gw150914.png"
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Track B error:\n{exc}",
                ha="center",
                va="center",
                wrap=True,
                fontsize=9,
            )
            fig.savefig(png_b, dpi=120, bbox_inches="tight")
            plt.close(fig)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = RESULTS_DIR / f"baseline_verification_{ts}.pdf"
    if track_b["status"] != "SKIPPED":
        save_verification_pdf(track_a, track_b, pdf_path, args.ckpt)
    else:
        # Still write a PDF with Track A only.
        save_verification_pdf(track_a, track_b, pdf_path, args.ckpt)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Track A (synthetic): {track_a['status']}")
    print(f"  Track B (GW150914):  {track_b['status']}")
    print(f"  PNG A: {png_a}")
    print(f"  PNG B: {png_b}")
    print(f"  PDF:   {pdf_path}")
    print("=" * 72)

    # Non-zero exit only if Track A fails (network/download issues shouldn't hide A).
    if not track_a["mc_rae_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
