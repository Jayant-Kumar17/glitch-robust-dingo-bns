#!/usr/bin/env python3
"""Locked multi-seed injection benchmark for glitch-robust DINGO.

Compares baseline DINGO vs custom glitch-robust corrector on identical
clean / held-in / held-out glitch twins across families, detectors,
severity bins, and ASD policies.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src
    export KMP_DUPLICATE_LIB_OK=TRUE

    python scripts/evaluate_bns_glitch_suite.py \\
      --custom-ckpt checkpoints/glitch_robust/best_glitch_robust.pt \\
      --n-injections 64 --num-samples 512
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "src",
    REPO_ROOT / "DINGO-BNS" / "dingo",
    REPO_ROOT,
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_BNS = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "downloads"
    / "dingo-bns-model_GW170817.pt"
)
DEFAULT_CUSTOM = REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "glitch_suite"

logger = logging.getLogger("evaluate_bns_glitch_suite")

SEVERITY_BINS = ((2.0, 4.0), (4.0, 8.0), (8.0, 12.0))


def select_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ci90(x: np.ndarray) -> Tuple[float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, med, hi = np.quantile(a, [0.05, 0.5, 0.95])
    return float(med), float(lo), float(hi)


def _coverage90(samples: np.ndarray, truth: float) -> bool:
    _, lo, hi = _ci90(samples)
    return bool(lo <= truth <= hi)


def build_locked_specs(
    n: int,
    *,
    seed: int,
    detectors: Sequence[str],
    held_out: bool,
) -> List[Any]:
    from adapt.glitch_augmentation import sample_glitch_spec

    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        # Cycle severity bins and ASD policies for locked coverage.
        sev_lo, sev_hi = SEVERITY_BINS[i % len(SEVERITY_BINS)]
        asd = "welch" if (i % 2 == 0) else "stationary"
        spec = sample_glitch_spec(
            rng,
            detectors=detectors,
            held_out=held_out,
            severity_range=(sev_lo, sev_hi),
            asd_policy=asd,
        )
        specs.append(spec)
    return specs


@torch.no_grad()
def sample_posterior(
    wrapper,
    *,
    strain: np.ndarray,
    context_z: np.ndarray,
    spectrogram: Optional[np.ndarray],
    log_energy: Optional[np.ndarray],
    device: torch.device,
    num_samples: int,
    batch_size: int,
    custom: bool,
) -> np.ndarray:
    """Return standardized y samples (N, D)."""
    remaining = int(num_samples)
    chunks = []
    s0 = torch.from_numpy(np.asarray(strain, dtype=np.float32))
    z0 = torch.from_numpy(np.asarray(context_z, dtype=np.float32))
    sp0 = (
        torch.from_numpy(np.asarray(spectrogram, dtype=np.float32))
        if spectrogram is not None
        else None
    )
    e0 = (
        torch.from_numpy(np.asarray(log_energy, dtype=np.float32))
        if log_energy is not None
        else None
    )
    while remaining > 0:
        bs = min(int(batch_size), remaining)
        s = s0.unsqueeze(0).expand(bs, *s0.shape).to(device)
        z = z0.unsqueeze(0).expand(bs, -1).to(device)
        if custom:
            sp = sp0.unsqueeze(0).expand(bs, *sp0.shape).to(device)
            e = e0.unsqueeze(0).expand(bs, -1).to(device)
            y = wrapper.sample(s, sp, e, z, num_samples=1)
        else:
            y = wrapper.sample(s, z, num_samples=1)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        chunks.append(y.detach().cpu().numpy())
        remaining -= bs
    return np.concatenate(chunks, axis=0)


def denorm_param(
    y: np.ndarray,
    name: str,
    inference_params: Sequence[str],
    standardization: Dict[str, Any],
    fixed_proxy: Optional[float] = None,
) -> np.ndarray:
    means = standardization["mean"]
    stds = standardization["std"]
    if name == "chirp_mass" and "delta_chirp_mass" in inference_params and fixed_proxy is not None:
        i = list(inference_params).index("delta_chirp_mass")
        mu = float(means.get("delta_chirp_mass", 0.0))
        sig = float(stds.get("delta_chirp_mass", 1.0) or 1.0)
        return y[:, i] * sig + mu + float(fixed_proxy)
    if name not in inference_params:
        raise KeyError(name)
    i = list(inference_params).index(name)
    mu = float(means.get(name, 0.0))
    sig = float(stds.get(name, 1.0) or 1.0)
    return y[:, i] * sig + mu


def eval_one_injection(
    *,
    teacher,
    student,
    metadata: Dict[str, Any],
    theta: Dict[str, float],
    strain_c: np.ndarray,
    strain_g: np.ndarray,
    spec_c: np.ndarray,
    spec_g: np.ndarray,
    e_c: np.ndarray,
    e_g: np.ndarray,
    context_z: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    glitch_meta: Dict[str, Any],
) -> Dict[str, Any]:
    inference_params = list(metadata["train_settings"]["data"]["inference_parameters"])
    standardization = metadata["train_settings"]["data"]["standardization"]
    proxy = float(theta["chirp_mass_proxy"])

    # NLL at true θ
    yt = torch.from_numpy(y_true).unsqueeze(0).to(device)
    sc = torch.from_numpy(strain_c).unsqueeze(0).to(device)
    sg = torch.from_numpy(strain_g).unsqueeze(0).to(device)
    z = torch.from_numpy(context_z).unsqueeze(0).to(device)
    pc = torch.from_numpy(spec_c).unsqueeze(0).to(device)
    pg = torch.from_numpy(spec_g).unsqueeze(0).to(device)
    ec = torch.from_numpy(e_c).unsqueeze(0).to(device)
    eg = torch.from_numpy(e_g).unsqueeze(0).to(device)

    with torch.no_grad():
        nll_tc = float((-teacher.log_prob(yt, sc, z)).item())
        nll_tg = float((-teacher.log_prob(yt, sg, z)).item())
        nll_sc = float((-student.log_prob(yt, sc, pc, ec, z)).item())
        nll_sg = float((-student.log_prob(yt, sg, pg, eg, z)).item())

    # Posterior samples (clean + glitch) for coverage / CI width on d_L
    y_b_c = sample_posterior(
        teacher,
        strain=strain_c,
        context_z=context_z,
        spectrogram=None,
        log_energy=None,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        custom=False,
    )
    y_b_g = sample_posterior(
        teacher,
        strain=strain_g,
        context_z=context_z,
        spectrogram=None,
        log_energy=None,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        custom=False,
    )
    y_c_c = sample_posterior(
        student,
        strain=strain_c,
        context_z=context_z,
        spectrogram=spec_c,
        log_energy=e_c,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        custom=True,
    )
    y_c_g = sample_posterior(
        student,
        strain=strain_g,
        context_z=context_z,
        spectrogram=spec_g,
        log_energy=e_g,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        custom=True,
    )

    truth_dl = float(theta["luminosity_distance"])
    rows = {
        "family": glitch_meta.get("family"),
        "detectors": ",".join(glitch_meta.get("detectors") or []),
        "severity": float(glitch_meta.get("severity", 0.0)),
        "asd_policy": glitch_meta.get("asd_policy"),
        "held_out": bool(glitch_meta.get("held_out", False)),
        "nll_teacher_clean": nll_tc,
        "nll_teacher_glitch": nll_tg,
        "nll_student_clean": nll_sc,
        "nll_student_glitch": nll_sg,
        "clean_gap": nll_sc - nll_tc,
        "glitch_improvement": nll_tg - nll_sg,
    }
    for tag, yarr in (
        ("base_clean", y_b_c),
        ("base_glitch", y_b_g),
        ("custom_clean", y_c_c),
        ("custom_glitch", y_c_g),
    ):
        dl = denorm_param(
            yarr, "luminosity_distance", inference_params, standardization, proxy
        )
        med, lo, hi = _ci90(dl)
        rows[f"{tag}_dl_med"] = med
        rows[f"{tag}_dl_lo"] = lo
        rows[f"{tag}_dl_hi"] = hi
        rows[f"{tag}_dl_width"] = hi - lo
        rows[f"{tag}_dl_cover"] = int(_coverage90(dl, truth_dl))
        rows[f"{tag}_dl_abserr"] = abs(med - truth_dl)
    return rows


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.glitch_augmentation import (
        corrupt_injection_fd_with_glitch,
        stft_whitening_asds,
    )
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        fd_waveform_to_td_crop,
    )
    from evaluate_gw170817_comparison import load_custom_wrapper
    from train_bns_glitch_robust import CONTEXT_PARAMS, PROXY_JITTER_FRAC
    from train_bns_spectrogram import (
        build_base_domain_injection,
        build_flow_wrapper,
        decimate_injection_to_domain,
        load_bns_checkpoint,
        package_strain_sample,
        standardize_vector,
    )

    device = select_device(args.device)
    logger.info("Device: %s", device)

    raw = load_bns_checkpoint(Path(args.baseline_ckpt))
    metadata = raw["metadata"]
    teacher = build_flow_wrapper(raw).float().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student, stu_meta = load_custom_wrapper(
        Path(args.baseline_ckpt), Path(args.custom_ckpt), device
    )
    if not stu_meta.get("_glitch_robust"):
        logger.warning(
            "Custom checkpoint is not flagged glitch_robust; suite still runs "
            "but expects 6-ch robust STFT when possible."
        )
    norm_stats = stu_meta.get("_norm_stats")
    stft_kw = stu_meta.get("_stft_kwargs") or {}
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    inference_params = list(metadata["train_settings"]["data"]["inference_parameters"])
    block = metadata["train_settings"]["data"]["standardization"]
    means = {k: float(v) for k, v in block["mean"].items()}
    stds = {k: float(v) for k, v in block["std"].items()}

    injection, base = build_base_domain_injection(metadata)
    from dingo.gw.domains import build_domain_from_model_metadata

    domain = build_domain_from_model_metadata(metadata)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    duration = float(base.duration)
    noise_std = float(base.noise_std)

    n = int(args.n_injections)
    held_in = build_locked_specs(
        n // 2, seed=int(args.seed), detectors=detectors, held_out=False
    )
    held_out = build_locked_specs(
        n - n // 2, seed=int(args.seed) + 1, detectors=detectors, held_out=True
    )
    specs = held_in + held_out

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 42)

    for i, spec in enumerate(specs):
        try:
            import bilby

            bilby.core.utils.random.seed(int(rng.integers(0, 2**31 - 1)))
        except Exception:
            pass
        theta = {k: float(v) for k, v in injection.prior.sample().items()}
        mc = float(theta["chirp_mass"])
        theta["chirp_mass_proxy"] = mc + PROXY_JITTER_FRAC * mc * float(rng.normal())
        theta["delta_chirp_mass"] = mc - theta["chirp_mass_proxy"]

        inj_c = injection.injection(theta)
        inj_c_mfd = decimate_injection_to_domain(inj_c, domain)
        strain_c = package_strain_sample(inj_c_mfd, detectors, domain)

        params = inj_c.get("parameters") or {}
        geocent = float(params.get("geocent_time", 0.0))
        td_c = {}
        for det in detectors:
            td_c[det] = fd_waveform_to_td_crop(
                inj_c["waveform"][det],
                sample_rate=sample_rate,
                duration=duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=inj_c["asds"][det],
                noise_std=noise_std,
                whiten=True,
            )
        robust_kw = {
            k: stft_kw[k]
            for k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
            if k in stft_kw
        }
        spec_c, e_c = build_robust_spectrogram_from_td(
            td_c,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=norm_stats,
            **robust_kw,
        )

        inj_g, _td_crop, gmeta = corrupt_injection_fd_with_glitch(
            inj_c,
            sample_rate=sample_rate,
            duration=duration,
            spec=spec,
            rng=rng,
        )
        # STFT: clean/stationary ASD whitening; FD may still use Welch.
        stft_asds = stft_whitening_asds(inj_c["asds"], inj_g["asds"])
        td_g = {}
        params_g = inj_g.get("parameters") or params
        for det in detectors:
            td_g[det] = fd_waveform_to_td_crop(
                inj_g["waveform"][det],
                sample_rate=sample_rate,
                duration=duration,
                trigger_time=float(params_g.get(f"{det}_time", geocent)),
                asd=stft_asds[det],
                noise_std=noise_std,
                whiten=True,
            )
        inj_g_mfd = decimate_injection_to_domain(inj_g, domain)
        strain_g = package_strain_sample(inj_g_mfd, detectors, domain)
        spec_g, e_g = build_robust_spectrogram_from_td(
            td_g,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=norm_stats,
            **robust_kw,
        )

        y_true = standardize_vector(theta, inference_params, means, stds)
        context_z = standardize_vector(theta, CONTEXT_PARAMS, means, stds)

        row = eval_one_injection(
            teacher=teacher,
            student=student,
            metadata=metadata,
            theta=theta,
            strain_c=strain_c,
            strain_g=strain_g,
            spec_c=spec_c,
            spec_g=spec_g,
            e_c=e_c,
            e_g=e_g,
            context_z=context_z,
            y_true=y_true,
            device=device,
            num_samples=int(args.num_samples),
            batch_size=int(args.batch_size),
            glitch_meta=gmeta,
        )
        row["index"] = i
        rows.append(row)
        logger.info(
            "[%d/%d] fam=%s sev=%.1f held_out=%s | clean_gap=%.3f "
            "glitchΔNLL=%.3f cover base/custom=%.0f/%.0f",
            i + 1,
            len(specs),
            row["family"],
            row["severity"],
            row["held_out"],
            row["clean_gap"],
            row["glitch_improvement"],
            100 * row["base_glitch_dl_cover"],
            100 * row["custom_glitch_dl_cover"],
        )

    df = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "glitch_suite_results.csv"
    df.to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {
        "n": int(len(df)),
        "mean_clean_gap": float(df["clean_gap"].mean()),
        "mean_glitch_improvement": float(df["glitch_improvement"].mean()),
        "cover_base_glitch": float(df["base_glitch_dl_cover"].mean()),
        "cover_custom_glitch": float(df["custom_glitch_dl_cover"].mean()),
        "cover_base_clean": float(df["base_clean_dl_cover"].mean()),
        "cover_custom_clean": float(df["custom_clean_dl_cover"].mean()),
        "median_width_ratio_glitch": float(
            (df["custom_glitch_dl_width"] / df["base_glitch_dl_width"].clip(lower=1e-6)).median()
        ),
    }
    by_family = (
        df.groupby("family")[["glitch_improvement", "custom_glitch_dl_cover", "base_glitch_dl_cover"]]
        .mean()
        .to_dict()
    )
    summary["by_family"] = by_family
    held = df.groupby("held_out")[["glitch_improvement", "custom_glitch_dl_cover"]].mean()
    summary["by_held_out"] = held.to_dict()

    json_path = outdir / "glitch_suite_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Wrote %s", csv_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2, default=str))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Broad glitch injection benchmark")
    p.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BNS)
    p.add_argument("--custom-ckpt", type=Path, default=DEFAULT_CUSTOM)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--n-injections", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
