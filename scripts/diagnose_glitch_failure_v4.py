#!/usr/bin/env python3
"""Phase-1 diagnosis for glitch-robust collapse (persist under results/diagnosis_v4).

Runs:
  1) Oracle: frozen NSF + DINGO(clean) embed on glitchy packaging
  2) Training STFT ASD-zero / WHITEN_EPS artifact check
  3) Eval leakage + in-band vs broadband severity
  4) Poison magnitude + v3 ctxhat quality
  5) Clean-FD / Welch-only ablations (persisted)

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src KMP_DUPLICATE_LIB_OK=TRUE
    python scripts/diagnose_glitch_failure_v4.py --num-samples 512 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "results" / "diagnosis_v4"
DEFAULT_V3 = REPO_ROOT / "checkpoints" / "glitch_robust_v3" / "best_glitch_robust.pt"

logger = logging.getLogger("diagnose_v4")


def _ensure_paths() -> None:
    for p in (
        REPO_ROOT / "DINGO-BNS" / "dingo",
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
    ):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _dl_ci(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return {"lo": float(lo), "med": float(med), "hi": float(hi), "n": int(x.size)}


def _band_rms(td: np.ndarray, sample_rate: float, f_min: float, f_max: float) -> float:
    x = np.asarray(td, dtype=np.float64).ravel()
    if x.size < 8:
        return float(np.std(x) or 1e-22)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    mask = (freqs >= float(f_min)) & (freqs <= float(f_max))
    if not np.any(mask):
        return float(np.std(x) or 1e-22)
    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    xb = np.fft.irfft(Xf, n=x.size)
    return float(np.std(xb) or 1e-22)


@torch.no_grad()
def _sample_flow_from_context(
    wrapper,
    metadata: Dict[str, Any],
    context: torch.Tensor,
    fixed: Dict[str, float],
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> np.ndarray:
    """Sample luminosity_distance from frozen flow given a precomputed 131-D context."""
    inference_params = list(metadata["train_settings"]["data"]["inference_parameters"])
    dl_idx = inference_params.index("luminosity_distance")
    std = metadata["train_settings"]["data"]["standardization"]
    ctx = context.to(device)
    if ctx.ndim == 1:
        ctx = ctx.unsqueeze(0)
    # NSF expects context as the second positional arg to flow.sample.
    flow = wrapper.flow
    chunks = []
    remaining = int(num_samples)
    while remaining > 0:
        bs = min(int(batch_size), remaining)
        c = ctx.expand(bs, -1)
        y = flow.sample(1, c)
        y = torch.squeeze(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        chunks.append(y.detach().cpu())
        remaining -= bs
    y_all = torch.cat(chunks, dim=0).numpy()
    mu = float(std["mean"]["luminosity_distance"])
    sig = float(std["std"]["luminosity_distance"]) or 1.0
    return y_all[:, dl_idx] * sig + mu


def diagnose_training_stft_artifact(metadata, out: Dict[str, Any]) -> None:
    from adapt.glitch_augmentation import (
        GlitchSpec,
        corrupt_injection_fd_with_glitch,
        stft_whitening_asds,
    )
    from adapt.stft_context import WHITEN_EPS, fd_waveform_to_td_crop, safe_whiten_fd
    from dingo.gw.domains import build_domain_from_model_metadata
    from train_bns_spectrogram import build_base_domain_injection

    domain = build_domain_from_model_metadata(metadata)
    injection, base_domain = build_base_domain_injection(metadata)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    duration = float(base_domain.duration)
    noise_std = float(base_domain.noise_std)
    detectors = list(metadata["train_settings"]["data"]["detectors"])

    rng = np.random.default_rng(0)
    try:
        import bilby

        bilby.core.utils.random.seed(0)
    except Exception:
        pass
    theta = {k: float(v) for k, v in injection.prior.sample().items()}
    inj_clean = injection.injection(theta)
    asd_h1 = np.asarray(inj_clean["asds"]["H1"], dtype=np.float64)
    zero_bins = np.where(asd_h1 <= WHITEN_EPS * 10)[0]
    f_zero = (
        float(zero_bins[-1] * (1.0 / duration)) if zero_bins.size else float("nan")
    )

    spec = GlitchSpec(
        family="sine_gaussian",
        detectors=("H1",),
        severity=8.0,
        t_rel=-1.0,
        asd_policy="stationary",
        held_out=False,
        params={"f0": 100.0, "q": 5.0},
    )
    inj_g, _td, _meta = corrupt_injection_fd_with_glitch(
        inj_clean,
        sample_rate=sample_rate,
        duration=duration,
        spec=spec,
        rng=rng,
    )
    stft_asds = stft_whitening_asds(inj_clean["asds"], inj_g["asds"])
    params = inj_clean.get("parameters") or {}
    geocent = float(params.get("geocent_time", 0.0))

    rows = {}
    for label, inj in ("clean", inj_clean), ("glitch", inj_g):
        for det in detectors:
            fd = np.asarray(inj["waveform"][det], dtype=np.complex128)
            asd = np.asarray(stft_asds[det], dtype=np.float64)
            w = safe_whiten_fd(fd, asd, noise_std=noise_std)
            td = fd_waveform_to_td_crop(
                fd,
                sample_rate=sample_rate,
                duration=duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=asd,
                noise_std=noise_std,
                whiten=True,
            )
            rows[f"{label}_{det}"] = {
                "whitened_fd_max": float(np.max(np.abs(w))),
                "whitened_fd_argmax_hz": float(
                    np.argmax(np.abs(w)) * (1.0 / duration)
                ),
                "td_std": float(np.std(td)),
                "asd_min": float(np.min(asd)),
                "asd_n_near_zero": int(np.sum(asd <= WHITEN_EPS * 10)),
            }
    out["training_stft_artifact"] = {
        "whiten_eps": float(WHITEN_EPS),
        "asd_h1_zero_bin_hz": f_zero,
        "asd_h1_n_near_zero": int(zero_bins.size),
        "per_ifo": rows,
        "l1_v1_detonation": bool(
            rows["glitch_L1"]["td_std"] > 100 * max(rows["clean_L1"]["td_std"], 1e-30)
            or rows["glitch_V1"]["td_std"] > 100 * max(rows["clean_V1"]["td_std"], 1e-30)
        ),
        "h1_whitened_pathological": bool(rows["glitch_H1"]["whitened_fd_max"] > 1e6),
        "artifact_suspected": bool(
            rows["glitch_L1"]["td_std"] > 100 * max(rows["clean_L1"]["td_std"], 1e-30)
            or rows["glitch_V1"]["td_std"] > 100 * max(rows["clean_V1"]["td_std"], 1e-30)
            or rows["glitch_H1"]["whitened_fd_max"] > 1e6
        ),
    }
    logger.info(
        "Training STFT artifact suspected=%s | glitch H1 fd_max=%.3e L1 td_std=%.3e "
        "(clean L1 td_std=%.3e)",
        out["training_stft_artifact"]["artifact_suspected"],
        rows["glitch_H1"]["whitened_fd_max"],
        rows["glitch_L1"]["td_std"],
        rows["clean_L1"]["td_std"],
    )


def diagnose_eval_scale(assets, event, out: Dict[str, Any]) -> None:
    from adapt.stft_context import WHITEN_EPS, whiten_td_crop_with_asd
    from evaluate_glitch_robustness import inject_h1_glitch_into_event, load_full_event_td
    from evaluate_gw170817_comparison import load_event_td_crops
    from scipy import signal as sp_signal

    settings = dict(event.settings)
    f_s = float(settings.get("f_s", 4096.0))
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    td_h1, _, _ = load_full_event_td(assets, settings, "H1")
    bb_rms = float(np.std(td_h1) or 1e-22)
    ib_rms = _band_rms(td_h1, f_s, f_min, f_max)
    out["severity_scale"] = {
        "broadband_rms": bb_rms,
        "inband_rms": ib_rms,
        "ratio_bb_over_ib": float(bb_rms / max(ib_rms, 1e-30)),
        "amp_at_scale8_broadband": 8.0 * bb_rms,
        "amp_at_scale8_inband": 8.0 * ib_rms,
    }

    crops = load_event_td_crops(assets, sample_rate=f_s, crop_seconds=4.0)
    asd = np.asarray(event.data["asds"]["H1"], dtype=np.float64)
    delta_f = 1.0 / float(settings.get("T", 128.0))
    noise_std = float(
        np.sqrt(f_s) / 2.0
    )  # placeholder; overwritten below from domain if possible
    try:
        from dingo.gw.domains import build_domain_from_model_metadata

        # noise_std from event domain via baseline metadata path later
    except Exception:
        pass

    def _whiten_stats(td, taper: bool, highpass: bool):
        x = np.asarray(td, dtype=np.float64).ravel().copy()
        if highpass:
            b, a = sp_signal.butter(4, 20.0 / (0.5 * f_s), btype="high")
            x = sp_signal.filtfilt(b, a, x)
        if taper:
            alpha = min(1.0, 2.0 * 0.4 * f_s / max(len(x), 1))
            x = x * sp_signal.windows.tukey(len(x), alpha=alpha)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / f_s)
        f_asd = np.arange(asd.size, dtype=np.float64) * delta_f
        asd_i = np.interp(freqs, f_asd, asd, left=asd[0], right=asd[-1])
        asd_i = np.maximum(asd_i, WHITEN_EPS)
        # Use same noise_std convention as domain when available; else 1.0 for relative.
        yw = np.fft.irfft(X / (asd_i * 1.0), n=x.size)
        return float(np.std(yw))

    # Prefer domain noise_std from packaged event settings if present.
    # Comparison helpers use base_domain.noise_std; approximate via current API.
    from evaluate_gw170817_comparison import discover_assets

    # Use whiten_td_crop_with_asd as implemented (untapered) vs tapered.
    # Discover noise_std from a loaded baseline metadata later in main.
    out["eval_leakage"] = {
        "note": "filled in main with domain noise_std",
        "h1_crop_std_raw": float(np.std(crops["H1"])),
        "untapered_whiten_std_noise1": _whiten_stats(crops["H1"], False, False),
        "tukey_whiten_std_noise1": _whiten_stats(crops["H1"], True, False),
        "tukey_hp20_whiten_std_noise1": _whiten_stats(crops["H1"], True, True),
    }
    _, _, gmeta = inject_h1_glitch_into_event(event, assets, snr_amp_scale=8.0)
    out["eval_glitch_meta_broadband"] = gmeta


def run(args: argparse.Namespace) -> None:
    _ensure_paths()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"outdir": str(outdir)}

    from adapt.stft_context import (
        ROBUST_CHANNEL_NAMES,
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )
    from dingo.gw.domains import build_domain_from_model_metadata
    from evaluate_glitch_robustness import inject_h1_glitch_into_event
    from evaluate_gw170817_comparison import (
        build_event_spectrogram_stack,
        discover_assets,
        load_custom_wrapper,
        load_event_dataset,
        load_event_td_crops,
        package_event_strain,
        run_baseline_sampling,
        select_device,
        standardize_context,
    )
    from train_bns_spectrogram import build_flow_wrapper, load_bns_checkpoint

    device = select_device(args.device)
    logger.info("Device: %s", device)

    custom_ckpt = Path(args.custom_ckpt) if args.custom_ckpt else DEFAULT_V3
    assets = discover_assets(
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=custom_ckpt if custom_ckpt.is_file() else None,
    )
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]
    raw = load_bns_checkpoint(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"] if "metadata" in raw else raw.get("model_metadata", {})
    # load_bns_checkpoint returns dict with metadata key
    if "model_metadata" in raw and "metadata" not in raw:
        metadata = raw["model_metadata"]
    metadata = raw.get("metadata") or raw.get("model_metadata")
    # train_bns_spectrogram.load_bns_checkpoint shape
    from train_bns_spectrogram import load_bns_checkpoint as _lb

    raw = _lb(Path(assets["baseline_ckpt"]))
    metadata = raw["metadata"]
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    noise_std = float(base_domain.noise_std)
    delta_f = float(base_domain.delta_f)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        event.settings.get("f_s")
        or metadata["train_settings"]["data"]["window"]["f_s"]
    )

    # ---- 2) training STFT artifact ----
    diagnose_training_stft_artifact(metadata, report)

    # ---- 3) eval scale / leakage ----
    diagnose_eval_scale(assets, event, report)
    crops = load_event_td_crops(assets, sample_rate=sample_rate, crop_seconds=4.0)
    asds_clean = {d: np.asarray(event.data["asds"][d]) for d in detectors}
    from adapt.stft_context import whiten_td_crop_with_asd
    from scipy import signal as sp_signal

    def _std_whiten(td, asd, taper: bool):
        x = np.asarray(td, dtype=np.float64).ravel().copy()
        if taper:
            alpha = float(np.clip(2.0 * 0.4 * sample_rate / max(len(x), 1), 0.0, 1.0))
            x = x * sp_signal.windows.tukey(len(x), alpha=alpha)
        return float(
            np.std(
                whiten_td_crop_with_asd(
                    x, sample_rate, asd, delta_f=delta_f, noise_std=noise_std
                )
            )
        )

    report["eval_leakage"] = {
        "noise_std": noise_std,
        "H1_untapered": _std_whiten(crops["H1"], asds_clean["H1"], False),
        "H1_tukey_pre": _std_whiten(crops["H1"], asds_clean["H1"], True),
        "L1_untapered": _std_whiten(crops["L1"], asds_clean["L1"], False),
        "L1_tukey_pre": _std_whiten(crops["L1"], asds_clean["L1"], True),
        "V1_untapered": _std_whiten(crops["V1"], asds_clean["V1"], False),
        "V1_tukey_pre": _std_whiten(crops["V1"], asds_clean["V1"], True),
    }
    logger.info("Eval leakage whitened stds: %s", report["eval_leakage"])

    # Clean / glitch event packages
    glitch_data, td_stft, gmeta = inject_h1_glitch_into_event(
        event, assets, snr_amp_scale=float(args.snr_amp_scale)
    )
    # In-band severity alternative
    settings = dict(event.settings)
    td_h1, f_s, _ = __import__(
        "evaluate_glitch_robustness", fromlist=["load_full_event_td"]
    ).load_full_event_td(assets, settings, "H1")
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    report["severity_scale"]["glitch_meta_broadband"] = {
        k: (float(v) if isinstance(v, (float, int, np.floating)) else v)
        for k, v in gmeta.items()
    }
    report["severity_scale"]["recommended_amp_scale8_inband"] = 8.0 * _band_rms(
        td_h1, f_s, f_min, f_max
    )

    from adapt.glitch_augmentation import stft_whitening_asds

    stft_asds = stft_whitening_asds(asds_clean, {d: glitch_data["asds"][d] for d in detectors})
    spec_g, loge_g = build_event_spectrogram_stack(
        td_stft,
        detectors,
        sample_rate,
        asds=stft_asds,
        delta_f=delta_f,
        noise_std=noise_std,
        robust=True,
    )
    td_clean = load_event_td_crops(assets, sample_rate=sample_rate, crop_seconds=4.0)
    spec_c, loge_c = build_event_spectrogram_stack(
        td_clean,
        detectors,
        sample_rate,
        asds=asds_clean,
        delta_f=delta_f,
        noise_std=noise_std,
        robust=True,
    )
    report["event_log_energy"] = {
        "clean": {n: float(v) for n, v in zip(("H1", "L1", "V1"), np.asarray(loge_c))},
        "glitch": {n: float(v) for n, v in zip(("H1", "L1", "V1"), np.asarray(loge_g))},
    }
    logger.info("Event log_energy clean=%s glitch=%s", loge_c, loge_g)

    # ---- baseline + oracle + ablations ----
    wrapper = build_flow_wrapper(raw).to(device)
    wrapper.eval()
    context_z = standardize_context(
        fixed, metadata["train_settings"]["data"]["standardization"]
    )
    z_t = torch.from_numpy(np.asarray(context_z, dtype=np.float32)).to(device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)

    strain_clean = package_event_strain(event.data, metadata, fixed)
    strain_glitch = package_event_strain(glitch_data, metadata, fixed)
    # Welch-only: clean waveform + Welch ASD on H1
    welch_only = __import__("copy").deepcopy(event.data)
    welch_only["asds"]["H1"] = np.asarray(glitch_data["asds"]["H1"]).copy()
    strain_welch_only = package_event_strain(welch_only, metadata, fixed)
    # Glitch waveform + clean ASD
    glitch_clean_asd = __import__("copy").deepcopy(glitch_data)
    glitch_clean_asd["asds"] = {d: np.asarray(event.data["asds"][d]).copy() for d in detectors}
    strain_glitch_clean_asd = package_event_strain(glitch_clean_asd, metadata, fixed)

    sc = torch.from_numpy(strain_clean).unsqueeze(0).to(device)
    sg = torch.from_numpy(strain_glitch).unsqueeze(0).to(device)
    sw = torch.from_numpy(strain_welch_only).unsqueeze(0).to(device)
    sgc = torch.from_numpy(strain_glitch_clean_asd).unsqueeze(0).to(device)

    emb = wrapper.embedding_net
    ctx_clean = emb(sc, z_t)
    ctx_glitch = emb(sg, z_t)
    ctx_welch = emb(sw, z_t)
    ctx_g_clean_asd = emb(sgc, z_t)

    poison = {
        "l2_glitch_vs_clean": float(torch.norm(ctx_glitch - ctx_clean).cpu()),
        "l2_welch_only_vs_clean": float(torch.norm(ctx_welch - ctx_clean).cpu()),
        "l2_glitch_wave_clean_asd_vs_clean": float(
            torch.norm(ctx_g_clean_asd - ctx_clean).cpu()
        ),
        "mse_sum_glitch_vs_clean": float(
            (ctx_glitch[:, :128] - ctx_clean[:, :128]).pow(2).sum().cpu()
        ),
    }
    report["poison_magnitude"] = poison
    logger.info("Poison magnitude: %s", poison)

    # Oracle: clean embed into flow while "glitch" situation
    dl_oracle = _sample_flow_from_context(
        wrapper,
        metadata,
        ctx_clean[0],
        fixed,
        device=device,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
    )
    dl_glitch_ctx = _sample_flow_from_context(
        wrapper,
        metadata,
        ctx_glitch[0],
        fixed,
        device=device,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
    )
    dl_welch_ctx = _sample_flow_from_context(
        wrapper,
        metadata,
        ctx_welch[0],
        fixed,
        device=device,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
    )
    report["oracle"] = {
        "clean_embed_d_L": _dl_ci(dl_oracle),
        "glitch_embed_d_L": _dl_ci(dl_glitch_ctx),
        "welch_only_embed_d_L": _dl_ci(dl_welch_ctx),
        "oracle_escaped": bool(_dl_ci(dl_oracle)["med"] > 20.0),
    }
    logger.info("Oracle d_L: %s", report["oracle"])

    # Baseline sampler on glitch event (full pipeline)
    glitch_event = SimpleNamespace(data=glitch_data, settings=event.settings)
    base_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        glitch_event,
        fixed,
        device=device,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
    )
    report["baseline_glitch_d_L"] = _dl_ci(base_df["luminosity_distance"].to_numpy())

    clean_event = SimpleNamespace(data=event.data, settings=event.settings)
    base_clean_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        clean_event,
        fixed,
        device=device,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
    )
    report["baseline_clean_d_L"] = _dl_ci(
        base_clean_df["luminosity_distance"].to_numpy()
    )

    # v3 ctxhat quality if checkpoint exists
    if custom_ckpt.is_file():
        student, meta_s = load_custom_wrapper(
            assets["baseline_ckpt"], custom_ckpt, device
        )
        student.eval()
        pc = torch.from_numpy(np.asarray(spec_c, dtype=np.float32)).unsqueeze(0).to(device)
        pg = torch.from_numpy(np.asarray(spec_g, dtype=np.float32)).unsqueeze(0).to(device)
        ec = torch.from_numpy(np.asarray(loge_c, dtype=np.float32)).unsqueeze(0).to(device)
        eg = torch.from_numpy(np.asarray(loge_g, dtype=np.float32)).unsqueeze(0).to(device)
        # Prefer student base embedding
        emb_s = student.embedding_net
        if hasattr(emb_s, "forward_with_diagnostics"):
            diag = emb_s.forward_with_diagnostics(sg, pg, eg, z_t)
            t_embed = ctx_clean[:, :128]
            ctxhat_mse = float((diag["ctx_hat"] - t_embed).pow(2).sum(dim=-1).mean().cpu())
            base_mse = float(
                (diag["base_embed"] - t_embed).pow(2).sum(dim=-1).mean().cpu()
            )
            corr_mse = float(
                (diag["corrected_embed"] - t_embed).pow(2).sum(dim=-1).mean().cpu()
            )
            report["v3_reconstruction"] = {
                "ctxhat_mse_sum": ctxhat_mse,
                "base_mse_sum": base_mse,
                "corrected_mse_sum": corr_mse,
                "gate": float(diag["gate"].mean().cpu()),
                "energy_gate": float(diag["energy_gate"].mean().cpu()),
                "closer_than_base": bool(ctxhat_mse < base_mse),
            }
            logger.info("v3 reconstruction: %s", report["v3_reconstruction"])

    # Persist
    path = outdir / "diagnosis_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    np.savez_compressed(
        outdir / "oracle_d_L_samples.npz",
        oracle_clean_embed=dl_oracle,
        glitch_embed=dl_glitch_ctx,
        welch_only_embed=dl_welch_ctx,
    )
    logger.info("Wrote %s", path)
    print(json.dumps(report, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--custom-ckpt", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--snr-amp-scale", type=float, default=8.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
