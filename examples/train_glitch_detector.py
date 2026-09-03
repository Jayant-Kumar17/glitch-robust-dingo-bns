#!/usr/bin/env python3
"""Train the STFT glitch detector for detect-and-gate excision.

Generates clean/glitch STFT pairs with known time-support labels from
``adapt.glitch_augmentation``, trains ``GlitchDetectorSTFT`` with BCE
segmentation loss, calibrates a threshold for <1% clean false-positive rate,
and validates held-out glitch families.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
    python examples/train_glitch_detector.py \\
      --epochs 30 --batch-size 16 --steps-per-epoch 100 \\
      --outdir checkpoints/glitch_detector_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "examples",
    REPO_ROOT / "src",
    REPO_ROOT / "DINGO-BNS" / "dingo",
    REPO_ROOT,
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_BNS_CKPT = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "downloads"
    / "dingo-bns-model_GW170817.pt"
)
DEFAULT_OUTDIR = REPO_ROOT / "checkpoints" / "glitch_detector_v1"
DETECTORS = ("H1", "L1", "V1")

logger = logging.getLogger("train_glitch_detector")


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DetectorPairDataset(Dataset):
    """On-the-fly clean / glitch spectrograms + per-IFO time-bin labels."""

    def __init__(
        self,
        metadata: Dict[str, Any],
        *,
        length: int,
        seed: int = 0,
        held_out: bool = False,
        clean_frac: float = 0.35,
        norm_stats: Optional[Dict[str, Any]] = None,
        stft_kwargs: Optional[Dict[str, Any]] = None,
        gate_half_s: float = 0.4,
        force_glitch: bool = False,
    ):
        from dingo.gw.domains import build_domain_from_model_metadata
        from adapt.dingo_bns_demo import build_base_domain_injection

        self.metadata = metadata
        self.length = int(length)
        self.base_seed = int(seed)
        self.held_out = bool(held_out)
        self.clean_frac = float(clean_frac)
        self.norm_stats = norm_stats
        self.stft_kwargs = dict(stft_kwargs or {})
        self.gate_half_s = float(gate_half_s)
        self.force_glitch = bool(force_glitch)
        self.detectors = list(DETECTORS)
        self.injection, self.base_domain = build_base_domain_injection(metadata)
        self.domain = build_domain_from_model_metadata(metadata)
        self.sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
        self.duration = float(self.base_domain.duration)
        self.noise_std = float(self.base_domain.noise_std)
        self.n_time = int(self.stft_kwargs.get("n_time", 32))

    def __len__(self) -> int:
        return self.length

    def _theta(self, rng: np.random.Generator) -> Dict[str, float]:
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            import bilby

            bilby.core.utils.random.seed(seed)
        except Exception:
            pass
        return {k: float(v) for k, v in self.injection.prior.sample().items()}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from adapt.glitch_augmentation import (
            corrupt_injection_fd_with_glitch,
            sample_glitch_spec,
            stft_whitening_asds,
        )
        from adapt.glitch_excision import (
            glitch_support_mask_on_crop,
            support_mask_to_time_bins,
        )
        from adapt.stft_context import (
            build_robust_spectrogram_from_td,
            crop_td_to_analysis_window,
            fd_waveform_to_td_crop,
        )

        rng = np.random.default_rng(self.base_seed + int(index) * 1009 + 17)
        theta = self._theta(rng)
        inj = self.injection.injection(theta)
        params = inj.get("parameters") or {}
        geocent = float(params.get("geocent_time", 0.0))

        # Clean whitened TD crops for all IFOs.
        td_clean = {}
        for det in self.detectors:
            td_clean[det] = fd_waveform_to_td_crop(
                inj["waveform"][det],
                sample_rate=self.sample_rate,
                duration=self.duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=inj["asds"][det],
                noise_std=self.noise_std,
                whiten=True,
            )
        n_crop = td_clean["H1"].size
        labels = np.zeros((len(self.detectors), self.n_time), dtype=np.float32)
        is_clean = (not self.force_glitch) and (
            float(rng.random()) < self.clean_frac
        )

        if is_clean:
            td_use = td_clean
            has_glitch = 0
        else:
            spec = sample_glitch_spec(
                rng,
                detectors=list(self.detectors),
                held_out=self.held_out,
            )
            # Prefer stationary ASD for STFT visibility; FD corruption still OK.
            from adapt.glitch_augmentation import GlitchSpec

            spec = GlitchSpec(
                family=spec.family,
                detectors=list(spec.detectors),
                t_rel=spec.t_rel,
                severity=spec.severity,
                params=dict(spec.params),
                asd_policy="stationary",
                held_out=spec.held_out,
            )
            inj_g, _td_raw, _meta = corrupt_injection_fd_with_glitch(
                inj,
                sample_rate=self.sample_rate,
                duration=self.duration,
                spec=spec,
                rng=rng,
            )
            stft_asds = stft_whitening_asds(inj["asds"], inj_g["asds"])
            td_use = {}
            params_g = inj_g.get("parameters") or params
            for det in self.detectors:
                td_use[det] = fd_waveform_to_td_crop(
                    inj_g["waveform"][det],
                    sample_rate=self.sample_rate,
                    duration=self.duration,
                    trigger_time=float(params_g.get(f"{det}_time", geocent)),
                    asd=stft_asds[det],
                    noise_std=self.noise_std,
                    whiten=True,
                )
            for di, det in enumerate(self.detectors):
                if det not in spec.detectors:
                    continue
                sm = glitch_support_mask_on_crop(
                    n_crop=n_crop,
                    sample_rate=self.sample_rate,
                    t_rel=float(spec.t_rel),
                    half_width_s=self.gate_half_s,
                    family=spec.family,
                    params=spec.params,
                )
                labels[di] = support_mask_to_time_bins(sm, self.n_time)
            has_glitch = 1

        spec_kw = {
            k: v
            for k, v in self.stft_kwargs.items()
            if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
        }
        tensor, energy = build_robust_spectrogram_from_td(
            td_use,
            self.sample_rate,
            energy_detectors=tuple(self.detectors),
            norm_stats=self.norm_stats,
            **spec_kw,
        )
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "spectrogram": torch.from_numpy(tensor.astype(np.float32)),
            "labels": torch.from_numpy(labels),
            "has_glitch": torch.tensor(has_glitch, dtype=torch.int64),
            "log_energy": torch.from_numpy(
                np.asarray(energy, dtype=np.float32).ravel()[:3]
            ),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def calibrate_norm_stats(
    metadata: Dict[str, Any],
    *,
    n_samples: int = 64,
    seed: int = 0,
    stft_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        calibrate_robust_norm_stats,
        fd_waveform_to_td_crop,
    )
    from adapt.dingo_bns_demo import build_base_domain_injection

    injection, base = build_base_domain_injection(metadata)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    detectors = list(DETECTORS)
    duration = float(base.duration)
    noise_std = float(base.noise_std)
    stft_kwargs = dict(stft_kwargs or {})
    tensors, energies = [], []
    rng = np.random.default_rng(seed)
    for _ in range(int(n_samples)):
        try:
            import bilby

            bilby.core.utils.random.seed(int(rng.integers(0, 2**31 - 1)))
        except Exception:
            pass
        theta = {k: float(v) for k, v in injection.prior.sample().items()}
        inj = injection.injection(theta)
        params = inj.get("parameters") or {}
        geocent = float(params.get("geocent_time", 0.0))
        td = {
            det: fd_waveform_to_td_crop(
                inj["waveform"][det],
                sample_rate=sample_rate,
                duration=duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=inj["asds"][det],
                noise_std=noise_std,
                whiten=True,
            )
            for det in detectors
        }
        ten, eng = build_robust_spectrogram_from_td(
            td,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=None,
            **{
                k: v
                for k, v in stft_kwargs.items()
                if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
            },
        )
        tensors.append(ten)
        energies.append(eng)
    return calibrate_robust_norm_stats(tensors, energies)


@torch.no_grad()
def eval_detector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0.0
    inter = union = 0.0
    n = 0.0
    clean_fire = 0.0
    n_clean = 0.0
    for batch in loader:
        spec = batch["spectrogram"].to(device)
        lab = batch["labels"].to(device)
        has = batch["has_glitch"].to(device)
        probs = model.predict_probs(spec)
        pred = (probs > float(threshold)).to(dtype=lab.dtype)
        tp += float(((pred == 1) & (lab == 1)).sum())
        fp += float(((pred == 1) & (lab == 0)).sum())
        tn += float(((pred == 0) & (lab == 0)).sum())
        fn += float(((pred == 0) & (lab == 1)).sum())
        inter += float(((pred == 1) & (lab == 1)).sum())
        union += float(((pred == 1) | (lab == 1)).sum())
        # Sample-level clean false positive: any IFO any bin fires on clean.
        clean_idx = has == 0
        if clean_idx.any():
            fires = (probs[clean_idx] > float(threshold)).any(dim=-1).any(dim=-1)
            clean_fire += float(fires.sum())
            n_clean += float(clean_idx.sum())
        n += float(lab.numel())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    iou = inter / max(union, 1.0)
    fpr_clean = clean_fire / max(n_clean, 1.0)
    return {
        "precision": prec,
        "recall": rec,
        "iou": iou,
        "fpr_clean": fpr_clean,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_clean": n_clean,
    }


@torch.no_grad()
def calibrate_threshold(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    *,
    target_fpr: float = 0.01,
    grid: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Pick the highest threshold whose clean sample-FPR ≤ target."""
    model.eval()
    # Collect max prob per clean sample (over ifo × time).
    max_probs: List[float] = []
    for batch in clean_loader:
        spec = batch["spectrogram"].to(device)
        probs = model.predict_probs(spec)
        # Only rows marked clean.
        has = batch["has_glitch"].to(device)
        clean = has == 0
        if not clean.any():
            # Force-clean loader: treat all as clean.
            mp = probs.amax(dim=(-1, -2))
        else:
            mp = probs[clean].amax(dim=(-1, -2))
        max_probs.extend(float(x) for x in mp.detach().cpu().view(-1))
    if not max_probs:
        return 0.5, {"fpr_clean": 1.0, "n_clean": 0.0}
    arr = np.sort(np.asarray(max_probs, dtype=np.float64))
    n = arr.size
    thresholds = (
        list(grid)
        if grid is not None
        else list(np.unique(np.concatenate([[0.05], arr, [0.99]])))
    )
    best_t = 0.99
    best_fpr = 1.0
    for t in sorted(thresholds):
        fpr = float(np.mean(arr > float(t)))
        if fpr <= float(target_fpr):
            # Prefer lower threshold among those that meet FPR (higher recall).
            if t < best_t or best_fpr > target_fpr:
                best_t = float(t)
                best_fpr = fpr
    # If nothing meets target, take quantile.
    if best_fpr > float(target_fpr):
        q = 1.0 - float(target_fpr)
        best_t = float(np.quantile(arr, q))
        best_fpr = float(np.mean(arr > best_t))
    return best_t, {"fpr_clean": best_fpr, "n_clean": float(n)}


def run_training(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from adapt.models import GlitchDetectorSTFT
    from adapt.models.glitch_detector import segmentation_bce_loss
    from adapt.dingo_bns_demo import load_bns_checkpoint

    device = select_device()
    logger.info("Device: %s", device)
    raw = load_bns_checkpoint(Path(args.ckpt))
    metadata = raw["metadata"]
    stft_kwargs = {
        "n_time": int(args.n_time),
        "n_freq": int(args.n_freq),
        "n_fft": int(args.n_fft),
        "win_length": int(args.win_length),
        "hop_length": int(args.hop_length),
    }
    logger.info("Calibrating STFT norm stats (%d samples)…", args.calib_samples)
    norm_stats = calibrate_norm_stats(
        metadata,
        n_samples=int(args.calib_samples),
        seed=int(args.seed) + 99,
        stft_kwargs=stft_kwargs,
    )

    train_len = int(args.steps_per_epoch) * int(args.batch_size)
    train_ds = DetectorPairDataset(
        metadata,
        length=train_len,
        seed=int(args.seed),
        held_out=False,
        clean_frac=float(args.clean_frac),
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        gate_half_s=float(args.gate_half_s),
    )
    val_in_ds = DetectorPairDataset(
        metadata,
        length=int(args.val_size),
        seed=int(args.seed) + 10_000,
        held_out=False,
        clean_frac=0.4,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        gate_half_s=float(args.gate_half_s),
    )
    val_out_ds = DetectorPairDataset(
        metadata,
        length=int(args.val_size),
        seed=int(args.seed) + 20_000,
        held_out=True,
        clean_frac=0.2,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        gate_half_s=float(args.gate_half_s),
    )
    # Pure-clean set for FPR calibration.
    clean_ds = DetectorPairDataset(
        metadata,
        length=max(128, int(args.val_size)),
        seed=int(args.seed) + 30_000,
        held_out=False,
        clean_frac=1.0,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        gate_half_s=float(args.gate_half_s),
    )

    loader_kw = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_in_loader = DataLoader(val_in_ds, shuffle=False, **loader_kw)
    val_out_loader = DataLoader(val_out_ds, shuffle=False, **loader_kw)
    clean_loader = DataLoader(clean_ds, shuffle=False, **loader_kw)

    model = GlitchDetectorSTFT(
        in_channels=6,
        n_ifo=3,
        n_time=int(args.n_time),
        n_freq=int(args.n_freq),
        base_channels=int(args.base_channels),
    ).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1), eta_min=1e-6)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_threshold = 0.5
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for step, batch in enumerate(train_loader, start=1):
            spec = batch["spectrogram"].to(device)
            lab = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(spec)
            # Positive bins are rare → mild pos_weight.
            loss = segmentation_bce_loss(logits, lab, pos_weight=float(args.pos_weight))
            if not torch.isfinite(loss):
                logger.warning("skip non-finite loss epoch %d step %d", epoch, step)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = spec.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs
            if step % max(1, args.log_every) == 0:
                logger.info(
                    "Epoch %d/%d step %d/%d loss=%.4f",
                    epoch,
                    args.epochs,
                    step,
                    args.steps_per_epoch,
                    float(loss.item()),
                )
        sched.step()
        train_loss = running / max(n_seen, 1)

        thr, cal = calibrate_threshold(
            model,
            clean_loader,
            device,
            target_fpr=float(args.target_fpr),
        )
        m_in = eval_detector(model, val_in_loader, device, threshold=thr)
        m_out = eval_detector(model, val_out_loader, device, threshold=thr)
        # Score: held-in IoU + held-out IoU − clean FPR penalty.
        score = (
            0.45 * m_in["iou"]
            + 0.45 * m_out["iou"]
            + 0.10 * m_in["recall"]
            - 2.0 * max(0.0, m_in["fpr_clean"] - float(args.target_fpr))
        )
        logger.info(
            "Epoch %d done | loss=%.4f | thr=%.3f (cal_fpr=%.4f) | "
            "in IoU=%.3f rec=%.3f fpr=%.4f | out IoU=%.3f rec=%.3f | score=%.4f",
            epoch,
            train_loss,
            thr,
            cal["fpr_clean"],
            m_in["iou"],
            m_in["recall"],
            m_in["fpr_clean"],
            m_out["iou"],
            m_out["recall"],
            score,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "threshold": thr,
            "cal_fpr": cal["fpr_clean"],
            "score": score,
            **{f"in_{k}": v for k, v in m_in.items()},
            **{f"out_{k}": v for k, v in m_out.items()},
        }
        history.append(row)

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "threshold": thr,
            "norm_stats": norm_stats,
            "stft_kwargs": stft_kwargs,
            "detectors": list(DETECTORS),
            "gate_half_s": float(args.gate_half_s),
            "target_fpr": float(args.target_fpr),
            "metrics": row,
            "model_kwargs": {
                "in_channels": 6,
                "n_ifo": 3,
                "n_time": int(args.n_time),
                "n_freq": int(args.n_freq),
                "base_channels": int(args.base_channels),
            },
        }
        torch.save(payload, outdir / "last.pt")
        if score > best_score + 1e-4:
            best_score = score
            best_threshold = thr
            torch.save(payload, outdir / "best_glitch_detector.pt")
            logger.info(
                "New best_glitch_detector score=%.4f thr=%.3f", best_score, thr
            )

    summary = {
        "best_score": best_score,
        "best_threshold": best_threshold,
        "history": history,
        "outdir": str(outdir),
    }
    with open(outdir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Training complete. best_score=%.4f thr=%.3f → %s",
        best_score,
        best_threshold,
        outdir,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_BNS_CKPT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--clean-frac", type=float, default=0.35)
    p.add_argument("--pos-weight", type=float, default=3.0)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--gate-half-s", type=float, default=0.4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--n-time", type=int, default=32)
    p.add_argument("--n-freq", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--win-length", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
