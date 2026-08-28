#!/usr/bin/env python3
"""Noise-aware glitch-robust BNS PE (clean≈DINGO, glitch≫DINGO).

Pretrained DINGO is the teacher and init. The student keeps a gated STFT
corrector **and** (by default) a trainable NSF so the PE path itself learns
an explicit noise/glitch dimension:

* clean twins → match teacher (NLL + distill + embedding anchor)
* glitch twins → NLL to true θ (+ soft distill of teacher's clean posterior)

Checkpoints prefer real-event GW170817-glitch ``d_L`` escape under a clean gate.

Usage::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src
    export KMP_DUPLICATE_LIB_OK=TRUE

    caffeinate -dims python scripts/train_bns_glitch_robust.py \\
      --epochs 150 --batch-size 4 --steps-per-epoch 100 \\
      --val-size 256 --lr 1e-4 --outdir checkpoints/glitch_robust_v4
"""

from __future__ import annotations

import argparse
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
_SCRIPTS = REPO_ROOT / "scripts"
_SRC = REPO_ROOT / "src"
_DINGO = REPO_ROOT / "DINGO-BNS" / "dingo"
for _p in (_SCRIPTS, _SRC, _DINGO, REPO_ROOT):
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
DEFAULT_OUTDIR = REPO_ROOT / "checkpoints" / "glitch_robust_v4"
BEST_CLEAN = "best_clean_matched.pt"
BEST_GLITCH = "best_glitch_robust.pt"
LAST_NAME = "last.pt"

CONTEXT_PARAMS = ["ra", "dec", "chirp_mass_proxy"]
PROXY_JITTER_FRAC = 0.01

logger = logging.getLogger("train_bns_glitch_robust")


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PairedGlitchInjectionDataset(Dataset):
    """On-the-fly clean/glitch twins for one θ."""

    def __init__(
        self,
        metadata: Dict[str, Any],
        *,
        length: int,
        seed: int = 0,
        held_out: bool = False,
        force_glitch: bool = True,
        curriculum_severity_max: Optional[float] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        stft_kwargs: Optional[Dict[str, Any]] = None,
        fixed_glitch_specs: Optional[List[Any]] = None,
        hard_eval_frac: float = 0.25,
    ):
        from dingo.gw.domains import build_domain_from_model_metadata
        from train_bns_spectrogram import build_base_domain_injection

        self.metadata = metadata
        self.length = int(length)
        self.base_seed = int(seed)
        self.held_out = bool(held_out)
        self.force_glitch = bool(force_glitch)
        self.curriculum_severity_max = curriculum_severity_max
        self.norm_stats = norm_stats
        self.stft_kwargs = dict(stft_kwargs or {})
        self.fixed_glitch_specs = fixed_glitch_specs
        self.hard_eval_frac = float(hard_eval_frac)

        self.detectors = list(metadata["train_settings"]["data"]["detectors"])
        self.inference_params = list(
            metadata["train_settings"]["data"]["inference_parameters"]
        )
        block = metadata["train_settings"]["data"]["standardization"]
        self.means = {k: float(v) for k, v in block["mean"].items()}
        self.stds = {k: float(v) for k, v in block["std"].items()}
        self.domain = build_domain_from_model_metadata(metadata)
        self.injection, self.base_domain = build_base_domain_injection(metadata)
        self.sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
        self.duration = float(self.base_domain.duration)
        self.noise_std = float(self.base_domain.noise_std)

    def __len__(self) -> int:
        return self.length

    def _sample_theta(self, rng: np.random.Generator) -> Dict[str, float]:
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            import bilby

            bilby.core.utils.random.seed(seed)
        except Exception:
            pass
        return {k: float(v) for k, v in self.injection.prior.sample().items()}

    def _std_vec(self, values: Dict[str, float], keys: Sequence[str]) -> np.ndarray:
        out = np.empty(len(keys), dtype=np.float32)
        for i, k in enumerate(keys):
            mu = self.means.get(k, 0.0)
            sig = self.stds.get(k, 1.0) or 1.0
            out[i] = (float(values[k]) - mu) / sig
        return out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from adapt.glitch_augmentation import (
            STFT_WHITEN_ASD_POLICY,
            corrupt_injection_fd_with_glitch,
            sample_glitch_spec_with_hard_eval,
            stft_whitening_asds,
        )
        from adapt.stft_context import (
            build_robust_spectrogram_from_td,
            fd_waveform_to_td_crop,
        )
        from train_bns_spectrogram import (
            decimate_injection_to_domain,
            package_strain_sample,
        )

        rng = np.random.default_rng(self.base_seed + int(index) * 1009 + 17)
        theta = self._sample_theta(rng)
        mc = float(theta["chirp_mass"])
        jitter = PROXY_JITTER_FRAC * mc * float(rng.normal())
        theta["chirp_mass_proxy"] = mc + jitter
        theta["delta_chirp_mass"] = mc - theta["chirp_mass_proxy"]

        inj_clean = self.injection.injection(theta)
        inj_clean_mfd = decimate_injection_to_domain(inj_clean, self.domain)
        strain_clean = package_strain_sample(
            inj_clean_mfd, self.detectors, self.domain
        )

        # Clean STFT from whitened FD→TD crops (same as prior signal-in-STFT).
        td_clean = {}
        params = inj_clean.get("parameters") or {}
        geocent = float(params.get("geocent_time", 0.0))
        for det in self.detectors:
            td_clean[det] = fd_waveform_to_td_crop(
                inj_clean["waveform"][det],
                sample_rate=self.sample_rate,
                duration=self.duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=inj_clean["asds"][det],
                noise_std=self.noise_std,
                whiten=True,
            )
        spec_kw = dict(self.stft_kwargs)
        spec_clean, e_clean = build_robust_spectrogram_from_td(
            td_clean,
            self.sample_rate,
            energy_detectors=tuple(self.detectors),
            norm_stats=self.norm_stats,
            **{k: v for k, v in spec_kw.items() if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")},
        )

        if self.fixed_glitch_specs is not None:
            spec = self.fixed_glitch_specs[int(index) % len(self.fixed_glitch_specs)]
        else:
            spec = sample_glitch_spec_with_hard_eval(
                rng,
                detectors=self.detectors,
                held_out=self.held_out,
                curriculum_severity_max=self.curriculum_severity_max,
                hard_eval_frac=float(getattr(self, "hard_eval_frac", 0.25)),
            )

        inj_g, td_g_raw, gmeta = corrupt_injection_fd_with_glitch(
            inj_clean,
            sample_rate=self.sample_rate,
            duration=self.duration,
            spec=spec,
            rng=rng,
        )
        # CRITICAL: whiten STFT with clean/stationary ASD, never Welch.
        # FD packaging still uses inj_g ASDs (Welch when selected).
        stft_asds = stft_whitening_asds(inj_clean["asds"], inj_g["asds"])
        assert STFT_WHITEN_ASD_POLICY == "stationary_clean"
        td_g = {}
        params_g = inj_g.get("parameters") or params
        for det in self.detectors:
            td_g[det] = fd_waveform_to_td_crop(
                inj_g["waveform"][det],
                sample_rate=self.sample_rate,
                duration=self.duration,
                trigger_time=float(params_g.get(f"{det}_time", geocent)),
                asd=stft_asds[det],
                noise_std=self.noise_std,
                whiten=True,
            )
        inj_g_mfd = decimate_injection_to_domain(inj_g, self.domain)
        strain_g = package_strain_sample(inj_g_mfd, self.detectors, self.domain)
        spec_g, e_g = build_robust_spectrogram_from_td(
            td_g,
            self.sample_rate,
            energy_detectors=tuple(self.detectors),
            norm_stats=self.norm_stats,
            **{k: v for k, v in spec_kw.items() if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")},
        )

        y = self._std_vec(theta, self.inference_params)
        context_z = self._std_vec(theta, CONTEXT_PARAMS)

        def _clean(a: np.ndarray) -> np.ndarray:
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "strain_clean": torch.from_numpy(_clean(strain_clean)),
            "strain_glitch": torch.from_numpy(_clean(strain_g)),
            "spectrogram_clean": torch.from_numpy(_clean(spec_clean)),
            "spectrogram_glitch": torch.from_numpy(_clean(spec_g)),
            "log_energy_clean": torch.from_numpy(_clean(e_clean).astype(np.float32)),
            "log_energy_glitch": torch.from_numpy(_clean(e_g).astype(np.float32)),
            "y": torch.from_numpy(_clean(y)),
            "context_z": torch.from_numpy(_clean(context_z)),
            "severity": torch.tensor(float(spec.severity), dtype=torch.float32),
            "held_out": torch.tensor(1 if spec.held_out else 0, dtype=torch.int64),
        }


def collate_paired(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


# ---------------------------------------------------------------------------
# Model surgery / losses
# ---------------------------------------------------------------------------


def attach_glitch_robust_embedding(
    wrapper: nn.Module,
    *,
    encoder_type: str = "resnet_deep",
    encoder_channels: Optional[Sequence[int]] = None,
    n_time: int = 32,
    n_spec_freq: int = 128,
    in_channels: int = 6,
) -> nn.Module:
    from adapt.models import ContextAwareGlitchCorrector, GlitchRobustBNSEmbedding

    corrector = ContextAwareGlitchCorrector(
        n_time=int(n_time),
        n_spec_freq=int(n_spec_freq),
        in_channels=int(in_channels),
        encoder_type=encoder_type,
        encoder_channels=encoder_channels,
    )
    wrapper.embedding_net = GlitchRobustBNSEmbedding(
        wrapper.embedding_net, corrector=corrector
    )
    return wrapper


def configure_trainable(
    wrapper: nn.Module,
    *,
    train_flow: bool = True,
    train_base_embedding: bool = False,
) -> None:
    """Freeze RB by default; train corrector (+ optional NSF / base embed)."""
    for p in wrapper.parameters():
        p.requires_grad = False
    emb = wrapper.embedding_net
    if hasattr(emb, "trainable_residual_parameters"):
        for p in emb.trainable_residual_parameters():
            p.requires_grad = True
    if train_flow:
        for p in wrapper.flow.parameters():
            p.requires_grad = True
    if train_base_embedding and hasattr(emb, "base_embedding"):
        for p in emb.base_embedding.parameters():
            p.requires_grad = True
    # Keep frozen submodules in eval for stable BN/RB stats.
    if hasattr(emb, "base_embedding") and not train_base_embedding:
        emb.base_embedding.eval()
    if not train_flow:
        wrapper.flow.eval()


def freeze_dingo(wrapper: nn.Module) -> None:
    """Backward-compatible: corrector-only (frozen NSF)."""
    configure_trainable(wrapper, train_flow=False, train_base_embedding=False)


def teacher_context(
    teacher: nn.Module,
    strain: torch.Tensor,
    context_z: torch.Tensor,
) -> torch.Tensor:
    """Frozen original DINGO embedding → 131-D."""
    with torch.no_grad():
        return teacher.embedding_net(strain, context_z)


def paired_loss(
    student: nn.Module,
    teacher: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    lambda_nll: float = 1.0,
    lambda_nll_clean: float = 1.0,
    lambda_anchor: float = 5.0,
    lambda_ctxhat: float = 2.0,
    lambda_teacher: float = 2.0,
    lambda_gate_bce: float = 1.0,
    lambda_distill: float = 2.0,
    lambda_reg: float = 0.01,
    use_nll: bool = True,
    use_distill_samples: bool = False,
    n_distill_samples: int = 8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Noise-aware paired loss: clean≈DINGO, glitch NLL to true θ.

    Auxiliary embedding reconstruction toward ``DINGO(clean)`` remains, but the
    primary glitch signal is the student's NSF log-prob of the true parameters
    under glitchy inputs (the extra noise dimension).
    """
    y = batch["y"].to(device)
    z = batch["context_z"].to(device)
    sc = batch["strain_clean"].to(device)
    sg = batch["strain_glitch"].to(device)
    pc = batch["spectrogram_clean"].to(device)
    pg = batch["spectrogram_glitch"].to(device)
    ec = batch["log_energy_clean"].to(device)
    eg = batch["log_energy_glitch"].to(device)

    emb = student.embedding_net
    diag_c = emb.forward_with_diagnostics(sc, pc, ec, z)
    diag_g = emb.forward_with_diagnostics(sg, pg, eg, z)

    t_clean = teacher_context(teacher, sc, z)
    t_embed = t_clean[:, :128]

    def _emb_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).pow(2).sum(dim=-1).mean()

    loss_anchor = (
        _emb_mse(diag_c["corrected_embed"], t_embed)
        + _emb_mse(diag_c["ctx_hat"], t_embed)
        + diag_c["gate"].pow(2).mean()
    )
    loss_teacher = _emb_mse(diag_g["corrected_embed"], t_embed)
    loss_ctxhat = _emb_mse(diag_g["ctx_hat"], t_embed)
    gate_logits = torch.cat([diag_c["gate_logit"], diag_g["gate_logit"]], dim=0)
    gate_targets = torch.cat(
        [
            torch.zeros_like(diag_c["gate_logit"]),
            torch.ones_like(diag_g["gate_logit"]),
        ],
        dim=0,
    )
    loss_gate_bce = nn.functional.binary_cross_entropy_with_logits(
        gate_logits, gate_targets
    )
    loss_reg = diag_c["delta"].abs().mean()

    loss_nll_glitch = torch.zeros((), device=device)
    loss_nll_clean = torch.zeros((), device=device)
    if use_nll:
        lp_g = torch.clamp(student.log_prob(y, sg, pg, eg, z), min=-1.0e4, max=1.0e4)
        loss_nll_glitch = -lp_g.mean()
        # Clean NLL must NOT go through the STFT corrector — otherwise the NSF
        # learns to read θ from spectrograms and destroys DINGO clean parity.
        # Use the frozen RB base embedding (same as teacher) with the student flow.
        if hasattr(emb, "base_embedding"):
            base_ctx = emb.base_embedding(sc, z)
            lp_c = torch.clamp(
                student.flow.log_prob(y, base_ctx), min=-1.0e4, max=1.0e4
            )
        else:
            lp_c = torch.clamp(
                student.log_prob(y, sc, pc, ec, z), min=-1.0e4, max=1.0e4
            )
        loss_nll_clean = -lp_c.mean()

    loss_distill = torch.zeros((), device=device)
    if use_distill_samples and n_distill_samples > 0:
        # Teacher samples on clean; student must assign high prob under glitch
        # (treat glitch as nuisance) and under clean (parity).
        with torch.no_grad():
            chunks = []
            for _ in range(int(n_distill_samples)):
                y_s = teacher.sample(sc, z, num_samples=1)
                if y_s.ndim == 1:
                    y_s = y_s.unsqueeze(0)
                chunks.append(y_s)
            samples = torch.stack(chunks, dim=1)  # (B, N, D)
        b, n_s, d = samples.shape
        y_flat = samples.reshape(b * n_s, d)
        sg_r = sg.repeat_interleave(n_s, dim=0)
        pg_r = pg.repeat_interleave(n_s, dim=0)
        eg_r = eg.repeat_interleave(n_s, dim=0)
        sc_r = sc.repeat_interleave(n_s, dim=0)
        z_r = z.repeat_interleave(n_s, dim=0)
        lp_g = torch.clamp(
            student.log_prob(y_flat, sg_r, pg_r, eg_r, z_r), min=-1.0e4, max=1.0e4
        )
        if hasattr(emb, "base_embedding"):
            base_c = emb.base_embedding(sc_r, z_r)
            lp_c = torch.clamp(
                student.flow.log_prob(y_flat, base_c), min=-1.0e4, max=1.0e4
            )
        else:
            pc_r = pc.repeat_interleave(n_s, dim=0)
            ec_r = ec.repeat_interleave(n_s, dim=0)
            lp_c = torch.clamp(
                student.log_prob(y_flat, sc_r, pc_r, ec_r, z_r), min=-1.0e4, max=1.0e4
            )
        loss_distill = -0.5 * (lp_g.mean() + lp_c.mean())

    total = (
        float(lambda_nll) * loss_nll_glitch
        + float(lambda_nll_clean) * loss_nll_clean
        + float(lambda_anchor) * loss_anchor
        + float(lambda_ctxhat) * loss_ctxhat
        + float(lambda_teacher) * loss_teacher
        + float(lambda_gate_bce) * loss_gate_bce
        + float(lambda_reg) * loss_reg
        + (float(lambda_distill) if use_distill_samples else 0.0) * loss_distill
    )
    stats = {
        "loss": float(total.detach()),
        "nll": float(loss_nll_glitch.detach()),
        "nll_clean": float(loss_nll_clean.detach()),
        "anchor": float(loss_anchor.detach()),
        "ctxhat": float(loss_ctxhat.detach()),
        "teacher": float(loss_teacher.detach()),
        "gate_bce": float(loss_gate_bce.detach()),
        "reg": float(loss_reg.detach()),
        "distill": float(loss_distill.detach()),
        "gate_clean": float(diag_c["gate"].mean().detach()),
        "gate_glitch": float(diag_g["gate"].mean().detach()),
        "egate_glitch": float(diag_g["energy_gate"].mean().detach()),
    }
    return total, stats


@torch.no_grad()
def eval_paired_metrics(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()
    totals = {
        "nll_clean_student": 0.0,
        "nll_clean_teacher": 0.0,
        "nll_glitch_student": 0.0,
        "nll_glitch_teacher": 0.0,
        "consist": 0.0,
        "teacher_mse": 0.0,
        "ctxhat_mse": 0.0,
        "gate_clean": 0.0,
        "gate_glitch": 0.0,
        "egate_glitch": 0.0,
        "n": 0.0,
    }
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        y = batch["y"].to(device)
        z = batch["context_z"].to(device)
        sc = batch["strain_clean"].to(device)
        sg = batch["strain_glitch"].to(device)
        pc = batch["spectrogram_clean"].to(device)
        pg = batch["spectrogram_glitch"].to(device)
        ec = batch["log_energy_clean"].to(device)
        eg = batch["log_energy_glitch"].to(device)
        bs = y.shape[0]

        # Teacher NLL uses (strain, z) only.
        lp_tc = teacher.log_prob(y, sc, z)
        lp_tg = teacher.log_prob(y, sg, z)
        lp_sc = student.log_prob(y, sc, pc, ec, z)
        lp_sg = student.log_prob(y, sg, pg, eg, z)
        for t in (lp_tc, lp_tg, lp_sc, lp_sg):
            t.clamp_(-1e4, 1e4)

        diag_c = student.embedding_net.forward_with_diagnostics(sc, pc, ec, z)
        diag_g = student.embedding_net.forward_with_diagnostics(sg, pg, eg, z)
        t_embed = teacher_context(teacher, sc, z)[:, :128]

        totals["nll_clean_student"] += float((-lp_sc).sum())
        totals["nll_clean_teacher"] += float((-lp_tc).sum())
        totals["nll_glitch_student"] += float((-lp_sg).sum())
        totals["nll_glitch_teacher"] += float((-lp_tg).sum())
        totals["consist"] += float(
            (diag_g["corrected_embed"] - diag_c["base_embed"]).pow(2).sum()
        )
        totals["teacher_mse"] += float(
            (diag_g["corrected_embed"] - t_embed).pow(2).sum()
        )
        totals["ctxhat_mse"] += float((diag_g["ctx_hat"] - t_embed).pow(2).sum())
        totals["gate_clean"] += float(diag_c["gate"].sum())
        totals["gate_glitch"] += float(diag_g["gate"].sum())
        totals["egate_glitch"] += float(diag_g["energy_gate"].sum())
        totals["n"] += bs

    n = max(totals["n"], 1.0)
    return {k: (v / n if k != "n" else v) for k, v in totals.items()}


def calibrate_norm_stats(
    metadata: Dict[str, Any],
    *,
    n_samples: int = 64,
    seed: int = 123,
    stft_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        calibrate_robust_norm_stats,
        fd_waveform_to_td_crop,
    )
    from train_bns_spectrogram import build_base_domain_injection

    injection, base = build_base_domain_injection(metadata)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    duration = float(base.duration)
    noise_std = float(base.noise_std)
    stft_kwargs = dict(stft_kwargs or {})
    tensors = []
    energies = []
    rng = np.random.default_rng(seed)
    for i in range(int(n_samples)):
        try:
            import bilby

            bilby.core.utils.random.seed(int(rng.integers(0, 2**31 - 1)))
        except Exception:
            pass
        theta = {k: float(v) for k, v in injection.prior.sample().items()}
        inj = injection.injection(theta)
        params = inj.get("parameters") or {}
        geocent = float(params.get("geocent_time", 0.0))
        td = {}
        for det in detectors:
            td[det] = fd_waveform_to_td_crop(
                inj["waveform"][det],
                sample_rate=sample_rate,
                duration=duration,
                trigger_time=float(params.get(f"{det}_time", geocent)),
                asd=inj["asds"][det],
                noise_std=noise_std,
                whiten=True,
            )
        tens, eng = build_robust_spectrogram_from_td(
            td,
            sample_rate,
            energy_detectors=tuple(detectors),
            norm_stats=None,
            **{k: v for k, v in stft_kwargs.items() if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")},
        )
        tensors.append(tens)
        energies.append(eng)
    return calibrate_robust_norm_stats(tensors, energies)


def save_ckpt(
    path: Path,
    *,
    student: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    adapt_config: Dict[str, Any],
    norm_stats: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "embedding_state_dict": student.embedding_net.state_dict(),
        "flow_state_dict": student.flow.state_dict(),
        "spectrogram_net_state_dict": student.embedding_net.corrector.state_dict(),
        "metadata": {
            "inference_parameters": metadata["train_settings"]["data"][
                "inference_parameters"
            ],
            "context_parameters": CONTEXT_PARAMS,
            "standardization": metadata["train_settings"]["data"]["standardization"],
            "detectors": metadata["train_settings"]["data"]["detectors"],
            "adapt_config": adapt_config,
            "norm_stats": norm_stats,
            "residual_dingo": True,
            "glitch_robust": True,
            "context_replace": bool(adapt_config.get("context_replace", True)),
            "csd_channels": adapt_config.get("csd_channels", 6),
            "spectrogram_layout": adapt_config.get("spectrogram_layout", "hlv_coh_6ch"),
            "n_time": adapt_config.get("n_time", 32),
            "n_freq": adapt_config.get("n_freq", 128),
            "encoder_type": adapt_config.get("encoder_type", "resnet_deep"),
        },
    }
    torch.save(payload, path)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Training loop (includes dual validation + constrained selection)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_gw170817_glitch_dl(
    student: nn.Module,
    metadata: Dict[str, Any],
    adapt_config: Dict[str, Any],
    norm_stats: Optional[Dict[str, Any]],
    device: torch.device,
    *,
    num_samples: int = 256,
    batch_size: int = 128,
    snr_amp_scale: float = 8.0,
) -> Dict[str, float]:
    """Cheap real-event kill metric: custom d_L CI on glitchy GW170817."""
    from evaluate_glitch_robustness import inject_h1_glitch_into_event
    from evaluate_gw170817_comparison import (
        build_event_spectrogram_stack,
        discover_assets,
        load_event_dataset,
        package_event_strain,
        run_custom_sampling,
        standardize_context,
    )
    from adapt.glitch_augmentation import stft_whitening_asds
    from dingo.gw.domains import build_domain_from_model_metadata

    # discover_assets requires some custom ckpt path on disk; we sample with the
    # in-memory student, so any existing glitch/custom checkpoint is fine.
    probe = None
    for cand in (
        REPO_ROOT / "checkpoints" / "glitch_robust_v3" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt",
        REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt",
    ):
        if cand.is_file():
            probe = cand
            break
    assets = discover_assets(baseline_ckpt=None, custom_ckpt=probe)
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]
    glitchy_data, td_stft, _meta = inject_h1_glitch_into_event(
        event, assets, snr_amp_scale=float(snr_amp_scale)
    )
    strain = package_event_strain(glitchy_data, metadata, fixed)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    sample_rate = float(
        event.settings.get("f_s")
        or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    base_domain = build_domain_from_model_metadata(metadata, base=True)
    stft_asds = stft_whitening_asds(
        {d: np.asarray(event.data["asds"][d]) for d in detectors},
        {d: np.asarray(glitchy_data["asds"][d]) for d in detectors},
    )
    spectrogram, log_energy = build_event_spectrogram_stack(
        td_stft,
        detectors,
        sample_rate,
        asds=stft_asds,
        delta_f=float(base_domain.delta_f),
        noise_std=float(base_domain.noise_std),
        robust=True,
        norm_stats=norm_stats,
        n_time=adapt_config.get("n_time"),
        n_freq=adapt_config.get("n_freq"),
        n_fft=adapt_config.get("n_fft"),
        win_length=adapt_config.get("win_length"),
        hop_length=adapt_config.get("hop_length"),
    )
    context_z = standardize_context(
        fixed, metadata["train_settings"]["data"]["standardization"]
    )
    # Temporarily stash metadata fields expected by run_custom_sampling
    meta_view = dict(metadata)
    meta_view["_glitch_robust"] = True
    meta_view["_norm_stats"] = norm_stats
    meta_view["_stft_kwargs"] = {
        "n_time": adapt_config.get("n_time"),
        "n_freq": adapt_config.get("n_freq"),
        "n_fft": adapt_config.get("n_fft"),
        "win_length": adapt_config.get("win_length"),
        "hop_length": adapt_config.get("hop_length"),
    }
    meta_view["train_settings"] = metadata["train_settings"]
    df = run_custom_sampling(
        student,
        meta_view,
        strain,
        spectrogram,
        log_energy,
        context_z,
        fixed,
        device=device,
        num_samples=int(num_samples),
        batch_size=int(batch_size),
    )
    x = np.asarray(df["luminosity_distance"], dtype=np.float64)
    x = x[np.isfinite(x)]
    lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
    return {
        "dl_lo": float(lo),
        "dl_med": float(med),
        "dl_hi": float(hi),
        "logE_h1": float(np.asarray(log_energy).ravel()[0]),
        "escaped": float(med > 20.0 and hi > 25.0),
    }


def run_training(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from train_bns_spectrogram import build_flow_wrapper, load_bns_checkpoint

    device = select_device()
    logger.info("Device: %s", device)

    raw = load_bns_checkpoint(Path(args.ckpt))
    metadata = raw["metadata"]
    teacher = build_flow_wrapper(raw).float().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_flow_wrapper(raw)
    stft_kwargs = {
        "n_time": int(args.n_time),
        "n_freq": int(args.n_freq),
        "n_fft": int(args.n_fft),
        "win_length": int(args.win_length),
        "hop_length": int(args.hop_length),
    }
    logger.info("Calibrating robust STFT norm stats (%d samples)...", args.calib_samples)
    norm_stats = calibrate_norm_stats(
        metadata,
        n_samples=int(args.calib_samples),
        seed=int(args.seed) + 999,
        stft_kwargs=stft_kwargs,
    )
    student = attach_glitch_robust_embedding(
        student,
        encoder_type=str(args.encoder_type),
        encoder_channels=list(args.encoder_channels),
        n_time=int(args.n_time),
        n_spec_freq=int(args.n_freq),
        in_channels=6,
    )
    student = student.float().to(device)
    train_flow_flag = bool(args.train_flow)
    # Warm up corrector first; NSF unfreezes at --flow-start-epoch.
    configure_trainable(
        student,
        train_flow=False,
        train_base_embedding=bool(args.train_base_embedding),
    )
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(
        "Trainable params (warmup): %.2fM (train_flow eventually=%s from epoch %d)",
        n_train / 1e6,
        train_flow_flag,
        int(args.flow_start_epoch),
    )
    corrector = student.embedding_net.corrector

    adapt_config = {
        "encoder_type": str(args.encoder_type),
        "encoder_channels": list(args.encoder_channels),
        "n_time": int(args.n_time),
        "n_freq": int(args.n_freq),
        "n_fft": int(args.n_fft),
        "win_length": int(args.win_length),
        "hop_length": int(args.hop_length),
        "csd_channels": 6,
        "spectrogram_layout": "hlv_coh_6ch",
        "signal_in_stft": True,
        "residual_dingo": True,
        "glitch_robust": True,
        "noise_aware_student": True,
        "train_flow": train_flow_flag,
        "flow_start_epoch": int(args.flow_start_epoch),
        "train_base_embedding": bool(args.train_base_embedding),
        "stft_whiten_asd": "stationary_clean",
        "hard_eval_frac": float(args.hard_eval_frac),
        "welch_asd_train_frac": 0.75,
        "context_replace": True,
        "max_delta": float(corrector.max_delta),
        "energy_gate_center": float(corrector.energy_gate_center),
        "energy_gate_scale": float(corrector.energy_gate_scale),
        "lr": float(args.lr),
        "lambda_nll": float(args.lambda_nll),
        "lambda_nll_clean": float(args.lambda_nll_clean),
        "lambda_anchor": float(args.lambda_anchor),
        "lambda_ctxhat": float(args.lambda_ctxhat),
        "lambda_teacher": float(args.lambda_teacher),
        "lambda_gate_bce": float(args.lambda_gate_bce),
        "lambda_distill": float(args.lambda_distill),
        "clean_gate_tol": float(args.clean_gate_tol),
    }

    from adapt.glitch_augmentation import make_fixed_eval_glitch

    train_len = int(args.steps_per_epoch) * int(args.batch_size)
    train_ds = PairedGlitchInjectionDataset(
        metadata,
        length=train_len,
        seed=int(args.seed),
        held_out=False,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        curriculum_severity_max=float(args.severity_start),
        hard_eval_frac=float(args.hard_eval_frac),
    )
    val_clean_ds = PairedGlitchInjectionDataset(
        metadata,
        length=int(args.val_size),
        seed=int(args.seed) + 10_000,
        held_out=False,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        curriculum_severity_max=6.0,
        hard_eval_frac=0.0,
    )
    val_holdout_ds = PairedGlitchInjectionDataset(
        metadata,
        length=int(args.val_size),
        seed=int(args.seed) + 20_000,
        held_out=True,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        curriculum_severity_max=None,
        hard_eval_frac=0.0,
    )
    # Locked GW170817-style hard val (H1 SG, t≈-1, Welch FD, clean STFT ASD).
    n_hard = max(32, int(args.val_size) // 4)
    hard_specs = [
        make_fixed_eval_glitch(
            severity=float(6.0 + (i % 7)),
            t_rel=-1.0,
            asd_policy="welch",
            f0=100.0,
            q=5.0,
        )
        for i in range(n_hard)
    ]
    val_hard_ds = PairedGlitchInjectionDataset(
        metadata,
        length=n_hard,
        seed=int(args.seed) + 30_000,
        held_out=False,
        norm_stats=norm_stats,
        stft_kwargs=stft_kwargs,
        fixed_glitch_specs=hard_specs,
        hard_eval_frac=0.0,
    )

    loader_kw = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate_paired,
    )
    if int(args.num_workers) > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_clean_loader = DataLoader(val_clean_ds, shuffle=False, **loader_kw)
    val_holdout_loader = DataLoader(val_holdout_ds, shuffle=False, **loader_kw)
    val_hard_loader = DataLoader(val_hard_ds, shuffle=False, **loader_kw)

    corr_params = list(student.embedding_net.corrector.parameters())
    flow_params = list(student.flow.parameters())
    param_groups = [
        {"params": corr_params, "lr": float(args.lr)},
        {
            "params": flow_params,
            "lr": float(args.lr) * float(args.flow_lr_mult),
        },
    ]
    opt = AdamW(param_groups, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1), eta_min=1e-7)
    # Flow params start frozen; optimizer still holds them for later unfreeze.

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best_clean_gap = float("inf")
    best_glitch_score = float("inf")
    best_eligible_glitch = float("inf")
    best_dl_med = -float("inf")
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        # Curriculum: severity + enable NLL/distill later.
        frac = epoch / max(int(args.epochs), 1)
        sev_max = float(args.severity_start) + frac * (
            float(args.severity_end) - float(args.severity_start)
        )
        train_ds.curriculum_severity_max = sev_max
        use_nll = epoch >= int(args.nll_start_epoch)
        use_distill = epoch >= int(args.distill_start_epoch)

        train_flow_now = bool(train_flow_flag) and epoch >= int(args.flow_start_epoch)
        student.train()
        configure_trainable(
            student,
            train_flow=train_flow_now,
            train_base_embedding=bool(args.train_base_embedding),
        )
        student.embedding_net.corrector.train()
        if train_flow_now:
            student.flow.train()
        else:
            student.flow.eval()
        running = 0.0
        n_seen = 0
        for step, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            loss, stats = paired_loss(
                student,
                teacher,
                batch,
                device,
                lambda_nll=float(args.lambda_nll),
                lambda_nll_clean=float(args.lambda_nll_clean),
                lambda_anchor=float(args.lambda_anchor),
                lambda_ctxhat=float(args.lambda_ctxhat),
                lambda_teacher=float(args.lambda_teacher),
                lambda_gate_bce=float(args.lambda_gate_bce),
                lambda_distill=float(args.lambda_distill),
                lambda_reg=float(args.lambda_reg),
                use_nll=use_nll,
                use_distill_samples=use_distill,
                n_distill_samples=int(args.n_distill_samples),
            )
            if not torch.isfinite(loss):
                logger.warning("Skip non-finite loss epoch %d step %d", epoch, step)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            bs = batch["y"].shape[0]
            running += float(loss.item()) * bs
            n_seen += bs
            if step % max(1, args.log_every) == 0:
                logger.info(
                    "Epoch %d/%d step %d/%d loss=%.4f nll_g=%.3f nll_c=%.3f "
                    "dst=%.3f anc=%.4f cxh=%.4f tch=%.4f bce=%.4f "
                    "g_c=%.3f g_g=%.3f eg_g=%.3f sev<=%.1f",
                    epoch,
                    args.epochs,
                    step,
                    args.steps_per_epoch,
                    stats["loss"],
                    stats["nll"],
                    stats["nll_clean"],
                    stats["distill"],
                    stats["anchor"],
                    stats["ctxhat"],
                    stats["teacher"],
                    stats["gate_bce"],
                    stats["gate_clean"],
                    stats["gate_glitch"],
                    stats["egate_glitch"],
                    sev_max,
                )
        sched.step()
        train_loss = running / max(n_seen, 1)

        # Dual validation (+ locked hard Welch H1 SG twin)
        logger.info("Validating clean / held-out / hard-eval …")
        m_clean = eval_paired_metrics(
            student, teacher, val_clean_loader, device, max_batches=None
        )
        m_hold = eval_paired_metrics(
            student, teacher, val_holdout_loader, device, max_batches=None
        )
        m_hard = eval_paired_metrics(
            student, teacher, val_hard_loader, device, max_batches=None
        )
        clean_gap = m_clean["nll_clean_student"] - m_clean["nll_clean_teacher"]
        # Secondary synthetic score (not the primary selection criterion).
        glitch_score = (
            0.70 * m_hard["nll_glitch_student"]
            + 0.15 * m_hold["nll_glitch_student"]
            + 0.08 * m_hard["teacher_mse"]
            + 0.05 * m_hard["ctxhat_mse"]
            + 0.02 * m_hold["consist"]
        )
        hard_improve = m_hard["nll_glitch_teacher"] - m_hard["nll_glitch_student"]

        dl_metrics: Dict[str, float] = {}
        if int(args.dl_eval_every) > 0 and (
            epoch == 1 or epoch % int(args.dl_eval_every) == 0
        ):
            logger.info("Mid-run GW170817 glitch d_L eval …")
            try:
                dl_metrics = eval_gw170817_glitch_dl(
                    student,
                    metadata,
                    adapt_config,
                    norm_stats,
                    device,
                    num_samples=int(args.dl_eval_samples),
                    batch_size=min(128, int(args.dl_eval_samples)),
                    snr_amp_scale=float(args.dl_snr_amp_scale),
                )
                logger.info(
                    "GW170817 glitch d_L 90%% CI [%.2f, %.2f] med=%.2f "
                    "logE_H1=%.2f escaped=%s",
                    dl_metrics["dl_lo"],
                    dl_metrics["dl_hi"],
                    dl_metrics["dl_med"],
                    dl_metrics["logE_h1"],
                    bool(dl_metrics["escaped"]),
                )
            except Exception as exc:
                logger.warning("GW170817 d_L eval failed: %s", exc)
                dl_metrics = {}

        logger.info(
            "Epoch %d done | train_loss=%.4f | clean_gap=%.4f "
            "(stu=%.3f tea=%.3f) | hold_glitch stu=%.3f tea=%.3f | "
            "hard_glitch stu=%.3f tea=%.3f Δ=%.3f | score=%.3f "
            "gate_c/g/h=%.3f/%.3f/%.3f egate_h=%.3f ctxhat_h=%.4f",
            epoch,
            train_loss,
            clean_gap,
            m_clean["nll_clean_student"],
            m_clean["nll_clean_teacher"],
            m_hold["nll_glitch_student"],
            m_hold["nll_glitch_teacher"],
            m_hard["nll_glitch_student"],
            m_hard["nll_glitch_teacher"],
            hard_improve,
            glitch_score,
            m_clean["gate_clean"],
            m_hold["gate_glitch"],
            m_hard["gate_glitch"],
            m_hard["egate_glitch"],
            m_hard["ctxhat_mse"],
        )

        metrics = {
            "train_loss": train_loss,
            "clean_gap": clean_gap,
            **{f"clean_{k}": v for k, v in m_clean.items()},
            **{f"hold_{k}": v for k, v in m_hold.items()},
            **{f"hard_{k}": v for k, v in m_hard.items()},
            "hard_improve": hard_improve,
            "glitch_score": glitch_score,
            "sev_max": sev_max,
            **{f"event_{k}": v for k, v in dl_metrics.items()},
        }
        save_ckpt(
            outdir / LAST_NAME,
            student=student,
            epoch=epoch,
            metrics=metrics,
            metadata=metadata,
            adapt_config=adapt_config,
            norm_stats=norm_stats,
        )

        # Best clean-matched: smallest |clean gap| (prefer near-zero residual).
        if abs(clean_gap) < abs(best_clean_gap) - 1e-4:
            best_clean_gap = clean_gap
            save_ckpt(
                outdir / BEST_CLEAN,
                student=student,
                epoch=epoch,
                metrics=metrics,
                metadata=metadata,
                adapt_config=adapt_config,
                norm_stats=norm_stats,
            )
            logger.info("New best_clean_matched |gap|=%.4f", abs(clean_gap))

        # Primary selection: clean gate + best real-event d_L median; else score.
        # Two-sided: large *negative* gap means the student is cheating via STFT
        # (reading θ from spectrograms) rather than matching DINGO on clean.
        eligible = abs(clean_gap) <= float(args.clean_gate_tol)
        improved = False
        if eligible and dl_metrics:
            med = float(dl_metrics["dl_med"])
            if med > best_dl_med + 0.05:
                best_dl_med = med
                improved = True
                save_ckpt(
                    outdir / BEST_GLITCH,
                    student=student,
                    epoch=epoch,
                    metrics=metrics,
                    metadata=metadata,
                    adapt_config=adapt_config,
                    norm_stats=norm_stats,
                )
                logger.info(
                    "New best_glitch_robust by event d_L med=%.3f "
                    "(clean_gap=%.4f <= tol)",
                    med,
                    clean_gap,
                )
        if eligible and not improved and glitch_score < best_eligible_glitch - 1e-4:
            # Only use synthetic score if we have never seen a better d_L ckpt.
            if best_dl_med < 0:
                best_eligible_glitch = glitch_score
                improved = True
                save_ckpt(
                    outdir / BEST_GLITCH,
                    student=student,
                    epoch=epoch,
                    metrics=metrics,
                    metadata=metadata,
                    adapt_config=adapt_config,
                    norm_stats=norm_stats,
                )
                logger.info(
                    "New best_glitch_robust score=%.4f (clean_gap=%.4f <= tol)",
                    glitch_score,
                    clean_gap,
                )
        if improved:
            bad_epochs = 0
        else:
            if glitch_score < best_glitch_score:
                best_glitch_score = glitch_score
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                logger.info(
                    "Early stop: no eligible improvement for %d epochs "
                    "(clean_gate_tol=%.3f, best_dl_med=%.3f)",
                    bad_epochs,
                    args.clean_gate_tol,
                    best_dl_med,
                )
                break

        # Kill switch after warmup: clean parity lost in either direction.
        if epoch >= int(args.flow_start_epoch) and abs(clean_gap) > float(
            args.clean_kill_gap
        ):
            logger.error(
                "Killing run: |clean_gap|=%.3f > %.3f (clean parity lost)",
                abs(clean_gap),
                args.clean_kill_gap,
            )
            break

    logger.info(
        "Training complete. best_clean_gap=%.4f best_dl_med=%.3f "
        "best_eligible_glitch=%.4f → %s",
        best_clean_gap,
        best_dl_med,
        best_eligible_glitch,
        outdir,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINGO-preserving glitch-robust trainer")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_BNS_CKPT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument(
        "--clean-gate-tol",
        type=float,
        default=2.0,
        help="Eligible ckpt requires |clean_gap| <= this (two-sided)",
    )
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--encoder-type", type=str, default="resnet_deep")
    p.add_argument(
        "--encoder-channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    p.add_argument("--n-time", type=int, default=32)
    p.add_argument("--n-freq", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--win-length", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    # Noise-aware defaults: glitch true-θ NLL is first-class; embed MSE is aux.
    p.add_argument("--lambda-nll", type=float, default=1.0, help="Glitch true-θ NLL")
    p.add_argument(
        "--lambda-nll-clean",
        type=float,
        default=1.0,
        help="Clean true-θ NLL (DINGO parity)",
    )
    p.add_argument("--lambda-anchor", type=float, default=5.0)
    p.add_argument(
        "--lambda-ctxhat",
        "--lambda-consist",
        dest="lambda_ctxhat",
        type=float,
        default=2.0,
        help="Aux weight on ||ctx_hat - DINGO(clean)||^2 for glitch twins",
    )
    p.add_argument("--lambda-teacher", type=float, default=2.0)
    p.add_argument("--lambda-gate-bce", type=float, default=1.0)
    p.add_argument("--lambda-distill", type=float, default=2.0)
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--severity-start", type=float, default=3.0)
    p.add_argument("--severity-end", type=float, default=12.0)
    p.add_argument("--nll-start-epoch", type=int, default=1)
    p.add_argument("--distill-start-epoch", type=int, default=2)
    p.add_argument("--n-distill-samples", type=int, default=4)
    p.add_argument(
        "--hard-eval-frac",
        type=float,
        default=0.25,
        help="Fraction of train glitches locked to GW170817-style H1 SG+Welch",
    )
    p.add_argument(
        "--train-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train NSF (noise-aware student); --no-train-flow = corrector-only",
    )
    p.add_argument(
        "--train-base-embedding",
        action="store_true",
        help="Also fine-tune frozen DINGO RB embedding (off by default)",
    )
    p.add_argument(
        "--flow-lr-mult",
        type=float,
        default=0.1,
        help="NSF LR multiplier relative to --lr (keep small to preserve clean)",
    )
    p.add_argument(
        "--flow-start-epoch",
        type=int,
        default=3,
        help="Unfreeze NSF only after this epoch (corrector warms up first)",
    )
    p.add_argument(
        "--dl-eval-every",
        type=int,
        default=2,
        help="Every N epochs run GW170817 glitch d_L mid-eval (0=disable)",
    )
    p.add_argument("--dl-eval-samples", type=int, default=256)
    p.add_argument("--dl-snr-amp-scale", type=float, default=8.0)
    p.add_argument(
        "--clean-kill-gap",
        type=float,
        default=5.0,
        help="Abort if clean student-teacher NLL gap exceeds this",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
