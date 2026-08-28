#!/usr/bin/env python3
"""Fine-tune Spectrogram2DNet residual on top of frozen DINGO-BNS NSF.

Keeps the pretrained DINGO RB embedding + NSF frozen and trains only a
zero-init STFT residual (3-ch mag/coherence, signal+noise TD). At init the
model matches baseline PE; training can add glitch robustness without
destroying ``d_L``.

Fresh train::

    conda activate adapt_env
    export PYTHONPATH=DINGO-BNS/dingo:src
    export KMP_DUPLICATE_LIB_OK=TRUE

    caffeinate -dims python scripts/train_bns_spectrogram.py \\
      --enable-glitch-augmentation \\
      --encoder-type resnet_deep \\
      --n-fft 2048 \\
      --lr 1e-5 \\
      --epochs 50 \\
      --batch-size 8 \\
      --steps-per-epoch 200
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BNS_CKPT = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "downloads"
    / "dingo-bns-model_GW170817.pt"
)
DEFAULT_OUTDIR = REPO_ROOT / "checkpoints"
BEST_NAME = "dingo_bns_custom_stft_best.pt"
LAST_NAME = "dingo_bns_custom_stft_last.pt"
REFINED_NAME = "dingo_bns_custom_stft_refined.pt"
DEFAULT_RESUME = DEFAULT_OUTDIR / BEST_NAME

CONTEXT_PARAMS = ["ra", "dec", "chirp_mass_proxy"]
PROXY_JITTER_FRAC = 0.01

logger = logging.getLogger("train_bns_spectrogram")


# ---------------------------------------------------------------------------
# Device / seeds
# ---------------------------------------------------------------------------


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    try:
        import bilby

        bilby.core.utils.random.seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Standardization helpers
# ---------------------------------------------------------------------------


def _std_maps(metadata: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    block = metadata["train_settings"]["data"]["standardization"]
    means = {k: float(v) for k, v in block["mean"].items()}
    stds = {k: float(v) for k, v in block["std"].items()}
    return means, stds


def standardize_vector(
    values: Dict[str, float],
    keys: Sequence[str],
    means: Dict[str, float],
    stds: Dict[str, float],
) -> np.ndarray:
    out = np.empty(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        mu = means.get(k, 0.0)
        sig = stds.get(k, 1.0) or 1.0
        out[i] = (float(values[k]) - mu) / sig
    return out


# ---------------------------------------------------------------------------
# ASD / packaging / STFT
# ---------------------------------------------------------------------------


def design_asd_on_domain(det: str, domain) -> np.ndarray:
    from pycbc.psd import AdvVirgo, aLIGOZeroDetHighPower

    base = domain.base_domain if hasattr(domain, "base_domain") else domain
    n = len(base)
    delta_f = float(base.delta_f)
    f_min = float(base.f_min)
    if det == "V1":
        psd = AdvVirgo(n, delta_f, f_min)
    else:
        psd = aLIGOZeroDetHighPower(n, delta_f, f_min)
    asd = np.sqrt(np.asarray(psd, dtype=np.float64))
    asd = base.update_data(asd, low_value=1.0)
    if hasattr(domain, "decimate"):
        asd = domain.decimate(asd)
    return np.asarray(asd, dtype=np.float64)


def package_strain_sample(
    injection_data: Dict[str, Any],
    detectors: Sequence[str],
    domain,
) -> np.ndarray:
    from dingo.gw.transforms import RepackageStrainsAndASDS, WhitenAndScaleStrain

    sample = {
        "waveform": injection_data["waveform"],
        "asds": injection_data["asds"],
    }
    sample = WhitenAndScaleStrain(domain.noise_std)(sample)
    sample = RepackageStrainsAndASDS(list(detectors), first_index=domain.min_idx)(
        sample
    )
    arr = np.asarray(sample["waveform"], dtype=np.float32)
    if arr.shape != (len(detectors), 3, len(domain)):
        raise RuntimeError(
            f"packaged strain shape {arr.shape} != "
            f"({len(detectors)}, 3, {len(domain)})"
        )
    return arr


def build_base_domain_injection(metadata: Dict[str, Any]):
    """Injection on uniform FrequencyDomain (IFFT-able) for signal+noise TD.

    The waveform-generator domain uses ``f_max=2048`` (Nyquist for 4096 Hz)
    so LALsimulation's FD array length matches the domain and does not spam
    truncation warnings. ``data_domain`` remains the BNS base domain
    (``f_max≈1535``) used for packaging / IFFT.
    """
    from dingo.gw.domains import build_domain, build_domain_from_model_metadata
    from dingo.gw.gwutils import get_extrinsic_prior_dict
    from dingo.gw.injection import Injection
    from dingo.gw.prior import build_prior_with_defaults

    base = build_domain_from_model_metadata(metadata, base=True)
    wfg_domain = build_domain(metadata["dataset_settings"]["domain"])
    if hasattr(wfg_domain, "base_domain"):
        wfg_domain = wfg_domain.base_domain
    # Align WFG grid with LAL SimInspiralFD (f_max power-of-two → no truncate warn).
    wfg_dict = dict(wfg_domain.domain_dict)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    wfg_dict["f_max"] = 0.5 * sample_rate
    wfg_domain = build_domain(wfg_dict)

    intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        metadata["train_settings"]["data"]["extrinsic_prior"]
    )
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
    injection = Injection(
        prior=prior,
        wfg_kwargs=metadata["dataset_settings"]["waveform_generator"],
        wfg_domain=wfg_domain,
        data_domain=base,
        ifo_list=metadata["train_settings"]["data"]["detectors"],
        t_ref=metadata["train_settings"]["data"]["ref_time"],
    )
    injection.asd = {
        det: design_asd_on_domain(det, base)
        for det in metadata["train_settings"]["data"]["detectors"]
    }
    return injection, base


def decimate_injection_to_domain(
    injection_data: Dict[str, Any], domain
) -> Dict[str, Any]:
    """Decimate uniform-FD waveforms/ASDs onto a MultibandedFrequencyDomain."""
    if not hasattr(domain, "decimate"):
        return injection_data
    return {
        **injection_data,
        "waveform": {
            det: domain.decimate(np.asarray(w))
            for det, w in injection_data["waveform"].items()
        },
        "asds": {
            det: domain.decimate(np.asarray(a))
            for det, a in injection_data["asds"].items()
        },
    }


def build_spectrogram_stack(
    detectors: Sequence[str],
    sample_rate: float,
    rng: np.random.Generator,
    *,
    td_signal_map: Mapping[str, np.ndarray],
    enable_glitch_augmentation: bool = False,
    glitch_prob: float = 0.15,
    stft_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return H1–L1 3-ch mag/coherence tensor and 3-IFO pre-norm log-energy.

    ``td_signal_map`` must already contain whitened signal+noise TD crops.
    Optional glitches are applied in the time domain before the complex STFT.
    """
    from adapt.stft_context import (
        build_csd_spectrogram_from_td,
        crop_td_to_analysis_window,
        inject_random_glitch,
    )

    stft_kwargs = dict(stft_kwargs or {})
    td_map: Dict[str, np.ndarray] = {}
    for det in detectors:
        if det not in td_signal_map:
            raise KeyError(f"Missing signal TD for detector {det}")
        td_map[det] = crop_td_to_analysis_window(
            np.asarray(td_signal_map[det], dtype=np.float64), sample_rate
        )

    if enable_glitch_augmentation and float(glitch_prob) > 0.0:
        if float(rng.random()) < float(glitch_prob):
            det = str(rng.choice(list(td_map.keys())))
            td_map[det] = inject_random_glitch(td_map[det], sample_rate, rng)

    return build_csd_spectrogram_from_td(
        td_map,
        sample_rate,
        energy_detectors=tuple(detectors),
        **stft_kwargs,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BNSSpectrogramInjectionDataset(Dataset):
    """On-the-fly DINGO BNS injections with signal+noise STFT contexts.

    FD packaging uses MFD-decimated whitened strain (NSF / strain encoder).
    STFT contexts are IFFT'd from the same base-domain signal+noise waveforms
    (ASD-whitened) so spectrogram amplitude / log-energy track ``d_L``.
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        *,
        length: int,
        seed: int = 0,
        use_hybrid_background: bool = True,
        enable_glitch_augmentation: bool = False,
        glitch_prob: float = 0.15,
        stft_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from dingo.gw.domains import build_domain_from_model_metadata

        self.metadata = metadata
        self.length = int(length)
        self.base_seed = int(seed)
        self.detectors = list(metadata["train_settings"]["data"]["detectors"])
        self.inference_params = list(
            metadata["train_settings"]["data"]["inference_parameters"]
        )
        self.context_params = list(CONTEXT_PARAMS)
        self.means, self.stds = _std_maps(metadata)
        # MFD for NSF packaging; uniform base FD for IFFT → STFT.
        self.domain = build_domain_from_model_metadata(metadata)
        self.injection, self.base_domain = build_base_domain_injection(metadata)
        self.sample_rate = float(
            metadata["train_settings"]["data"]["window"]["f_s"]
        )
        self.enable_glitch_augmentation = bool(enable_glitch_augmentation)
        self.glitch_prob = float(glitch_prob)
        self.stft_kwargs = dict(stft_kwargs or {})
        # Hybrid noise pool is unused: STFT carries injection signal+noise.
        self.use_hybrid_background = bool(use_hybrid_background)
        if self.use_hybrid_background:
            logger.info(
                "HybridBackgroundPool skipped: STFT uses injection signal+noise TD"
            )

    def __len__(self) -> int:
        return self.length

    def _sample_theta(self, rng: np.random.Generator) -> Dict[str, float]:
        # Bilby prior sampling is not numpy-Generator aware; reseed for variety.
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            import bilby

            bilby.core.utils.random.seed(seed)
        except Exception:
            pass
        theta = self.injection.prior.sample()
        # Ensure float scalars
        return {k: float(v) for k, v in theta.items()}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from adapt.stft_context import injection_waveforms_to_td_map

        rng = np.random.default_rng(self.base_seed + int(index) * 1009 + 17)
        theta = self._sample_theta(rng)

        mc = float(theta["chirp_mass"])
        jitter = PROXY_JITTER_FRAC * mc * float(rng.normal())
        mc_proxy = mc + jitter
        delta_mc = mc - mc_proxy
        theta["chirp_mass_proxy"] = mc_proxy
        theta["delta_chirp_mass"] = delta_mc

        inj = self.injection.injection(theta)
        inj_mfd = decimate_injection_to_domain(inj, self.domain)
        strain = package_strain_sample(inj_mfd, self.detectors, self.domain)
        td_signal_map = injection_waveforms_to_td_map(
            inj,
            self.detectors,
            sample_rate=self.sample_rate,
            duration=float(self.base_domain.duration),
            noise_std=float(self.base_domain.noise_std),
            whiten=True,
        )
        spectrogram, log_energy = build_spectrogram_stack(
            self.detectors,
            self.sample_rate,
            rng,
            td_signal_map=td_signal_map,
            enable_glitch_augmentation=self.enable_glitch_augmentation,
            glitch_prob=self.glitch_prob,
            stft_kwargs=self.stft_kwargs,
        )

        y = standardize_vector(
            theta, self.inference_params, self.means, self.stds
        )
        context_z = standardize_vector(
            theta, self.context_params, self.means, self.stds
        )

        strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
        spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
        log_energy = np.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        context_z = np.nan_to_num(context_z, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "strain": torch.from_numpy(strain),
            "spectrogram": torch.from_numpy(spectrogram),
            "log_energy": torch.from_numpy(log_energy.astype(np.float32)),
            "y": torch.from_numpy(y),
            "context_z": torch.from_numpy(context_z),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()
    }


# ---------------------------------------------------------------------------
# Model load / surgery
# ---------------------------------------------------------------------------


def load_bns_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_flow_wrapper(ckpt: Dict[str, Any]):
    from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net

    mk = copy.deepcopy(ckpt["model_kwargs"])
    wrapper = create_nsf_with_rb_projection_embedding_net(
        mk["nsf_kwargs"], mk["embedding_net_kwargs"]
    )
    incompatible = wrapper.load_state_dict(ckpt["model_state_dict"], strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(
        incompatible, "unexpected_keys", None
    ):
        raise RuntimeError(
            f"state_dict mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return wrapper


def attach_spectrogram_embedding(
    wrapper: nn.Module,
    *,
    encoder_type: str = "resnet_deep",
    encoder_channels: Optional[Sequence[int]] = None,
    n_time: int = 5,
    n_spec_freq: int = 128,
    in_channels: int = 3,
    residual_dingo: bool = True,
) -> nn.Module:
    """Attach STFT context.

    Default ``residual_dingo=True`` keeps the pretrained DINGO RB embedding and
    adds a zero-init STFT-only residual (preserves baseline PE at init). Setting
    ``residual_dingo=False`` replaces the embedding entirely (legacy).
    """
    from adapt.models import (
        ResidualSpectrogramBNSEmbedding,
        Spectrogram2DNet,
        SpectrogramBNSEmbedding,
    )
    from adapt.models.custom_embedding import SpectrogramResidualHead

    if residual_dingo:
        spect_net = SpectrogramResidualHead(
            n_time=int(n_time),
            n_spec_freq=int(n_spec_freq),
            encoder_type=encoder_type,
            encoder_channels=encoder_channels,
            in_channels=int(in_channels),
        )
        base_emb = wrapper.embedding_net
        wrapper.embedding_net = ResidualSpectrogramBNSEmbedding(
            base_emb, spect_net=spect_net
        )
        logger.info(
            "Residual DINGO embedding: frozen RB base + STFT-only residual "
            "(init_scale=0.05, max_delta=0.5)"
        )
    else:
        spect_net = Spectrogram2DNet(
            n_time=int(n_time),
            n_spec_freq=int(n_spec_freq),
            encoder_type=encoder_type,
            encoder_channels=encoder_channels,
            in_channels=int(in_channels),
        )
        wrapper.embedding_net = SpectrogramBNSEmbedding(spect_net)
        logger.info("Legacy full-replacement SpectrogramBNSEmbedding")
    return wrapper


def set_phase_trainable(wrapper: nn.Module, phase: int) -> None:
    """Phase 1: freeze flow. Phases 2–3: train embedding + flow.

    Residual-DINGO mode always freezes the RB base and keeps the NSF frozen
    (calibrated PE); only the STFT residual + gate train.
    """
    emb = wrapper.embedding_net
    residual = hasattr(emb, "base_embedding") and hasattr(
        emb, "trainable_residual_parameters"
    )
    if residual:
        for p in wrapper.flow.parameters():
            p.requires_grad = False
        for p in emb.base_embedding.parameters():
            p.requires_grad = False
        for p in emb.parameters():
            p.requires_grad = False
        for p in emb.trainable_residual_parameters():
            p.requires_grad = True
        return

    for p in wrapper.flow.parameters():
        p.requires_grad = phase >= 2
    for p in wrapper.embedding_net.parameters():
        p.requires_grad = True


def apply_train_modes(wrapper: nn.Module, phase: int) -> None:
    """Set train/eval modes for the current phase.

    Phase 1 freezes NSF BatchNorm running stats via ``flow.eval()``. Without
    this, a single large embedding update can poison frozen-flow BN buffers and
    permanently NaN all subsequent losses. Residual mode always keeps flow.eval().
    """
    wrapper.train()
    emb = wrapper.embedding_net
    residual = hasattr(emb, "base_embedding")
    if residual or int(phase) < 2:
        wrapper.flow.eval()
        if residual and hasattr(emb, "base_embedding"):
            emb.base_embedding.eval()


def _state_dict_has_nonfinite(state: Dict[str, Any]) -> List[str]:
    bad: List[str] = []
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
            if not torch.isfinite(tensor).all():
                bad.append(name)
    return bad


def _module_has_nonfinite(module: nn.Module) -> List[str]:
    bad: List[str] = []
    for name, tensor in module.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            bad.append(name)
    return bad


def default_learning_rates(phase: int, *, fine_tune: bool = False) -> Tuple[float, Optional[float]]:
    """Return ``(emb_lr, flow_lr)`` defaults for a training phase.

    ``flow_lr`` is ``None`` for phase 1 (flow frozen).
    """
    if fine_tune:
        return 1e-5, 5e-6
    if int(phase) == 1:
        return 1e-4, None
    if int(phase) == 3:
        return 2e-5, 1e-5
    # Phase 2
    return 1e-4, 1e-5


def resolve_learning_rates(
    phase: int,
    *,
    fine_tune: bool = False,
    lr: Optional[float] = None,
    flow_lr: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    """Apply CLI overrides on top of phase defaults.

    ``--lr`` overrides embedding LR. ``--flow-lr`` overrides flow LR when the
    flow is trained; if unset, the phase default flow LR is kept.
    """
    emb_default, flow_default = default_learning_rates(phase, fine_tune=fine_tune)
    emb = float(lr) if lr is not None else float(emb_default)
    if flow_default is None:
        return emb, None
    flow = float(flow_lr) if flow_lr is not None else float(flow_default)
    return emb, flow


def build_optimizer(
    wrapper: nn.Module,
    phase: int,
    *,
    fine_tune: bool = False,
    lr: Optional[float] = None,
    flow_lr: Optional[float] = None,
) -> AdamW:
    emb_params = [p for p in wrapper.embedding_net.parameters() if p.requires_grad]
    flow_params = [p for p in wrapper.flow.parameters() if p.requires_grad]
    emb_lr, resolved_flow_lr = resolve_learning_rates(
        phase, fine_tune=fine_tune, lr=lr, flow_lr=flow_lr
    )
    if not emb_params and not flow_params:
        raise RuntimeError("no trainable parameters")
    if flow_params and (fine_tune or int(phase) >= 2):
        if resolved_flow_lr is None:
            raise RuntimeError("flow is trainable but flow_lr is None")
        groups = [
            {"params": emb_params, "lr": emb_lr},
            {"params": flow_params, "lr": resolved_flow_lr},
        ]
        return AdamW(groups, weight_decay=1e-4)
    # Embedding / residual only (phase 1, or residual-DINGO with frozen NSF).
    return AdamW(emb_params, lr=emb_lr, weight_decay=1e-4)


def load_matching_state_dict(
    module: nn.Module,
    state_dict: Dict[str, Any],
    *,
    label: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Load overlapping equal-shape tensors with ``strict=False``.

    PyTorch still raises on shape mismatches even when ``strict=False``, so we
    drop unequal-shape keys first (e.g. widened ``out.0`` / new ``energy_proj``
    when resuming a pre-energy ``best.pt``).
    """
    current = module.state_dict()
    filtered: Dict[str, Any] = {}
    skipped_shape: List[str] = []
    for key, tensor in state_dict.items():
        if key not in current:
            continue
        if (
            torch.is_tensor(tensor)
            and torch.is_tensor(current[key])
            and current[key].shape != tensor.shape
        ):
            skipped_shape.append(key)
            continue
        filtered[key] = tensor
    incompatible = module.load_state_dict(filtered, strict=False)
    missing = list(getattr(incompatible, "missing_keys", incompatible[0]))
    unexpected = list(getattr(incompatible, "unexpected_keys", incompatible[1]))
    logger.info(
        "Partial load %s (strict=False): loaded=%d skipped_shape=%d "
        "missing=%d unexpected=%d",
        label,
        len(filtered),
        len(skipped_shape),
        len(missing),
        len(unexpected),
    )
    if skipped_shape:
        logger.info(
            "Partial load %s skipped shape-mismatched keys (e.g. %s)",
            label,
            skipped_shape[:5],
        )
    if missing:
        logger.info(
            "Partial load %s left randomly-initialized keys (e.g. %s)",
            label,
            missing[:5],
        )
    return missing, unexpected, skipped_shape


def load_stft_weights(wrapper: nn.Module, resume_path: Path) -> Dict[str, Any]:
    """Load ADAPT STFT checkpoint into a surgeried wrapper (energy-head safe).

    Flow weights load with ``strict=True``. Embedding loads with shape-matched
    ``strict=False`` so pre-energy checkpoints (e.g.
    ``dingo_bns_custom_stft_best.pt``) resume into the log-energy architecture
    without shape-mismatch errors; new/changed layers stay at init.
    """
    ckpt = load_bns_checkpoint(resume_path)
    if "flow_state_dict" not in ckpt or "embedding_state_dict" not in ckpt:
        raise KeyError(
            f"{resume_path} missing flow_state_dict / embedding_state_dict "
            "(expected an ADAPT STFT checkpoint, not the raw BNS model)"
        )
    bad_emb = _state_dict_has_nonfinite(ckpt["embedding_state_dict"])
    bad_flow = _state_dict_has_nonfinite(ckpt["flow_state_dict"])
    if bad_emb or bad_flow:
        raise RuntimeError(
            f"Refusing to resume {resume_path}: checkpoint contains non-finite "
            f"tensors (embedding={len(bad_emb)}, flow={len(bad_flow)}). "
            "Start a fresh run without --resume / --fine-tune so weights reload "
            "from the baseline BNS NSF checkpoint."
        )
    # NSF body is architecture-stable across the energy-head upgrade.
    wrapper.flow.load_state_dict(ckpt["flow_state_dict"], strict=True)
    # Equivalent to load_state_dict(..., strict=False) but also skips shape
    # mismatches (required for energy_proj / widened out.0).
    load_matching_state_dict(
        wrapper.embedding_net,
        ckpt["embedding_state_dict"],
        label="embedding",
    )
    logger.info(
        "Resumed STFT weights from %s (epoch=%s best_val_nll=%s) "
        "with embedding strict=False / shape-matched load",
        resume_path,
        ckpt.get("epoch"),
        ckpt.get("best_val_nll"),
    )
    return ckpt


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Halt training when ``val_nll`` stops improving (lower is better)."""

    def __init__(self, patience: int = 50, min_delta: float = 0.01):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val_nll: float) -> bool:
        """Update state with the latest ``val_nll``.

        Returns
        -------
        improved : bool
            True if ``val_nll`` improved by more than ``min_delta``.
        """
        if not math.isfinite(val_nll):
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
            return False

        improved = val_nll < (self.best_score - self.min_delta)
        if improved:
            self.best_score = float(val_nll)
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False

    def reset(self) -> None:
        """Reset patience counter (keeps best score)."""
        self.bad_epochs = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def batch_nll(wrapper: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    strain = batch["strain"].to(device=device, dtype=torch.float32)
    spectrogram = batch["spectrogram"].to(device=device, dtype=torch.float32)
    log_energy = batch["log_energy"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    context_z = batch["context_z"].to(device=device, dtype=torch.float32)
    # Guard against rare STFT/glitch NaNs propagating into the NSF.
    strain = torch.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
    spectrogram = torch.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
    log_energy = torch.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)
    # NSF *x order: strain, spectrogram, log_energy, context_z
    log_prob = wrapper.log_prob(y, strain, spectrogram, log_energy, context_z)
    # Clamp extreme log-probs from OOD contexts during early embedding training.
    log_prob = torch.clamp(log_prob, min=-1.0e4, max=1.0e4)
    return -log_prob.mean()


@torch.no_grad()
def evaluate_nll(
    wrapper: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    wrapper.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss = batch_nll(wrapper, batch, device)
        if not torch.isfinite(loss):
            continue
        total += float(loss.item()) * batch["y"].shape[0]
        n += batch["y"].shape[0]
    wrapper.train()
    if n == 0:
        return float("nan")
    return total / n


def save_checkpoint(
    path: Path,
    *,
    wrapper: nn.Module,
    epoch: int,
    phase: int,
    best_val_nll: float,
    metadata: Dict[str, Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    adapt_config: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nsf_kwargs = None
    if model_kwargs is not None:
        nsf_kwargs = model_kwargs.get("nsf_kwargs")
    if nsf_kwargs is None:
        nsf_kwargs = metadata["train_settings"]["model"].get("nsf_kwargs")
    meta_out: Dict[str, Any] = {
        "inference_parameters": metadata["train_settings"]["data"][
            "inference_parameters"
        ],
        "context_parameters": CONTEXT_PARAMS,
        "standardization": metadata["train_settings"]["data"]["standardization"],
        "detectors": metadata["train_settings"]["data"]["detectors"],
        "nsf_kwargs": nsf_kwargs,
    }
    if adapt_config:
        meta_out["adapt_config"] = dict(adapt_config)
        # Flatten commonly needed keys for eval loaders.
        for k in (
            "encoder_type",
            "encoder_channels",
            "n_time",
            "n_freq",
            "n_fft",
            "win_length",
            "hop_length",
            "energy_conditioning",
            "csd_channels",
            "spectrogram_layout",
            "signal_in_stft",
            "residual_dingo",
            "lr",
            "flow_lr",
        ):
            if k in adapt_config:
                meta_out[k] = adapt_config[k]
    bad = _module_has_nonfinite(wrapper)
    if bad:
        logger.error(
            "Refusing to write %s: model has %d non-finite tensors (e.g. %s)",
            path,
            len(bad),
            bad[0],
        )
        return
    if math.isnan(float(best_val_nll)):
        logger.warning("Saving %s with NaN best_val_nll", path)
    payload = {
        "epoch": epoch,
        "phase": phase,
        "best_val_nll": best_val_nll,
        "spectrogram_net_state_dict": wrapper.embedding_net.spect_net.state_dict(),
        "embedding_state_dict": wrapper.embedding_net.state_dict(),
        "flow_state_dict": wrapper.flow.state_dict(),
        "metadata": meta_out,
    }
    torch.save(payload, path)
    logger.info("Wrote checkpoint %s", path)


def run_training(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    device = select_device()
    logger.info("Device: %s", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"BNS checkpoint not found: {ckpt_path}")

    raw = load_bns_checkpoint(ckpt_path)
    metadata = raw["metadata"]
    model_kwargs = raw.get("model_kwargs") or {}

    logger.info("Loading NSF + RB embedding from %s", ckpt_path)
    wrapper = build_flow_wrapper(raw)

    encoder_type = str(args.encoder_type)
    encoder_channels = list(args.encoder_channels)
    stft_kwargs = {
        "n_time": int(args.n_time),
        "n_freq": int(args.n_freq),
        "n_fft": int(args.n_fft) if args.n_fft is not None else None,
        "win_length": int(args.win_length) if args.win_length is not None else None,
        "hop_length": int(args.hop_length) if args.hop_length is not None else None,
    }
    # Drop Nones so helpers fall back to their defaults when unset.
    stft_kwargs = {k: v for k, v in stft_kwargs.items() if v is not None}

    # If resuming an older checkpoint that recorded encoder/STFT config, prefer it
    # unless the user explicitly changed architecture flags (we always use CLI here).
    # Resolve LRs for logging / metadata (phase-1 defaults; per-phase rebuild uses same CLI).
    log_phase = 3 if args.fine_tune else (int(args.phase) if args.phase is not None else 1)
    emb_lr0, flow_lr0 = resolve_learning_rates(
        log_phase,
        fine_tune=bool(args.fine_tune),
        lr=args.lr,
        flow_lr=args.flow_lr,
    )

    adapt_config: Dict[str, Any] = {
        "encoder_type": encoder_type,
        "encoder_channels": encoder_channels,
        "n_time": int(args.n_time),
        "n_freq": int(args.n_freq),
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "enable_glitch_augmentation": bool(args.enable_glitch_augmentation),
        "glitch_prob": float(args.glitch_prob),
        "energy_conditioning": True,
        "csd_channels": 3,
        "spectrogram_layout": "hl_coh_3ch",
        "signal_in_stft": True,
        "residual_dingo": True,
        "batch_size": int(args.batch_size),
        "lr": float(emb_lr0),
        "flow_lr": float(flow_lr0) if flow_lr0 is not None else None,
        "lr_cli": args.lr,
        "flow_lr_cli": args.flow_lr,
        "fine_tune": bool(args.fine_tune),
        "phase": args.phase,
    }
    logger.info(
        "Hyperparams | encoder=%s channels=%s | STFT n_time=%s n_freq=%s n_fft=%s "
        "win=%s hop=%s | layout=hl_coh_3ch csd_channels=3 signal_in_stft=True "
        "residual_dingo=True | glitch_aug=%s prob=%.3f | energy_conditioning=True | "
        "batch_size=%d | emb_lr=%.2e flow_lr=%s | fine_tune=%s phase=%s",
        encoder_type,
        encoder_channels,
        args.n_time,
        args.n_freq,
        args.n_fft,
        args.win_length,
        args.hop_length,
        bool(args.enable_glitch_augmentation),
        float(args.glitch_prob),
        int(args.batch_size),
        emb_lr0,
        f"{flow_lr0:.2e}" if flow_lr0 is not None else "n/a",
        bool(args.fine_tune),
        args.phase,
    )

    wrapper = attach_spectrogram_embedding(
        wrapper,
        encoder_type=encoder_type,
        encoder_channels=encoder_channels,
        n_time=int(args.n_time),
        n_spec_freq=int(args.n_freq),
        in_channels=3,
        residual_dingo=True,
    )
    # Residual path is sensitive to large steps; default emb LR → 1e-5 if unset.
    if hasattr(wrapper.embedding_net, "base_embedding") and args.lr is None:
        args.lr = 1e-5
        emb_lr0, flow_lr0 = resolve_learning_rates(
            1, fine_tune=False, lr=args.lr, flow_lr=args.flow_lr
        )
        adapt_config["lr"] = float(emb_lr0)
        logger.info("Residual-DINGO default emb_lr overridden to %.2e", emb_lr0)


    resume_meta: Dict[str, Any] = {}
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_meta = load_stft_weights(wrapper, resume_path)
        if resume_meta.get("metadata"):
            # Prefer standardization from the STFT ckpt when present.
            rmeta = resume_meta["metadata"]
            if "standardization" in rmeta:
                metadata = copy.deepcopy(metadata)
                metadata["train_settings"]["data"]["standardization"] = rmeta[
                    "standardization"
                ]

    # MPS does not support float64 parameters/buffers from the BNS checkpoint.
    wrapper = wrapper.float()
    wrapper.to(device)

    train_len = int(args.steps_per_epoch) * int(args.batch_size)
    ds_common = dict(
        use_hybrid_background=not args.no_hybrid_background,
        enable_glitch_augmentation=bool(args.enable_glitch_augmentation),
        glitch_prob=float(args.glitch_prob),
        stft_kwargs=stft_kwargs,
    )
    train_ds = BNSSpectrogramInjectionDataset(
        metadata,
        length=train_len,
        seed=args.seed,
        **ds_common,
    )
    val_ds = BNSSpectrogramInjectionDataset(
        metadata,
        length=int(args.val_size),
        seed=args.seed + 10_000,
        # Keep val clean for stable early-stopping unless glitch aug enabled.
        enable_glitch_augmentation=False,
        glitch_prob=0.0,
        stft_kwargs=stft_kwargs,
        use_hybrid_background=not args.no_hybrid_background,
    )

    num_workers = int(args.num_workers)
    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase_arg = args.phase
    total_epochs = int(args.epochs)
    if phase_arg == 3:
        schedule = [(3, total_epochs, 1)]
        best_path = outdir / REFINED_NAME
        last_path = outdir / "dingo_bns_custom_stft_refined_last.pt"
    elif phase_arg == 1:
        schedule = [(1, total_epochs, 1)]
        best_path = outdir / BEST_NAME
        last_path = outdir / LAST_NAME
    elif phase_arg == 2:
        schedule = [(2, total_epochs, 1)]
        best_path = outdir / BEST_NAME
        last_path = outdir / LAST_NAME
    else:
        # Default: phases 1 then 2 spanning --epochs (10 + remainder).
        phase1_epochs = min(10, total_epochs)
        phase2_epochs = max(0, total_epochs - phase1_epochs)
        schedule = [
            (1, phase1_epochs, 1),
            (2, phase2_epochs, phase1_epochs + 1),
        ]
        best_path = outdir / BEST_NAME
        last_path = outdir / LAST_NAME

    # Residual-DINGO keeps the calibrated NSF frozen; train residual for all epochs.
    if hasattr(wrapper.embedding_net, "base_embedding") and not args.fine_tune:
        if phase_arg is None or int(phase_arg) in (1, 2):
            schedule = [(1, total_epochs, 1)]
            best_path = outdir / BEST_NAME
            last_path = outdir / LAST_NAME
            logger.info(
                "Residual-DINGO schedule: %d epochs residual-only "
                "(NSF + RB base frozen; STFT gate/residual trainable)",
                total_epochs,
            )

    if args.fine_tune:
        if args.resume is None:
            raise ValueError("--fine-tune requires --resume pointing at an STFT checkpoint")
        ft_emb, ft_flow = resolve_learning_rates(
            3, fine_tune=True, lr=args.lr, flow_lr=args.flow_lr
        )
        logger.info(
            "Fine-tune mode: emb_lr=%.2e flow_lr=%.2e cosine eta_min=1e-7 "
            "(weights loaded from resume; optimizer state not restored)",
            ft_emb,
            ft_flow if ft_flow is not None else 0.0,
        )

    best_val = float("inf")
    if resume_meta and math.isfinite(float(resume_meta.get("best_val_nll", float("inf")))):
        # Do not carry prior best across phase-3; start fresh ranking for refined.pt
        if phase_arg != 3 and not args.fine_tune:
            best_val = float(resume_meta["best_val_nll"])
        elif args.fine_tune:
            # Preserve best score for early-stopping continuity when fine-tuning.
            best_val = float(resume_meta["best_val_nll"])
    global_epoch = 0
    early_stopper = EarlyStopping(
        patience=int(args.patience),
        min_delta=float(args.min_delta),
    )
    if math.isfinite(best_val):
        early_stopper.best_score = best_val
    logger.info(
        "EarlyStopping: patience=%d min_delta=%.4g",
        early_stopper.patience,
        early_stopper.min_delta,
    )
    stopped_early = False

    for phase, n_epochs, phase_start in schedule:
        if n_epochs <= 0:
            continue
        # Fine-tune always trains embedding + flow jointly.
        train_phase = max(phase, 2) if args.fine_tune else phase
        logger.info(
            "===== Phase %d%s | epochs %d–%d =====",
            train_phase,
            " (fine-tune)" if args.fine_tune else "",
            phase_start,
            phase_start + n_epochs - 1,
        )
        set_phase_trainable(wrapper, train_phase)
        emb_lr, flow_lr = resolve_learning_rates(
            train_phase,
            fine_tune=bool(args.fine_tune),
            lr=args.lr,
            flow_lr=args.flow_lr,
        )
        adapt_config["lr"] = float(emb_lr)
        adapt_config["flow_lr"] = float(flow_lr) if flow_lr is not None else None
        logger.info(
            "Phase %d optimizer LRs: emb=%.2e flow=%s",
            train_phase,
            emb_lr,
            f"{flow_lr:.2e}" if flow_lr is not None else "n/a",
        )
        optimizer = build_optimizer(
            wrapper,
            train_phase,
            fine_tune=bool(args.fine_tune),
            lr=args.lr,
            flow_lr=args.flow_lr,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(n_epochs, 1),
            eta_min=1e-7 if args.fine_tune else 0.0,
        )
        # Fresh patience budget at each phase boundary (keeps best_score).
        early_stopper.reset()

        for local_epoch in range(1, n_epochs + 1):
            global_epoch = phase_start + local_epoch - 1
            apply_train_modes(wrapper, train_phase)
            running = 0.0
            n_seen = 0
            n_skipped = 0
            consecutive_nonfinite = 0
            for step, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = batch_nll(wrapper, batch, device)
                if not torch.isfinite(loss):
                    n_skipped += 1
                    consecutive_nonfinite += 1
                    logger.warning(
                        "Skipping non-finite loss at epoch %d step %d",
                        global_epoch,
                        step,
                    )
                    if consecutive_nonfinite >= 20:
                        raise RuntimeError(
                            f"Aborting: {consecutive_nonfinite} consecutive non-finite "
                            f"losses at epoch {global_epoch} step {step}. "
                            "This usually means poisoned weights or unstable Phase-1 "
                            "BN updates. Restart without --resume from the baseline "
                            "BNS checkpoint (do not load prior STFT best/last)."
                        )
                    continue
                consecutive_nonfinite = 0
                loss.backward()
                trainable = [p for p in wrapper.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                # Drop the step if gradients exploded.
                grads_ok = True
                for p in trainable:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    n_skipped += 1
                    logger.warning(
                        "Skipping non-finite gradients at epoch %d step %d",
                        global_epoch,
                        step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                bs = batch["y"].shape[0]
                running += float(loss.item()) * bs
                n_seen += bs
                if step % max(1, args.log_every) == 0:
                    logger.info(
                        "Phase %d epoch %d/%d step %d/%d  train_nll=%.4f",
                        train_phase,
                        global_epoch,
                        total_epochs,
                        step,
                        args.steps_per_epoch,
                        float(loss.item()),
                    )

            if n_seen > 0:
                scheduler.step()
            else:
                logger.warning(
                    "Skipping LR scheduler step at epoch %d (no optimizer updates)",
                    global_epoch,
                )
            if n_seen == 0:
                train_nll = float("nan")
                logger.error(
                    "Epoch %d had zero finite training steps (%d skipped). "
                    "Not updating best checkpoint.",
                    global_epoch,
                    n_skipped,
                )
            else:
                train_nll = running / n_seen
            val_nll = evaluate_nll(wrapper, val_loader, device)
            lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
            logger.info(
                "Phase %d epoch %d done | train_nll=%.4f val_nll=%.4f lr=%s skipped=%d",
                train_phase,
                global_epoch,
                train_nll,
                val_nll,
                ",".join(lrs),
                n_skipped,
            )

            if _module_has_nonfinite(wrapper):
                raise RuntimeError(
                    f"Model weights became non-finite during epoch {global_epoch}. "
                    "Aborting before writing checkpoints. Restart without --resume."
                )

            # Always persist latest weights for resume (refuses if non-finite).
            save_checkpoint(
                last_path,
                wrapper=wrapper,
                epoch=global_epoch,
                phase=train_phase,
                best_val_nll=best_val,
                metadata=metadata,
                model_kwargs=model_kwargs,
                adapt_config=adapt_config,
            )

            # Prefer finite val_nll for early stopping; fall back to train_nll.
            metric = val_nll if math.isfinite(val_nll) else train_nll
            if not math.isfinite(metric):
                early_stopper.step(float("nan"))
                if early_stopper.should_stop:
                    logger.info(
                        "Early stopping triggered at epoch %d after non-finite metrics.",
                        global_epoch,
                    )
                    stopped_early = True
                    break
                continue

            improved = early_stopper.step(metric)
            if improved:
                best_val = float(early_stopper.best_score)
                save_checkpoint(
                    best_path,
                    wrapper=wrapper,
                    epoch=global_epoch,
                    phase=train_phase,
                    best_val_nll=best_val,
                    metadata=metadata,
                    model_kwargs=model_kwargs,
                    adapt_config=adapt_config,
                )
                logger.info("New best val_nll=%.4f → %s", best_val, best_path)

            if early_stopper.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d. Best val_nll=%.4f",
                    global_epoch,
                    best_val if math.isfinite(best_val) else early_stopper.best_score,
                )
                # Final state already on last_path; ensure best exists.
                if not best_path.is_file():
                    save_checkpoint(
                        best_path,
                        wrapper=wrapper,
                        epoch=global_epoch,
                        phase=train_phase,
                        best_val_nll=best_val,
                        metadata=metadata,
                        model_kwargs=model_kwargs,
                        adapt_config=adapt_config,
                    )
                stopped_early = True
                break

        if stopped_early:
            break

    if not best_path.is_file() and last_path.is_file():
        save_checkpoint(
            best_path,
            wrapper=wrapper,
            epoch=global_epoch,
            phase=schedule[-1][0] if schedule else 1,
            best_val_nll=best_val,
            metadata=metadata,
            model_kwargs=model_kwargs,
            adapt_config=adapt_config,
        )

    logger.info(
        "Training complete%s. Best val NLL=%.4f  best=%s",
        " (early stopped)" if stopped_early else "",
        best_val,
        best_path,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Spectrogram2DNet + BNS NSF")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_BNS_CKPT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--phase",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Run a single phase (3 = refinement). Default: phases 1→2.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="ADAPT STFT checkpoint to load after architecture surgery "
        f"(default for --phase 3: {DEFAULT_RESUME})",
    )
    p.add_argument(
        "--fine-tune",
        action="store_true",
        help="Low-LR continuation from --resume: emb 1e-5, flow 5e-6, "
        "cosine eta_min=1e-7; defaults patience=100, min-delta=1e-4",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 100 with --fine-tune, else 50)",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Early stopping min improvement (default: 1e-4 with --fine-tune, else 0.01)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override embedding learning rate (default: phase-dependent, "
        "e.g. 1e-4 phase1/2, 1e-5 fine-tune)",
    )
    p.add_argument(
        "--flow-lr",
        type=float,
        default=None,
        help="Override flow learning rate when flow is trained "
        "(default: phase-dependent; ignored in phase 1)",
    )
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--no-hybrid-background",
        action="store_true",
        help="Deprecated no-op: STFT always uses injection signal+noise TD (not hybrid noise)",
    )
    p.add_argument(
        "--enable-glitch-augmentation",
        action="store_true",
        help="Randomly inject sine-Gaussian glitches into TD STFT contexts",
    )
    p.add_argument(
        "--glitch-prob",
        type=float,
        default=0.15,
        help="Probability of glitch injection per sample when augmentation enabled",
    )
    p.add_argument(
        "--encoder-type",
        type=str,
        choices=("cnn_base", "resnet_deep"),
        default="resnet_deep",
        help="2-D spectrogram encoder architecture",
    )
    p.add_argument(
        "--encoder-channels",
        type=str,
        default="64,128,256,512",
        help="Comma-separated channel widths for resnet_deep (ignored for cnn_base)",
    )
    p.add_argument("--n-fft", type=int, default=2048, help="STFT n_fft before resample")
    p.add_argument(
        "--win-length", type=int, default=1024, help="STFT window / nperseg length"
    )
    p.add_argument(
        "--hop-length", type=int, default=256, help="STFT hop length (win - noverlap)"
    )
    p.add_argument(
        "--n-time", type=int, default=5, help="Resampled STFT time bins (network input)"
    )
    p.add_argument(
        "--n-freq",
        type=int,
        default=128,
        help="Resampled STFT frequency bins (network input)",
    )
    args = p.parse_args(argv)
    # Parse encoder channels list
    try:
        args.encoder_channels = [
            int(x.strip()) for x in str(args.encoder_channels).split(",") if x.strip()
        ]
    except ValueError as exc:
        p.error(f"Invalid --encoder-channels: {args.encoder_channels!r} ({exc})")
    if not args.encoder_channels:
        p.error("--encoder-channels must list at least one integer")
    if args.fine_tune and args.resume is None:
        p.error("--fine-tune requires --resume")
    if args.phase == 3 and args.resume is None:
        args.resume = DEFAULT_RESUME
    if args.epochs is None:
        args.epochs = 100 if args.phase == 3 or args.fine_tune else 50
    if args.fine_tune:
        if args.patience is None:
            args.patience = 100
        if args.min_delta is None:
            args.min_delta = 0.0001
        # Default to joint training when phase not specified.
        if args.phase is None:
            args.phase = 3
    else:
        if args.patience is None:
            args.patience = 50
        if args.min_delta is None:
            args.min_delta = 0.01
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    # Prefer DINGO-BNS sources if present (avoids circular import in dingo-t1 editable).
    bns_dingo = REPO_ROOT / "DINGO-BNS" / "dingo"
    if bns_dingo.is_dir():
        sys.path.insert(0, str(bns_dingo))
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    run_training(args)


if __name__ == "__main__":
    main()
