#!/usr/bin/env python3
"""Head-to-head GW170817 evaluation: baseline DINGO-BNS vs custom STFT model.

Discovers the official demo strain/PSD/event assets, runs identical-input
5 000-sample inference for both models, and writes sample tensors plus three
PDF reports under ``results/``.

Usage::

    conda activate adapt_env
    export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=DINGO-BNS/dingo:src
    python scripts/evaluate_gw170817_comparison.py
"""

from __future__ import annotations

import argparse
import ast
import copy
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
)
DOWNLOADS_DIR = DEMO_ROOT / "downloads"
PIPE_DIR = DEMO_ROOT / "inference-dingo-pipe"
DEFAULT_BASELINE = DOWNLOADS_DIR / "dingo-bns-model_GW170817.pt"
DEFAULT_CUSTOM = REPO_ROOT / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt"
LEGACY_CUSTOM = REPO_ROOT / "checkpoints" / "dingo_bns_custom_stft_best.pt"
DEFAULT_OUTDIR = REPO_ROOT / "results"

CONTEXT_PARAMS = ["ra", "dec", "chirp_mass_proxy"]
COMPARE_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "ra",
    "dec",
    "theta_jn",
]
DEFAULT_FIXED_CONTEXT = {
    "chirp_mass_proxy": 1.19786,
    "ra": 3.44616,
    "dec": -0.408084,
}

logger = logging.getLogger("evaluate_gw170817")


# ---------------------------------------------------------------------------
# Device / helpers
# ---------------------------------------------------------------------------


def select_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _quantile_ci(x: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> Tuple[float, float, float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, med, hi = np.quantile(arr, [q_lo, 0.5, q_hi])
    return float(med), float(lo), float(hi)


def _has_dynamic_range(x: np.ndarray, eps: float = 1e-12) -> bool:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return False
    return float(np.ptp(arr)) > eps


def dataframe_to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {c: df[c].to_numpy() for c in df.columns}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _glob_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {label} matching {pattern!r} in {directory}")
    return matches[0]


def parse_ini_value(text: str, key: str) -> Optional[str]:
    # bilby/dingo ini: key = value  (value may be quoted or a dict literal)
    pat = re.compile(rf"^{re.escape(key)}\s*=\s*(.+)$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def parse_fixed_context(ini_text: str) -> Dict[str, float]:
    raw = parse_ini_value(ini_text, "fixed-context-parameters")
    if raw is None:
        logger.warning("No fixed-context-parameters in ini; using demo defaults")
        return dict(DEFAULT_FIXED_CONTEXT)
    raw = raw.strip()
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        raw = raw[1:-1]
    # Demo ini uses bare keys: {chirp_mass_proxy: 1.19786, ra: 3.44616, ...}
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            parsed = {}
            for part in inner.split(","):
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                parsed[k.strip().strip("'\"")] = float(v.strip())
        if not parsed:
            raise ValueError(f"Failed to parse fixed-context-parameters: {raw!r}")
    if not isinstance(parsed, dict):
        raise ValueError(f"fixed-context-parameters is not a dict: {parsed!r}")
    return {str(k): float(v) for k, v in parsed.items()}


def discover_assets(
    downloads: Path = DOWNLOADS_DIR,
    pipe_dir: Path = PIPE_DIR,
    baseline_ckpt: Optional[Path] = None,
    custom_ckpt: Optional[Path] = None,
) -> Dict[str, Any]:
    downloads = Path(downloads)
    pipe_dir = Path(pipe_dir)
    if not downloads.is_dir():
        raise FileNotFoundError(f"downloads dir missing: {downloads}")
    if not pipe_dir.is_dir():
        raise FileNotFoundError(f"inference-dingo-pipe dir missing: {pipe_dir}")

    strain = {
        "H1": _glob_one(downloads, "H-H1_*.gwf", "H1 strain"),
        "L1": _glob_one(downloads, "L-L1_*.gwf", "L1 strain"),
        "V1": _glob_one(downloads, "V-V1_*.gwf", "V1 strain"),
    }
    psd = {
        "H1": _glob_one(downloads, "GWTC1_GW170817_PSD_H1.txt", "H1 PSD"),
        "L1": _glob_one(downloads, "GWTC1_GW170817_PSD_L1.txt", "L1 PSD"),
        "V1": _glob_one(downloads, "GWTC1_GW170817_PSD_V1.txt", "V1 PSD"),
    }
    baseline = Path(baseline_ckpt) if baseline_ckpt else DEFAULT_BASELINE
    if not baseline.is_file():
        baseline = _glob_one(downloads, "dingo-bns-model_*.pt", "baseline BNS checkpoint")
    if custom_ckpt is not None:
        custom = Path(custom_ckpt)
    elif DEFAULT_CUSTOM.is_file():
        custom = DEFAULT_CUSTOM
    elif LEGACY_CUSTOM.is_file():
        custom = LEGACY_CUSTOM
    else:
        custom = DEFAULT_CUSTOM
    if not custom.is_file():
        raise FileNotFoundError(
            f"Custom STFT checkpoint not found; tried {DEFAULT_CUSTOM} and {LEGACY_CUSTOM}"
        )

    ini_path = pipe_dir / "GW170817.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Demo ini not found: {ini_path}")
    ini_text = ini_path.read_text()
    fixed = parse_fixed_context(ini_text)
    trigger_raw = parse_ini_value(ini_text, "trigger-time")
    trigger_time = float(trigger_raw) if trigger_raw else 1187008882.42

    event_candidates = sorted((pipe_dir / "outdir" / "data").glob("*_generation_event_data.hdf5"))
    event_hdf5 = event_candidates[0] if event_candidates else None

    assets = {
        "downloads": downloads,
        "pipe_dir": pipe_dir,
        "strain_gwf": strain,
        "psd": psd,
        "baseline_ckpt": baseline,
        "custom_ckpt": custom,
        "ini_path": ini_path,
        "fixed_context": fixed,
        "trigger_time": trigger_time,
        "event_hdf5": event_hdf5,
        "channels": {"H1": "H1:LOSC-STRAIN", "L1": "L1:LOSC-STRAIN", "V1": "V1:LOSC-STRAIN"},
    }
    logger.info("Discovered H1 strain: %s", strain["H1"].name)
    logger.info("Discovered event HDF5: %s", event_hdf5)
    logger.info("Fixed context: %s", fixed)
    return assets


def load_event_dataset(assets: Dict[str, Any]):
    from dingo.gw.data.event_dataset import EventDataset

    hdf5 = assets.get("event_hdf5")
    if hdf5 is not None and Path(hdf5).is_file():
        logger.info("Loading packaged event from %s", hdf5)
        return EventDataset(file_name=str(hdf5))

    logger.info("No packaged event HDF5; building EventDataset from GWF+PSD")
    return build_event_dataset_from_gwf(assets)


def build_event_dataset_from_gwf(assets: Dict[str, Any]):
    """Fallback: condition GWF+PSD into an EventDataset matching the demo settings."""
    from dingo.gw.data.event_dataset import EventDataset
    from dingo.pipe.data_generation import DataGenerationInput

    # Prefer the resolved complete ini when available (has duration/f_s filled in).
    complete = assets["pipe_dir"] / "outdir" / "GW170817_config_complete.ini"
    ini = complete if complete.is_file() else assets["ini_path"]
    logger.info("Building event via DataGenerationInput from %s", ini)

    sys_argv_backup = sys.argv[:]
    try:
        sys.argv = ["dingo_pipe_generation", str(ini)]
        from bilby_pipe.parser import create_parser
        try:
            from dingo.pipe.parser import create_parser as create_dingo_parser

            parser = create_dingo_parser()
        except Exception:
            parser = create_parser(top_level=False)
        args, _ = parser.parse_known_args([str(ini)])
        dgi = DataGenerationInput(args, [])
        dgi.save_hdf5 = lambda: None  # type: ignore[method-assign]
        if hasattr(dgi, "run"):
            dgi.run()
        elif hasattr(dgi, "create_data"):
            dgi.create_data()
        ifos = getattr(dgi, "interferometers", None)
        if ifos is None:
            raise RuntimeError("DataGenerationInput did not populate interferometers")
        detectors = list(assets["strain_gwf"].keys())
        waveform = {}
        asds = {}
        for det in detectors:
            ifo = ifos[det] if isinstance(ifos, dict) else ifos.get_ifo_by_name(det)
            waveform[det] = np.asarray(ifo.frequency_domain_strain, dtype=np.complex128)
            asds[det] = np.asarray(ifo.amplitude_spectral_density_array, dtype=np.float64)
        settings = {
            "time_event": float(assets["trigger_time"]),
            "time_buffer": 2.0,
            "detectors": detectors,
            "f_s": 4096.0,
            "T": 128.0,
            "f_min": 23.0,
            "f_max": 1535.3046875,
            "window_type": "tukey",
            "roll_off": 0.4,
        }
        return EventDataset(
            dictionary={"data": {"waveform": waveform, "asds": asds}, "settings": settings}
        )
    finally:
        sys.argv = sys_argv_backup


# ---------------------------------------------------------------------------
# Baseline sampling
# ---------------------------------------------------------------------------


def run_baseline_sampling(
    baseline_ckpt: Path,
    event,
    fixed_context: Dict[str, float],
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> pd.DataFrame:
    from dingo.core.models import PosteriorModel
    from dingo.core.samplers import FixedInitSampler
    from dingo.gw.inference.gw_samplers import GWSamplerGNPE

    # PosteriorModel historically expects a string device; MPS can be flaky with
    # float64 buffers — fall back to CPU if needed.
    device_str = str(device)
    try:
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )
        # Probe a tiny tensor move
        if device_str == "mps":
            _ = torch.zeros(1, device=device)
    except Exception as exc:
        logger.warning("Baseline on %s failed (%s); falling back to CPU", device_str, exc)
        device_str = "cpu"
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )

    init = FixedInitSampler(fixed_context, log_prob=0.0)
    sampler = GWSamplerGNPE(
        model=pm,
        init_sampler=init,
        num_iterations=1,
        fixed_context_parameters=fixed_context,
    )
    sampler.context = event.data
    sampler.event_metadata = event.settings
    sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
    df = sampler.samples.copy()
    logger.info(
        "Baseline sampling done: %d samples, columns=%s",
        len(df),
        list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Custom STFT sampling
# ---------------------------------------------------------------------------


def package_event_strain(
    event_data: Dict[str, Any],
    metadata: Dict[str, Any],
    fixed_context: Dict[str, float],
) -> np.ndarray:
    """Whiten/decimate/repackage event FD strain → ``(3, 3, n_mfd)``."""
    from dingo.gw.domains import MultibandedFrequencyDomain, build_domain_from_model_metadata
    from dingo.gw.transforms import (
        DecimateWaveformsAndASDS,
        HeterodynePhase,
        RepackageStrainsAndASDS,
        WhitenAndScaleStrain,
    )

    domain = build_domain_from_model_metadata(metadata)
    base_domain = getattr(domain, "base_domain", domain)
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    gnpe_chirp = metadata["train_settings"]["data"].get("gnpe_chirp") or {}
    order = int(gnpe_chirp.get("order", 0))

    sample = {
        "waveform": {k: np.asarray(v).copy() for k, v in event_data["waveform"].items()},
        "asds": {k: np.asarray(v).copy() for k, v in event_data["asds"].items()},
    }

    # Match baseline fixed-proxy heterodyning before MFD decimation.
    sample = HeterodynePhase(
        domain=base_domain,
        order=order,
        inverse=False,
        fixed_parameters={"chirp_mass": float(fixed_context["chirp_mass_proxy"])},
    )(sample)

    if isinstance(domain, MultibandedFrequencyDomain):
        sample = DecimateWaveformsAndASDS(domain, decimation_mode="whitened")(sample)

    sample = WhitenAndScaleStrain(domain.noise_std)(sample)
    sample = RepackageStrainsAndASDS(detectors, first_index=domain.min_idx)(sample)
    arr = np.asarray(sample["waveform"], dtype=np.float32)
    expected = (len(detectors), 3, len(domain))
    if arr.shape != expected:
        raise RuntimeError(f"packaged strain shape {arr.shape} != {expected}")
    return arr


def load_event_td_crops(
    assets: Dict[str, Any],
    *,
    sample_rate: float,
    crop_seconds: float,
) -> Dict[str, np.ndarray]:
    """Load centered TD crops matching the shared STFT analysis window."""
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    half = 0.5 * float(crop_seconds)
    # Fetch exactly the analysis window around trigger (no extra pad into STFT).
    t0 = trigger - half
    t1 = trigger + half
    out: Dict[str, np.ndarray] = {}
    for det, path in assets["strain_gwf"].items():
        channel = assets["channels"][det]
        logger.info("Loading TD %s from %s [%s]", det, path.name, channel)
        ts = TimeSeries.read(str(path), channel=channel)
        ts = ts.crop(t0, t1)
        if float(ts.sample_rate.value) != float(sample_rate):
            ts = ts.resample(sample_rate)
        out[det] = np.asarray(ts.value, dtype=np.float64)
    return out


def build_event_spectrogram_stack(
    td_map: Dict[str, np.ndarray],
    detectors: Sequence[str],
    sample_rate: float,
    *,
    asds: Optional[Dict[str, np.ndarray]] = None,
    delta_f: Optional[float] = None,
    noise_std: Optional[float] = None,
    robust: bool = False,
    norm_stats: Optional[Dict[str, Any]] = None,
    n_time: Optional[int] = None,
    n_freq: Optional[int] = None,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build STFT context; ASD-whiten TD when ``asds`` are provided.

    Training uses whitened FD→TD injections. Raw GWF crops must be whitened
    the same way or ``log_energy`` floors near ``log(eps)`` and PE collapses.

    ``robust=True`` selects the 6-channel fixed-normalized HLV stack used by
    ``train_bns_glitch_robust.py``.
    """
    from adapt.stft_context import (
        build_csd_spectrogram_from_td,
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )

    td_use: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in td_map.items()}
    if asds is not None:
        if delta_f is None or noise_std is None:
            raise ValueError("delta_f and noise_std are required when whitening with asds")
        td_use = whiten_td_map_with_asds(
            td_use,
            asds,
            sample_rate=sample_rate,
            delta_f=float(delta_f),
            noise_std=float(noise_std),
            detectors=detectors,
        )
    if robust:
        kw: Dict[str, Any] = {"energy_detectors": tuple(detectors), "norm_stats": norm_stats}
        if n_time is not None:
            kw["n_time"] = int(n_time)
        if n_freq is not None:
            kw["n_freq"] = int(n_freq)
        if n_fft is not None:
            kw["n_fft"] = int(n_fft)
        if win_length is not None:
            kw["win_length"] = int(win_length)
        if hop_length is not None:
            kw["hop_length"] = int(hop_length)
        return build_robust_spectrogram_from_td(td_use, sample_rate, **kw)
    return build_csd_spectrogram_from_td(
        td_use,
        sample_rate,
        energy_detectors=tuple(detectors),
    )


def _load_matching_state_dict(module: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Copy equal-shape keys; leave new energy head / widened fuse layers at init."""
    current = module.state_dict()
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in current and current[k].shape == v.shape
    }
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    skipped = [k for k in state_dict if k in current and current[k].shape != state_dict[k].shape]
    logger.info(
        "Partial embedding load: loaded=%d skipped_shape=%d missing=%d unexpected=%d",
        len(filtered),
        len(skipped),
        len(missing),
        len(unexpected),
    )


def standardize_context(
    fixed_context: Dict[str, float],
    standardization: Dict[str, Any],
) -> np.ndarray:
    means = standardization["mean"]
    stds = standardization["std"]
    out = np.empty(len(CONTEXT_PARAMS), dtype=np.float32)
    for i, k in enumerate(CONTEXT_PARAMS):
        mu = float(means.get(k, 0.0))
        sig = float(stds.get(k, 1.0) or 1.0)
        out[i] = (float(fixed_context[k]) - mu) / sig
    return out


def load_custom_wrapper(
    baseline_ckpt: Path,
    custom_ckpt: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    from adapt.models import (
        ContextAwareGlitchCorrector,
        GlitchRobustBNSEmbedding,
        ResidualSpectrogramBNSEmbedding,
        Spectrogram2DNet,
        SpectrogramBNSEmbedding,
        SpectrogramResidualHead,
    )
    from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net

    baseline = _torch_load(baseline_ckpt)
    custom = _torch_load(custom_ckpt)
    mk = copy.deepcopy(baseline["model_kwargs"])
    wrapper = create_nsf_with_rb_projection_embedding_net(
        mk["nsf_kwargs"], mk["embedding_net_kwargs"]
    )
    wrapper.load_state_dict(baseline["model_state_dict"], strict=True)

    cmeta = custom.get("metadata") or {}
    adapt = cmeta.get("adapt_config") or {}
    # Prefer flattened keys, then adapt_config, then cnn_base for legacy ckpts.
    encoder_type = str(
        cmeta.get("encoder_type")
        or adapt.get("encoder_type")
        or "cnn_base"
    )
    encoder_channels = (
        cmeta.get("encoder_channels")
        or adapt.get("encoder_channels")
        or [64, 128, 256, 512]
    )
    n_time = int(cmeta.get("n_time") or adapt.get("n_time") or 5)
    n_spec_freq = int(cmeta.get("n_freq") or adapt.get("n_freq") or 128)
    in_channels = int(
        cmeta.get("csd_channels") or adapt.get("csd_channels") or 3
    )
    layout = str(
        cmeta.get("spectrogram_layout")
        or adapt.get("spectrogram_layout")
        or "hl_coh_3ch"
    )
    glitch_robust = bool(
        cmeta.get("glitch_robust", adapt.get("glitch_robust", False))
    )
    residual = bool(
        cmeta.get("residual_dingo", adapt.get("residual_dingo", True))
    )
    # Legacy full-replacement ckpts have no base_embedding in state dict.
    emb_sd = custom.get("embedding_state_dict") or {}
    if residual and not any(k.startswith("base_embedding.") for k in emb_sd):
        residual = False
    # Detect STFT-only residual head vs legacy full Spectrogram2DNet residual.
    residual_stft_only = residual and any(
        k.startswith("spect_net.log_scale") or k.startswith("spect_net.energy_proj.0")
        for k in emb_sd
    ) and not any(k.startswith("spect_net.strain_encoder.") for k in emb_sd)
    # Detect gated glitch corrector (corrector.* or spect_net.delta_net.*).
    has_corrector = any(
        k.startswith("corrector.") or k.startswith("spect_net.delta_net.")
        or k.startswith("spect_net.gate_net.")
        for k in emb_sd
    )
    glitch_robust = glitch_robust or has_corrector
    logger.info(
        "Building custom embedding: glitch_robust=%s residual_dingo=%s stft_only=%s "
        "encoder_type=%s channels=%s grid=(%d,%d) csd_channels=%d layout=%s",
        glitch_robust,
        residual,
        residual_stft_only,
        encoder_type,
        encoder_channels,
        n_time,
        n_spec_freq,
        in_channels,
        layout,
    )
    if glitch_robust:
        corrector = ContextAwareGlitchCorrector(
            n_time=n_time,
            n_spec_freq=n_spec_freq,
            in_channels=in_channels,
            encoder_type=encoder_type,
            encoder_channels=encoder_channels,
        )
        wrapper.embedding_net = GlitchRobustBNSEmbedding(
            wrapper.embedding_net, corrector=corrector
        )
    elif residual:
        if residual_stft_only or not any(
            k.startswith("spect_net.strain_encoder.") for k in emb_sd
        ):
            spect_net: nn.Module = SpectrogramResidualHead(
                n_time=n_time,
                n_spec_freq=n_spec_freq,
                encoder_type=encoder_type,
                encoder_channels=encoder_channels,
                in_channels=in_channels,
            )
        else:
            spect_net = Spectrogram2DNet(
                n_time=n_time,
                n_spec_freq=n_spec_freq,
                encoder_type=encoder_type,
                encoder_channels=encoder_channels,
                in_channels=in_channels,
            )
        wrapper.embedding_net = ResidualSpectrogramBNSEmbedding(
            wrapper.embedding_net, spect_net=spect_net
        )
    else:
        spect_net = Spectrogram2DNet(
            n_time=n_time,
            n_spec_freq=n_spec_freq,
            encoder_type=encoder_type,
            encoder_channels=encoder_channels,
            in_channels=in_channels,
        )
        wrapper.embedding_net = SpectrogramBNSEmbedding(spect_net)
    if "flow_state_dict" in custom:
        wrapper.flow.load_state_dict(custom["flow_state_dict"], strict=True)
    _load_matching_state_dict(wrapper.embedding_net, emb_sd)
    wrapper = wrapper.float().to(device).eval()

    metadata = baseline["metadata"]
    # Prefer standardization / inference params from custom metadata when present.
    if cmeta:
        if "standardization" in cmeta:
            metadata = copy.deepcopy(metadata)
            metadata["train_settings"]["data"]["standardization"] = cmeta[
                "standardization"
            ]
        if "inference_parameters" in cmeta:
            metadata["train_settings"]["data"]["inference_parameters"] = cmeta[
                "inference_parameters"
            ]
    # Attach STFT preprocessing metadata for callers (norm_stats, layout).
    metadata = copy.deepcopy(metadata)
    metadata["_custom_meta"] = cmeta
    metadata["_norm_stats"] = cmeta.get("norm_stats") or custom.get("norm_stats")
    metadata["_glitch_robust"] = glitch_robust
    metadata["_stft_kwargs"] = {
        "n_time": n_time,
        "n_freq": n_spec_freq,
        "n_fft": int(adapt.get("n_fft") or cmeta.get("n_fft") or 2048),
        "win_length": int(adapt.get("win_length") or cmeta.get("win_length") or 1024),
        "hop_length": int(adapt.get("hop_length") or cmeta.get("hop_length") or 256),
        "csd_channels": in_channels,
        "spectrogram_layout": layout,
    }
    return wrapper, metadata


def denormalize_samples(
    y: torch.Tensor,
    inference_params: Sequence[str],
    standardization: Dict[str, Any],
    fixed_context: Dict[str, float],
) -> pd.DataFrame:
    arr = y.detach().cpu().numpy()
    means = standardization["mean"]
    stds = standardization["std"]
    data: Dict[str, np.ndarray] = {}
    for i, name in enumerate(inference_params):
        mu = float(means.get(name, 0.0))
        sig = float(stds.get(name, 1.0) or 1.0)
        data[name] = arr[:, i] * sig + mu

    n = arr.shape[0]
    for k, v in fixed_context.items():
        data[k] = np.full(n, float(v), dtype=np.float64)

    # Reconstruct chirp_mass = delta_chirp_mass + chirp_mass_proxy (DINGO post-process)
    if "delta_chirp_mass" in data and "chirp_mass_proxy" in data:
        data["chirp_mass"] = data["delta_chirp_mass"] + data["chirp_mass_proxy"]

    return pd.DataFrame(data)


def run_custom_sampling(
    wrapper: nn.Module,
    metadata: Dict[str, Any],
    strain: np.ndarray,
    spectrogram: np.ndarray,
    log_energy: np.ndarray,
    context_z: np.ndarray,
    fixed_context: Dict[str, float],
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> pd.DataFrame:
    inference_params = list(metadata["train_settings"]["data"]["inference_parameters"])
    standardization = metadata["train_settings"]["data"]["standardization"]

    strain_t = torch.from_numpy(np.asarray(strain, dtype=np.float32))
    spec_t = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32))
    e_t = torch.from_numpy(np.asarray(log_energy, dtype=np.float32))
    z_t = torch.from_numpy(np.asarray(context_z, dtype=np.float32))

    chunks: List[pd.DataFrame] = []
    remaining = int(num_samples)
    with torch.no_grad():
        while remaining > 0:
            bs = min(int(batch_size), remaining)
            # Context batch of size bs; draw one sample per context row.
            s = strain_t.unsqueeze(0).expand(bs, -1, -1, -1).to(device)
            # Spectrogram is (5, T, F) CSD image → batch to (B, 5, T, F).
            sp = spec_t.unsqueeze(0).expand(bs, -1, -1, -1).to(device)
            e = e_t.unsqueeze(0).expand(bs, -1).to(device)
            z = z_t.unsqueeze(0).expand(bs, -1).to(device)
            y = wrapper.sample(s, sp, e, z, num_samples=1)
            if y.ndim == 3:
                y = y.squeeze(1)
            if y.ndim == 1:
                y = y.unsqueeze(0)
            chunks.append(
                denormalize_samples(y, inference_params, standardization, fixed_context)
            )
            remaining -= bs

    df = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Custom sampling done: %d samples, columns=%s",
        len(df),
        list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# PDF reports
# ---------------------------------------------------------------------------


def _summary_rows(df: pd.DataFrame, params: Sequence[str]) -> List[List[str]]:
    rows = []
    for p in params:
        if p not in df.columns:
            continue
        med, lo, hi = _quantile_ci(df[p].to_numpy())
        rows.append([p, f"{med:.6g}", f"[{lo:.6g}, {hi:.6g}]"])
    return rows


def _draw_table(ax, title: str, rows: List[List[str]], col_labels: Sequence[str]) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, pad=10)
    if not rows:
        ax.text(0.5, 0.5, "No parameters available", ha="center", va="center")
        return
    table = ax.table(
        cellText=rows,
        colLabels=list(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)


def _corner_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    params: Sequence[str],
    title: str,
    color: str = "#1f77b4",
) -> None:
    import corner

    use = [
        p for p in params if p in df.columns and _has_dynamic_range(df[p].to_numpy())
    ]
    if len(use) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient varying parameters for corner", ha="center")
        pdf.savefig(fig, dpi=200)
        plt.close(fig)
        return

    serif_old = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    data = df[use].to_numpy()
    ranges = []
    for j in range(data.shape[1]):
        col = data[:, j]
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi <= lo:
            hi = lo + 1e-6
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    fig = corner.corner(
        data,
        labels=use,
        color=color,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        levels=(0.5, 0.9),
        bins=30,
        no_fill_contours=True,
        range=ranges,
    )
    fig.suptitle(title, fontsize=14, y=1.02)
    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    mpl.rcParams["font.family"] = serif_old


def write_individual_report(
    path: Path,
    df: pd.DataFrame,
    *,
    model_name: str,
    source_note: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [p for p in COMPARE_PARAMS if p in df.columns]
    extra = [
        p
        for p in ("lambda_1", "lambda_2", "chi_1", "chi_2", "geocent_time")
        if p in df.columns and p not in params
    ]
    summary_params = params + extra

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 7))
        rows = _summary_rows(df, summary_params)
        _draw_table(
            ax,
            f"{model_name} — median & 90% CI (N={len(df)})\n{source_note}",
            rows,
            ["parameter", "median", "90% CI [5%, 95%]"],
        )
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
        _corner_page(pdf, df, params, f"{model_name} posterior corner")
    logger.info("Wrote %s", path)


def write_comparison_report(
    path: Path,
    baseline_df: pd.DataFrame,
    custom_df: pd.DataFrame,
) -> None:
    import corner

    path.parent.mkdir(parents=True, exist_ok=True)
    # Summary table includes fixed proxies (ra/dec); corner only varying params.
    table_params = [
        p
        for p in COMPARE_PARAMS
        if p in baseline_df.columns and p in custom_df.columns
    ]
    params = [
        p
        for p in table_params
        if _has_dynamic_range(baseline_df[p].to_numpy())
        and _has_dynamic_range(custom_df[p].to_numpy())
    ]

    with PdfPages(path) as pdf:
        # Summary comparison table
        fig, ax = plt.subplots(figsize=(11, 7))
        rows = []
        for p in table_params:
            b_med, b_lo, b_hi = _quantile_ci(baseline_df[p].to_numpy())
            c_med, c_lo, c_hi = _quantile_ci(custom_df[p].to_numpy())
            rows.append(
                [
                    p,
                    f"{b_med:.6g}",
                    f"[{b_lo:.6g}, {b_hi:.6g}]",
                    f"{c_med:.6g}",
                    f"[{c_lo:.6g}, {c_hi:.6g}]",
                ]
            )
        _draw_table(
            ax,
            (
                f"GW170817 model comparison — medians & 90% CI\n"
                f"baseline N={len(baseline_df)} | custom STFT N={len(custom_df)}"
            ),
            rows,
            [
                "parameter",
                "baseline median",
                "baseline 90% CI",
                "custom median",
                "custom 90% CI",
            ],
        )
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Overlaid corner
        if len(params) >= 2:
            serif_old = mpl.rcParams["font.family"]
            mpl.rcParams["font.family"] = "serif"
            colors = ["#1f77b4", "#d62728"]
            labels = ["baseline DINGO-BNS", "custom Spectrogram2DNet"]
            fig = None
            handles = []
            for color, label, df in zip(
                colors, labels, [baseline_df, custom_df]
            ):
                data = df[params].to_numpy()
                ranges = []
                for j in range(data.shape[1]):
                    col = data[:, j]
                    lo, hi = float(np.min(col)), float(np.max(col))
                    if hi <= lo:
                        hi = lo + 1e-6
                    pad = 0.05 * (hi - lo)
                    ranges.append((lo - pad, hi + pad))
                fig = corner.corner(
                    data,
                    labels=params,
                    color=color,
                    smooth=1.0,
                    smooth1d=1.0,
                    plot_datapoints=False,
                    plot_density=False,
                    plot_contours=True,
                    levels=(0.5, 0.9),
                    bins=30,
                    no_fill_contours=True,
                    fig=fig,
                    range=ranges,
                )
                handles.append(
                    plt.Line2D([], [], color=color, label=label, linewidth=3)
                )
            assert fig is not None
            fig.legend(handles=handles, loc="upper right", fontsize=11)
            fig.suptitle("GW170817 baseline vs custom STFT", fontsize=14, y=1.02)
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            plt.close(fig)
            mpl.rcParams["font.family"] = serif_old

    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def save_samples_pt(
    path: Path,
    df: pd.DataFrame,
    *,
    fixed_context: Dict[str, float],
    source_paths: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "samples": dataframe_to_arrays(df),
        "columns": list(df.columns),
        "num_samples": int(len(df)),
        "fixed_context": dict(fixed_context),
        "source_paths": {k: str(v) if isinstance(v, Path) else v for k, v in source_paths.items()},
    }
    if "log_prob" in df.columns:
        payload["log_prob"] = df["log_prob"].to_numpy()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Wrote %s", path)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = select_device(args.device)
    logger.info("Device: %s", device)

    assets = discover_assets(
        downloads=Path(args.downloads) if args.downloads else DOWNLOADS_DIR,
        pipe_dir=Path(args.pipe_dir) if args.pipe_dir else PIPE_DIR,
        baseline_ckpt=Path(args.baseline_ckpt) if args.baseline_ckpt else None,
        custom_ckpt=Path(args.custom_ckpt) if args.custom_ckpt else None,
    )
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Baseline ----
    logger.info("===== Baseline DINGO-BNS sampling (%d) =====", args.num_samples)
    baseline_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        event,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # ---- Custom ----
    logger.info("===== Custom STFT sampling (%d) =====", args.num_samples)
    wrapper, metadata = load_custom_wrapper(
        assets["baseline_ckpt"], assets["custom_ckpt"], device
    )
    strain = package_event_strain(event.data, metadata, fixed)
    sample_rate = float(
        event.settings.get("f_s")
        or metadata["train_settings"]["data"]["window"]["f_s"]
    )
    from adapt.train_t1 import SPECTROGRAM_ANALYSIS_SECONDS

    td_map = load_event_td_crops(
        assets, sample_rate=sample_rate, crop_seconds=SPECTROGRAM_ANALYSIS_SECONDS
    )
    detectors = list(metadata["train_settings"]["data"]["detectors"])
    from dingo.gw.domains import build_domain_from_model_metadata

    base_domain = build_domain_from_model_metadata(metadata, base=True)
    stft_kw = metadata.get("_stft_kwargs") or {}
    spectrogram, log_energy = build_event_spectrogram_stack(
        td_map,
        detectors,
        sample_rate,
        asds={det: np.asarray(event.data["asds"][det]) for det in detectors},
        delta_f=float(base_domain.delta_f),
        robust=bool(metadata.get("_glitch_robust")),
        norm_stats=metadata.get("_norm_stats"),
        n_time=stft_kw.get("n_time"),
        n_freq=stft_kw.get("n_freq"),
        n_fft=stft_kw.get("n_fft"),
        win_length=stft_kw.get("win_length"),
        hop_length=stft_kw.get("hop_length"),
        noise_std=float(base_domain.noise_std),
    )
    context_z = standardize_context(
        fixed, metadata["train_settings"]["data"]["standardization"]
    )
    logger.info(
        "Custom inputs: strain=%s spectrogram=%s log_energy=%s context_z=%s",
        strain.shape,
        spectrogram.shape,
        log_energy.shape,
        context_z.shape,
    )
    custom_df = run_custom_sampling(
        wrapper,
        metadata,
        strain,
        spectrogram,
        log_energy,
        context_z,
        fixed,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    source_paths = {
        "event_hdf5": assets["event_hdf5"],
        "baseline_ckpt": assets["baseline_ckpt"],
        "custom_ckpt": assets["custom_ckpt"],
        "strain_gwf": {k: str(v) for k, v in assets["strain_gwf"].items()},
        "psd": {k: str(v) for k, v in assets["psd"].items()},
    }

    baseline_pt = outdir / "gw170817_baseline_samples.pt"
    custom_pt = outdir / "gw170817_custom_stft_samples.pt"
    save_samples_pt(
        baseline_pt,
        baseline_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={"model": "baseline_dingo_bns"},
    )
    save_samples_pt(
        custom_pt,
        custom_df,
        fixed_context=fixed,
        source_paths=source_paths,
        extra={
            "model": "custom_spectrogram_stft",
            "spectrogram_shape": tuple(spectrogram.shape),
            "strain_shape": tuple(strain.shape),
            "stft_source": "event_gwf_td_crops",
        },
    )

    write_individual_report(
        outdir / "GW170817_baseline_report.pdf",
        baseline_df,
        model_name="Baseline DINGO-BNS",
        source_note=f"ckpt={assets['baseline_ckpt'].name}",
    )
    write_individual_report(
        outdir / "GW170817_custom_stft_report.pdf",
        custom_df,
        model_name="Custom Spectrogram2DNet + NSF",
        source_note=f"ckpt={assets['custom_ckpt'].name}",
    )
    write_comparison_report(
        outdir / "GW170817_model_comparison.pdf",
        baseline_df,
        custom_df,
    )
    logger.info("Evaluation complete. Artifacts in %s", outdir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GW170817 baseline vs custom STFT evaluation")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--custom-ckpt", type=Path, default=None)
    p.add_argument("--downloads", type=Path, default=None)
    p.add_argument("--pipe-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
