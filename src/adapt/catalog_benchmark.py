"""Pure PyTorch shared-flow NPE catalog benchmark.

One shared DINGO-T1 normalizing-flow backbone (``flow.*`` from
``dingo_t1.pt``) plus dual ``DingoT1Network`` embedding heads:

  * Head A (official): Cross-Docked ``embedding_net`` + 1-D PSD→640 context
  * Head B (adapted): ``dingo_t1_adapted.pt`` + 2-D STFT→640 context

Each arm draws 5000 posterior samples; medians are denormalized with the
checkpoint standardization dict and compared to live GWOSC PE.
"""

from __future__ import annotations

import argparse
import copy
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch

from adapt.gwosc_events import fetch_published_parameters
from adapt.models.dingo_t1 import DingoT1Network
from adapt.train_t1 import (
    ADAPTED_CHECKPOINT,
    CHECKPOINT_DIR,
    CONTEXT_DIM,
    DEFAULT_CHECKPOINT,
    FREQ_HI_HZ,
    FREQ_LO_HZ,
    HybridBackgroundPool,
    NUM_PARAMS,
    PARAM_HI,
    PARAM_LO,
    SAMPLE_RATE_HZ,
    generate_15d_gw_strain,
    load_hybrid_background_pair,
    spectrogram_noise_context_from_ifo_pair,
    translate_and_load_checkpoint,
)

logger = logging.getLogger(__name__)

FLOW_CONTEXT_DIM = 128
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_SAMPLE_BATCH = 1000

FAIR_EVENTS: List[Tuple[str, str, int]] = [
    ("GW150914", "GWTC-1-confident", 3),
    ("GW151226", "GWTC-1-confident", 2),
    ("GW170104", "GWTC-1-confident", 2),
    ("GW170814", "GWTC-1-confident", 3),
]

INFERENCE_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]

UNIT_AXIS_LABEL = {
    "Msun": r"Absolute error $\Delta M_{\odot}$",
    "Mpc": r"Absolute error $\Delta\mathrm{Mpc}$",
    "rad": r"Absolute error $\Delta\mathrm{rad}$",
}

GWOSC_HOST = "https://gwosc.org"
EVENTAPI_EVENT_URL = "{host}/eventapi/json/{catalog}/{event_name}/v{version}/"


# ---------------------------------------------------------------------------
# Checkpoint-compatible NSF ResidualNet (layer-norm + resize_layers.4)
# ---------------------------------------------------------------------------


class _CheckpointResidualBlock(nn.Module):
    """Residual block matching ``dingo_t1.pt`` flow weights (layer-norm path)."""

    def __init__(
        self,
        features: int,
        context_features: Optional[int],
        activation=F.relu,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.activation = activation
        self.layer_norm_layers = nn.ModuleList(
            [nn.LayerNorm(features), nn.LayerNorm(features)]
        )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        else:
            self.context_layer = None
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features), nn.Linear(features, features)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None):
        temps = self.layer_norm_layers[0](inputs)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        temps = self.layer_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None and self.context_layer is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1), dim=1
            )
        return inputs + temps


class _CheckpointResidualNet(nn.Module):
    """ResidualNet with context only in blocks; ``resize_layers.4`` as final linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_blocks: int = 2,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,  # ignored; ckpt uses layer-norm
    ):
        super().__init__()
        del use_batch_norm
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                _CheckpointResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                )
                for _ in range(num_blocks)
            ]
        )
        # Identities have no state_dict entries → only resize_layers.4 appears in ckpt.
        self.resize_layers = nn.ModuleList(
            [nn.Identity() for _ in range(4)] + [nn.Linear(hidden_features, out_features)]
        )

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps, context=context)
        for layer in self.resize_layers:
            temps = layer(temps)
        return temps


def _install_checkpoint_residual_net_patch() -> None:
    """Point glasflow ResidualNet at the checkpoint-compatible implementation."""
    import glasflow.nflows.nn.nets as nflows_nets

    nflows_nets.ResidualNet = _CheckpointResidualNet


def load_shared_nsf_flow(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> Tuple[nn.Module, Dict[str, Any], List[str], Dict[str, Dict[str, float]]]:
    """Build NSF from ``posterior_kwargs`` and load ``flow.*`` weights.

    Returns
    -------
    flow, posterior_kwargs, inference_parameters, standardization
    """
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"official checkpoint not found: {checkpoint_path}")

    _install_checkpoint_residual_net_patch()
    from dingo.core.nn.nsf import create_nsf_wrapped

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")

    pk = copy.deepcopy(raw["model_kwargs"]["posterior_kwargs"])
    pk["base_transform_kwargs"].pop("layer_norm", None)

    wrapper = create_nsf_wrapped(**pk)
    flow = wrapper.flow if hasattr(wrapper, "flow") else wrapper

    flow_sd = OrderedDict(
        (k[len("flow.") :], v)
        for k, v in raw["model_state_dict"].items()
        if k.startswith("flow.")
    )
    missing, unexpected = flow.load_state_dict(flow_sd, strict=False)
    n_model = len(flow.state_dict())
    n_loaded = n_model - len(missing)
    logger.info(
        "Shared NSF: loaded %d/%d tensors (missing=%d unexpected=%d) "
        "context_dim=%d input_dim=%d",
        n_loaded,
        n_model,
        len(missing),
        len(unexpected),
        int(pk["context_dim"]),
        int(pk["input_dim"]),
    )
    if n_loaded < 0.9 * n_model:
        raise RuntimeError(
            f"shared NSF weight load too incomplete ({n_loaded}/{n_model}); "
            "ResidualNet patch may not match checkpoint"
        )

    # Probe initial_layer shape.
    try:
        il = flow._transform._transforms[0]._transforms[1].transform_net.initial_layer
        logger.info("NSF initial_layer weight shape=%s", tuple(il.weight.shape))
    except Exception as exc:
        logger.debug("Could not probe initial_layer: %s", exc)

    flow.eval()

    meta = raw.get("metadata") or {}
    data = (meta.get("train_settings") or {}).get("data") or {}
    inference_params = list(data.get("inference_parameters") or INFERENCE_PARAMS)
    std_block = data.get("standardization") or {}
    standardization = {
        "mean": {k: float(v) for k, v in (std_block.get("mean") or {}).items()},
        "std": {k: float(v) for k, v in (std_block.get("std") or {}).items()},
    }
    if len(inference_params) != int(pk["input_dim"]):
        logger.warning(
            "inference_parameters length %d != flow input_dim %d",
            len(inference_params),
            int(pk["input_dim"]),
        )

    return flow, pk, inference_params, standardization


# ---------------------------------------------------------------------------
# Dual embedding heads
# ---------------------------------------------------------------------------


def _load_adapted_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise TypeError(f"unsupported adapted checkpoint format: {path}")


def load_official_final_proj(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> nn.Linear:
    """Load ``embedding_net.final_net`` Linear(1024→128) from the official ckpt."""
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")
    sd = raw["model_state_dict"]
    w_key = "embedding_net.final_net.linear.weight"
    b_key = "embedding_net.final_net.linear.bias"
    if w_key not in sd or b_key not in sd:
        raise KeyError("official checkpoint missing embedding_net.final_net.linear.*")
    weight = sd[w_key]
    bias = sd[b_key]
    proj = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    with torch.no_grad():
        proj.weight.copy_(weight)
        proj.bias.copy_(bias)
    proj.eval()
    logger.info(
        "Official final_proj loaded: Linear(%d→%d)", weight.shape[1], weight.shape[0]
    )
    return proj


def load_dual_embedding_heads(
    official_ckpt: Path = DEFAULT_CHECKPOINT,
    adapted_ckpt: Path = ADAPTED_CHECKPOINT,
) -> Tuple[DingoT1Network, DingoT1Network, nn.Linear, float]:
    """Return (head_A industrial, head_B lite, final_proj_A, official_match_pct)."""
    head_a = DingoT1Network(
        n_freq=1024,
        embed_dim=1024,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
        context_dim=CONTEXT_DIM,
        num_params=NUM_PARAMS,
        dropout=0.0,
    )
    match_pct, n_matched, n_target = translate_and_load_checkpoint(head_a, official_ckpt)
    logger.info(
        "Head A (official embedding) Cross-Dock: %.2f%% (%d/%d)",
        match_pct,
        n_matched,
        n_target,
    )
    final_proj = load_official_final_proj(official_ckpt)

    head_b = DingoT1Network(
        n_freq=128,
        embed_dim=128,
        context_dim=CONTEXT_DIM,
        num_params=NUM_PARAMS,
        dropout=0.0,
    )
    head_b.load_state_dict(_load_adapted_state(adapted_ckpt))
    logger.info("Head B (adapted) loaded from %s", adapted_ckpt)

    head_a.eval()
    head_b.eval()
    return head_a, head_b, final_proj, match_pct


# ---------------------------------------------------------------------------
# Dual 640-D noise contexts + strain synthesis
# ---------------------------------------------------------------------------


def _unit_scale(noise: torch.Tensor) -> torch.Tensor:
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def psd_1024_context_from_ifo_pair(
    h1: np.ndarray,
    l1: np.ndarray,
    sample_rate: float,
) -> torch.Tensor:
    """Official 1-D Welch PSD on 1024-bin grid → packed CONTEXT_DIM=640."""
    x_h = np.asarray(h1, dtype=np.float64).ravel()
    x_l = np.asarray(l1, dtype=np.float64).ravel()
    n = min(x_h.size, x_l.size)
    if n < 32:
        raise ValueError("background crop too short for Welch PSD")
    x = 0.5 * (x_h[:n] + x_l[:n])
    nperseg = int(min(max(256, sample_rate), n))
    freqs, pxx = welch(
        x,
        fs=float(sample_rate),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    mask = freqs > 0
    src_f, src_p = freqs[mask], np.maximum(pxx[mask], 1e-60)
    target_1024 = np.linspace(FREQ_LO_HZ, FREQ_HI_HZ, 1024, dtype=np.float64)
    log_psd = np.log10(
        np.maximum(
            np.interp(target_1024, src_f, src_p, left=src_p[0], right=src_p[-1]),
            1e-60,
        )
    )
    packed = np.interp(
        np.linspace(0.0, 1.0, CONTEXT_DIM),
        np.linspace(0.0, 1.0, 1024),
        log_psd,
    ).astype(np.float32)
    return _unit_scale(torch.from_numpy(packed).unsqueeze(0)).contiguous()


def synthesize_dual_strains(physical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_1024 = torch.linspace(FREQ_LO_HZ, FREQ_HI_HZ, 1024)
    freqs_128 = torch.linspace(FREQ_LO_HZ, FREQ_HI_HZ, 128)
    s1024, _ = generate_15d_gw_strain(physical, freqs_1024)
    s128, _ = generate_15d_gw_strain(physical, freqs_128)
    return s1024.unsqueeze(0), s128.unsqueeze(0)


def build_physical_from_event(event: Dict[str, Any]) -> torch.Tensor:
    mid = 0.5 * (PARAM_LO + PARAM_HI)
    physical = mid.clone()
    m1 = float(event["mass_1_source"])
    m2 = float(event["mass_2_source"])
    if m2 > m1:
        m1, m2 = m2, m1
    physical[0] = min(max(m1, float(PARAM_LO[0])), float(PARAM_HI[0]))
    physical[1] = min(max(m2, float(PARAM_LO[1])), float(PARAM_HI[1]))
    dist = float(event["luminosity_distance"])
    physical[9] = min(max(dist, float(PARAM_LO[9])), float(PARAM_HI[9]))
    if "right_ascension" in event:
        physical[11] = float(event["right_ascension"])
    if "declination" in event:
        physical[12] = float(event["declination"])
    return physical


# ---------------------------------------------------------------------------
# GWOSC PE + denormalization
# ---------------------------------------------------------------------------


def fetch_event_sky(
    event_name: str, catalog: str, version: int, timeout: float = 30.0
) -> Tuple[Optional[float], Optional[float]]:
    url = EVENTAPI_EVENT_URL.format(
        host=GWOSC_HOST.rstrip("/"),
        catalog=catalog,
        event_name=event_name,
        version=version,
    )
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        events = (resp.json().get("events") or {})
        if not events:
            return None, None
        ev = next(iter(events.values()))
        ra = ev.get("right_ascension", ev.get("ra"))
        dec = ev.get("declination", ev.get("dec"))
        if ra is None or dec is None:
            return None, None
        return float(ra), float(dec)
    except Exception as exc:
        logger.debug("sky fetch failed for %s: %s", event_name, exc)
        return None, None


def fetch_fair_event_record(
    event_name: str, catalog: str, version: int
) -> Dict[str, Any]:
    candidates: List[Optional[int]] = [version, 3, 2, 1, None]
    seen = set()
    ordered: List[Optional[int]] = []
    for v in candidates:
        key = "default" if v is None else int(v)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(v)

    published = None
    used_version: Optional[int] = version
    last_err: Optional[Exception] = None
    for ver in ordered:
        try:
            published = fetch_published_parameters(event_name, catalog, version=ver)
            used_version = ver
            break
        except requests.HTTPError as exc:
            last_err = exc
            continue
    if published is None:
        raise RuntimeError(f"failed to fetch PE for {event_name}: {last_err}")

    ra = dec = None
    for ver in (used_version, 3, 2, 1):
        if ver is None:
            continue
        ra, dec = fetch_event_sky(event_name, catalog, int(ver))
        if ra is not None:
            break

    record: Dict[str, Any] = {
        "name": str(published["name"]),
        "mass_1_source": float(published["m1"]),
        "mass_2_source": float(published["m2"]),
        "luminosity_distance": float(published["distance_mpc"]),
        "version": used_version,
    }
    if ra is not None and dec is not None:
        record["right_ascension"] = ra
        record["declination"] = dec
    return record


def chirp_mass_q_to_m1_m2(mc: float, q: float) -> Tuple[float, float]:
    """Convert chirp mass + mass ratio (m2/m1 ≤ 1) to component masses."""
    q = float(np.clip(q, 1e-6, 1.0))
    mc = float(max(mc, 1e-6))
    # Mc = (m1 m2)^{3/5} / (m1+m2)^{1/5}, q = m2/m1
    m1 = mc * ((1.0 + q) ** 0.2) / (q**0.6)
    m2 = q * m1
    return float(m1), float(m2)


def denormalize_flow_samples(
    samples: torch.Tensor,
    inference_params: Sequence[str],
    standardization: Dict[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """Map standardized (N, 14) samples → physical arrays per parameter name."""
    arr = samples.detach().cpu().numpy()
    means = standardization["mean"]
    stds = standardization["std"]
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(inference_params):
        mu = means.get(name, 0.0)
        sig = stds.get(name, 1.0)
        out[name] = arr[:, i] * sig + mu
    if "chirp_mass" in out and "mass_ratio" in out:
        m1s, m2s = [], []
        for mc, q in zip(out["chirp_mass"], out["mass_ratio"]):
            m1, m2 = chirp_mass_q_to_m1_m2(float(mc), float(q))
            m1s.append(m1)
            m2s.append(m2)
        out["mass_1"] = np.asarray(m1s, dtype=np.float64)
        out["mass_2"] = np.asarray(m2s, dtype=np.float64)
    return out


def median_physical(phys: Dict[str, np.ndarray]) -> Dict[str, float]:
    keys = ["mass_1", "mass_2", "luminosity_distance", "ra", "dec"]
    return {k: float(np.median(phys[k])) for k in keys if k in phys}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_from_flow(
    flow: nn.Module,
    context: torch.Tensor,
    num_samples: int,
    batch_size: int = DEFAULT_SAMPLE_BATCH,
    *,
    event_label: str = "",
    arm_label: str = "",
) -> torch.Tensor:
    """Draw ``num_samples`` from the shared NSF conditioned on ``context`` (1, 128).

    Uses nflows semantics: ``flow.sample(n, context=(1, C))`` → ``(1, n, D)``.
    Never expands context to ``(n, C)`` (that yields an ``(n, n, D)`` tensor and
    O(n²) CPU cost / hangs).
    """
    if context.ndim == 1:
        context = context.unsqueeze(0)
    if context.shape[0] != 1:
        context = context[:1]
    if context.shape[-1] != FLOW_CONTEXT_DIM:
        raise ValueError(
            f"flow context dim {context.shape[-1]} != {FLOW_CONTEXT_DIM}"
        )

    context = context.detach().contiguous()
    samples_list: List[torch.Tensor] = []
    gathered = 0
    n_total = int(num_samples)
    bs = max(1, int(batch_size))
    prefix = f"{event_label}: " if event_label else ""
    arm = f" [{arm_label}]" if arm_label else ""

    while gathered < n_total:
        n = min(bs, n_total - gathered)
        # Keep context as (1, 128); request n draws → (1, n, 14).
        raw = flow.sample(n, context=context)
        if raw.ndim == 3:
            # Expected: [context_size=1, num_samples=n, features]
            raw = raw.reshape(-1, raw.shape[-1])
        elif raw.ndim != 2:
            raise RuntimeError(
                f"unexpected sample ndim={raw.ndim} shape={tuple(raw.shape)}"
            )
        if raw.shape[0] < n:
            raise RuntimeError(
                f"flow.sample returned {raw.shape[0]} rows, expected >= {n}"
            )
        samples_list.append(raw[:n, :14].float())
        gathered += n
        print(
            f"{prefix}Gathered {gathered}/{n_total} samples{arm}...",
            flush=True,
        )

    return torch.cat(samples_list, dim=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_event_grid(
    event_name: str,
    truth: Dict[str, float],
    official: Dict[str, float],
    adapted: Dict[str, float],
) -> None:
    rows = [
        ("Mass 1", "mass_1", "Msun"),
        ("Mass 2", "mass_2", "Msun"),
        ("Distance", "luminosity_distance", "Mpc"),
        ("Right Ascension", "ra", "rad"),
        ("Declination", "dec", "rad"),
    ]
    header = (
        f"{'Event':<14} | {'Parameter':<16} | {'API True':>12} | "
        f"{'Official NPE':>12} | {'Adapted NPE':>12} | {'Unit':<6}"
    )
    if not hasattr(print_event_grid, "_hdr"):
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        print_event_grid._hdr = True  # type: ignore[attr-defined]

    for display, key, unit in rows:
        if key not in truth:
            continue
        print(
            f"{event_name:<14} | {display:<16} | {truth[key]:12.4f} | "
            f"{official.get(key, float('nan')):12.4f} | "
            f"{adapted.get(key, float('nan')):12.4f} | {unit:<6}",
            flush=True,
        )


def save_fair_comparison_pdf(
    event_names: Sequence[str],
    abs_err_official: Dict[str, List[float]],
    abs_err_adapted: Dict[str, List[float]],
    param_units: Dict[str, str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unit_order = ["Msun", "Mpc", "rad"]
    panels: List[Tuple[str, List[str]]] = []
    for unit in unit_order:
        names = [p for p, u in param_units.items() if u == unit and p in abs_err_official]
        if names:
            panels.append((unit, names))
    if not panels:
        raise RuntimeError("no scored parameters for PDF")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.8), squeeze=False)
    x_events = np.arange(len(event_names))
    width = 0.35
    for ax, (unit, param_names) in zip(axes[0], panels):
        mean_off = [
            float(np.mean([abs_err_official[p][i] for p in param_names]))
            for i in range(len(event_names))
        ]
        mean_ada = [
            float(np.mean([abs_err_adapted[p][i] for p in param_names]))
            for i in range(len(event_names))
        ]
        ax.bar(x_events - width / 2, mean_off, width, label="Official NPE", color="#4682B4")
        ax.bar(
            x_events + width / 2, mean_ada, width, label="Adapted NPE", color="#FF7F0E"
        )
        ax.set_xticks(x_events)
        ax.set_xticklabels(list(event_names), rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(UNIT_AXIS_LABEL.get(unit, f"Absolute error [{unit}]"))
        ax.set_title(", ".join(param_names), fontsize=10)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Shared-Flow NPE — Official PSD Head vs Adapted STFT Head (5000 samples)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote fair comparison PDF: %s", out_path)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_fair_catalog_benchmark(
    *,
    seed: int = 7,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_SAMPLE_BATCH,
) -> Dict[str, Any]:
    flow, _pk, inference_params, standardization = load_shared_nsf_flow()
    head_a, head_b, final_proj, match_pct = load_dual_embedding_heads()

    # Disable dropout / tracking before any event work.
    flow.eval()
    head_a.eval()
    head_b.eval()
    final_proj.eval()

    h1_full, l1_full, bg_sr, bg_source = load_hybrid_background_pair(
        duration_seconds=256.0, sample_rate=SAMPLE_RATE_HZ, seed=seed
    )
    pool = HybridBackgroundPool(h1_full, l1_full, sample_rate=bg_sr, source=bg_source)

    print()
    print("Pure PyTorch Shared-Flow NPE Catalog Benchmark")
    print(f"Shared NSF backbone from {DEFAULT_CHECKPOINT.name}")
    print("Head A: official embedding + 1-D PSD→640")
    print("Head B: adapted spectrogram encoder + 2-D STFT→640")
    print(
        f"Samples/event: {num_samples}  batch_size: {batch_size}  "
        f"background: {bg_source}"
    )
    print(f"Events: {[e[0] for e in FAIR_EVENTS]}")
    print()

    scored_names: List[str] = []
    abs_err_official: Dict[str, List[float]] = {}
    abs_err_adapted: Dict[str, List[float]] = {}
    param_units: Dict[str, str] = {
        "Mass 1": "Msun",
        "Mass 2": "Msun",
        "Distance": "Mpc",
        "Right Ascension": "rad",
        "Declination": "rad",
    }
    display_to_key = {
        "Mass 1": "mass_1",
        "Mass 2": "mass_2",
        "Distance": "luminosity_distance",
        "Right Ascension": "ra",
        "Declination": "dec",
    }

    with torch.no_grad():
        for i, (name, catalog, version) in enumerate(FAIR_EVENTS):
            event_label = f"Event {i + 1}/{len(FAIR_EVENTS)} ({name})"
            event = fetch_fair_event_record(name, catalog, version)
            physical = build_physical_from_event(event)
            strain_1024, strain_128 = synthesize_dual_strains(physical)

            rng = np.random.default_rng(int(seed) + 17 * i)
            h1_crop, l1_crop = pool.sample_pair(rng)
            noise_psd = psd_1024_context_from_ifo_pair(
                h1_crop, l1_crop, pool.sample_rate
            )
            noise_stft = spectrogram_noise_context_from_ifo_pair(
                h1_crop, l1_crop, pool.sample_rate
            )

            ctx_a = head_a.encode_flow_context(
                strain_1024, noise_psd, final_proj=final_proj
            )
            ctx_b = head_b.encode_flow_context(
                strain_128, noise_stft, final_proj=None
            )
            if (
                ctx_a.shape[-1] != FLOW_CONTEXT_DIM
                or ctx_b.shape[-1] != FLOW_CONTEXT_DIM
            ):
                raise RuntimeError(
                    f"{name}: bad context shapes {tuple(ctx_a.shape)} / "
                    f"{tuple(ctx_b.shape)}"
                )

            logger.info(
                "[%d/%d] %s ctxΔ=%.4f sampling n=%d batch=%d",
                i + 1,
                len(FAIR_EVENTS),
                name,
                float(torch.norm(ctx_a - ctx_b).item()),
                num_samples,
                batch_size,
            )

            print(f"{event_label}: sampling official (PSD head)...", flush=True)
            samp_a = sample_from_flow(
                flow,
                ctx_a,
                num_samples,
                batch_size=batch_size,
                event_label=event_label,
                arm_label="official",
            )
            print(f"{event_label}: sampling adapted (STFT head)...", flush=True)
            samp_b = sample_from_flow(
                flow,
                ctx_b,
                num_samples,
                batch_size=batch_size,
                event_label=event_label,
                arm_label="adapted",
            )
            if samp_a.shape != (num_samples, 14) or samp_b.shape != (num_samples, 14):
                raise RuntimeError(
                    f"{name}: sample shapes {tuple(samp_a.shape)} / "
                    f"{tuple(samp_b.shape)}"
                )

            phys_a = denormalize_flow_samples(
                samp_a, inference_params, standardization
            )
            phys_b = denormalize_flow_samples(
                samp_b, inference_params, standardization
            )
            med_a = median_physical(phys_a)
            med_b = median_physical(phys_b)

            truth = {
                "mass_1": float(event["mass_1_source"]),
                "mass_2": float(event["mass_2_source"]),
                "luminosity_distance": float(event["luminosity_distance"]),
            }
            if "right_ascension" in event:
                truth["ra"] = float(event["right_ascension"])
                truth["dec"] = float(event["declination"])

            print_event_grid(name, truth, med_a, med_b)
            scored_names.append(name)

            for display, key in display_to_key.items():
                if key not in truth:
                    continue
                e_off = abs(med_a[key] - truth[key])
                e_ada = abs(med_b[key] - truth[key])
                abs_err_official.setdefault(display, []).append(e_off)
                abs_err_adapted.setdefault(display, []).append(e_ada)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = CHECKPOINT_DIR / f"fair_comparison_{ts}.pdf"
    save_fair_comparison_pdf(
        scored_names, abs_err_official, abs_err_adapted, param_units, pdf_path
    )
    print("-" * 100, flush=True)
    print(f"Wrote visual report: {pdf_path}", flush=True)

    return {
        "events": scored_names,
        "num_samples": num_samples,
        "official_match_pct": match_pct,
        "background_source": bg_source,
        "plot_path": str(pdf_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-flow NPE catalog benchmark (PSD vs STFT heads)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SAMPLE_BATCH)
    args = parser.parse_args()
    run_fair_catalog_benchmark(
        seed=args.seed,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
