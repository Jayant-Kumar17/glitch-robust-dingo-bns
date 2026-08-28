"""GW150914 head-to-head: industrial official DINGO-T1 vs site-adapted lite model.

Loads an industrial-geometry wrapper from ``models_checkpoint/dingo_t1.pt`` and
the streaming adapted weights from ``dingo_t1_adapted.pt``, simulates GW150914
at dual frequency resolutions, unnormalizes predictions to physical units, and
writes ``models_checkpoint/real_event_comparison.pdf``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adapt.models.dingo_t1 import DingoT1Network
from adapt.pipeline_manager import ADAPTPipelineManager
from adapt.train_t1 import (
    ADAPTED_CHECKPOINT,
    CHECKPOINT_DIR,
    CONTEXT_DIM,
    DEFAULT_CHECKPOINT,
    NUM_PARAMS,
    PARAM_HI,
    PARAM_LO,
    generate_15d_gw_strain,
    make_pipeline_noise_template,
    normalize_targets,
    translate_and_load_checkpoint,
)

logger = logging.getLogger(__name__)

REAL_EVENT_PDF = CHECKPOINT_DIR / "real_event_comparison.pdf"

# Published GW150914 median source-frame / extrinsic values (LIGO/Virgo).
GW150914 = {
    "m1": 35.6,  # Msun
    "m2": 30.6,  # Msun
    "distance": 410.0,  # Mpc
    "ra": 1.37,  # rad
    "dec": -1.26,  # rad
}

# Table / plot rows: (display name, param index, unit label)
KEY_PARAMS: List[Tuple[str, int, str]] = [
    ("Mass 1", 0, "Msun"),
    ("Mass 2", 1, "Msun"),
    ("Distance", 9, "Mpc"),
    ("Right Ascension", 11, "rad"),
    ("Declination", 12, "rad"),
]


def unnormalize_predictions(x_norm: torch.Tensor) -> torch.Tensor:
    """Map network outputs from [-1, 1] back to physical ``PARAM_LO``/``PARAM_HI`` units."""
    lo = PARAM_LO.to(device=x_norm.device, dtype=x_norm.dtype)
    hi = PARAM_HI.to(device=x_norm.device, dtype=x_norm.dtype)
    return lo + (x_norm + 1.0) * 0.5 * (hi - lo)


def build_gw150914_physical() -> torch.Tensor:
    """Length-15 PE vector with published GW150914 medians; others at bound midpoints."""
    mid = 0.5 * (PARAM_LO + PARAM_HI)
    physical = mid.clone()
    physical[0] = GW150914["m1"]
    physical[1] = GW150914["m2"]
    physical[9] = GW150914["distance"]
    physical[11] = GW150914["ra"]
    physical[12] = GW150914["dec"]
    logger.info(
        "GW150914 physical vector: m1=%.2f m2=%.2f D=%.1f ra=%.3f dec=%.3f "
        "(remaining dims = PARAM midpoints)",
        float(physical[0]),
        float(physical[1]),
        float(physical[9]),
        float(physical[11]),
        float(physical[12]),
    )
    return physical


def _load_adapted_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise TypeError(f"unsupported adapted checkpoint format: {path}")


def load_dual_models(
    official_ckpt: Path = DEFAULT_CHECKPOINT,
    adapted_ckpt: Path = ADAPTED_CHECKPOINT,
) -> Tuple[DingoT1Network, DingoT1Network, float, int, int]:
    """Return (official industrial, adapted lite) in eval mode + official match stats."""
    if not official_ckpt.is_file():
        raise FileNotFoundError(f"official checkpoint not found: {official_ckpt}")
    if not adapted_ckpt.is_file():
        raise FileNotFoundError(
            f"adapted weights not found: {adapted_ckpt}. Run `python -m adapt.train_t1`."
        )

    official = DingoT1Network(
        n_freq=1024,
        embed_dim=1024,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
        context_dim=CONTEXT_DIM,
        num_params=NUM_PARAMS,
        dropout=0.0,
    )
    match_pct, n_matched, n_target = translate_and_load_checkpoint(
        official, official_ckpt
    )
    logger.info(
        "Official industrial load: match=%.2f%% (%d/%d)",
        match_pct,
        n_matched,
        n_target,
    )

    adapted = DingoT1Network(
        n_freq=128,
        context_dim=CONTEXT_DIM,
        num_params=NUM_PARAMS,
        dropout=0.0,
    )
    adapted.load_state_dict(_load_adapted_state(adapted_ckpt))
    logger.info("Loaded site-adapted weights from %s", adapted_ckpt)

    official.eval()
    adapted.eval()
    return official, adapted, match_pct, n_matched, n_target


def simulate_event_strains(
    physical: torch.Tensor,
    noise_template: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (strain_1024, noise_1024, strain_128, noise_128) for batch size 1."""
    freqs_1024 = torch.linspace(20.0, 512.0, 1024)
    freqs_128 = torch.linspace(20.0, 512.0, 128)
    strain_1024, _ = generate_15d_gw_strain(physical, freqs_1024)
    strain_128, _ = generate_15d_gw_strain(physical, freqs_128)
    strain_1024 = strain_1024.unsqueeze(0)
    strain_128 = strain_128.unsqueeze(0)
    noise = noise_template.expand(1, -1).contiguous()
    return strain_1024, noise, strain_128, noise.clone()


def print_comparison_table(
    published: torch.Tensor,
    official_phys: torch.Tensor,
    adapted_phys: torch.Tensor,
    rows: Sequence[Tuple[str, int, str]] = KEY_PARAMS,
) -> None:
    """Pretty terminal table of published vs dual-model predictions."""
    header = (
        f"{'Parameter':<20} | {'Published':>14} | {'Official DINGO-T1':>18} | "
        f"{'Site-Adapted':>14} | {'Unit':<6}"
    )
    sep = "-" * len(header)
    print(sep)
    print("GW150914 Physical Parameter Verification")
    print(sep)
    print(header)
    print(sep)
    for name, idx, unit in rows:
        pub = float(published[idx])
        off = float(official_phys[idx])
        ada = float(adapted_phys[idx])
        print(
            f"{name:<20} | {pub:14.4f} | {off:18.4f} | {ada:14.4f} | {unit:<6}"
        )
    print(sep)


def save_real_event_pdf(
    published: torch.Tensor,
    official_phys: torch.Tensor,
    adapted_phys: torch.Tensor,
    out_path: Path = REAL_EVENT_PDF,
    rows: Sequence[Tuple[str, int, str]] = KEY_PARAMS,
) -> None:
    """Grouped absolute-error bars vs published GW150914 values."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in rows]
    err_off = [abs(float(official_phys[i]) - float(published[i])) for _, i, _ in rows]
    err_ada = [abs(float(adapted_phys[i]) - float(published[i])) for _, i, _ in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        x - width / 2,
        err_off,
        width,
        label="Official DINGO-T1",
        color="#4682B4",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        err_ada,
        width,
        label="Site-adapted (ADAPT)",
        color="#FF7F0E",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Absolute error vs published GW150914")
    ax.set_xlabel("Physical parameter")
    ax.set_title("GW150914: Official Industrial vs Site-Adapted DINGO-T1")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote real-event comparison PDF: %s", out_path)


@torch.no_grad()
def run_gw150914_benchmark(
    official_ckpt: str = str(DEFAULT_CHECKPOINT),
    adapted_ckpt: str = str(ADAPTED_CHECKPOINT),
) -> Dict[str, Any]:
    """Dual-architecture GW150914 verification with physical readout + PDF."""
    official, adapted, match_pct, n_matched, n_target = load_dual_models(
        Path(official_ckpt), Path(adapted_ckpt)
    )

    manager = ADAPTPipelineManager(
        expected_duration_seconds=256.0,
        sample_rate=4096.0,
        window_size_seconds=4.0,
        history_size=4,
    )
    noise_template = make_pipeline_noise_template(manager, seed=7)

    physical = build_gw150914_physical()
    strain_1024, noise_1024, strain_128, noise_128 = simulate_event_strains(
        physical, noise_template
    )

    pred_off_norm = official(strain_1024, noise_1024).squeeze(0)
    pred_ada_norm = adapted(strain_128, noise_128).squeeze(0)
    official_phys = unnormalize_predictions(pred_off_norm)
    adapted_phys = unnormalize_predictions(pred_ada_norm)

    print_comparison_table(physical, official_phys, adapted_phys)
    save_real_event_pdf(physical, official_phys, adapted_phys, REAL_EVENT_PDF)

    # Normalized-space MAE vs published-filled vector (for logging only).
    tgt_norm = normalize_targets(physical.unsqueeze(0)).squeeze(0)
    mae_off = float((pred_off_norm - tgt_norm).abs().mean())
    mae_ada = float((pred_ada_norm - tgt_norm).abs().mean())
    logger.info(
        "Normalized MAE vs GW150914 vector — official: %.6f | adapted: %.6f",
        mae_off,
        mae_ada,
    )

    return {
        "official_match_pct": match_pct,
        "official_n_matched": n_matched,
        "official_n_target": n_target,
        "official_physical": official_phys.tolist(),
        "adapted_physical": adapted_phys.tolist(),
        "published_physical": physical.tolist(),
        "plot_path": str(REAL_EVENT_PDF),
        "normalized_mae_official": mae_off,
        "normalized_mae_adapted": mae_ada,
    }


def run_benchmark() -> Dict[str, Any]:
    """CLI entry point."""
    return run_gw150914_benchmark()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_gw150914_benchmark()
