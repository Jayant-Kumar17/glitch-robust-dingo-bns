#!/usr/bin/env python3
"""Minimal inspection of the downloaded DINGO-BNS checkpoint.

Searches ``DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/downloads/``
for a ``.pt`` file, loads it on CPU, and prints architecture / embedding-head
specs (including the noise/PSD input dimensions).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
    / "downloads"
)


def find_pt(downloads_dir: Path) -> Path:
    pts = sorted(downloads_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"no .pt file found in {downloads_dir}")
    if len(pts) > 1:
        print(f"NOTE: found {len(pts)} .pt files; using {pts[0].name}", flush=True)
    return pts[0]


def load_ckpt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _pretty(obj: Any, max_chars: int = 8000) -> str:
    text = json.dumps(obj, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n  ... (truncated)"
    return text


def _get_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return embedding / flow kwargs from model_kwargs or metadata.train_settings.model."""
    mk = raw.get("model_kwargs") or {}
    train_model = (
        ((raw.get("metadata") or {}).get("train_settings") or {}).get("model") or {}
    )
    # BNS demo uses embedding_net_kwargs + nsf_kwargs; newer dingo uses
    # embedding_kwargs + posterior_kwargs. Accept both.
    emb = (
        mk.get("embedding_net_kwargs")
        or mk.get("embedding_kwargs")
        or train_model.get("embedding_net_kwargs")
        or train_model.get("embedding_kwargs")
        or {}
    )
    flow = (
        mk.get("nsf_kwargs")
        or mk.get("flow_kwargs")
        or mk.get("posterior_kwargs")
        or train_model.get("nsf_kwargs")
        or train_model.get("flow_kwargs")
        or train_model.get("posterior_kwargs")
        or {}
    )
    return {
        "model_type": mk.get("type") or train_model.get("type"),
        "embedding_net_kwargs": emb,
        "flow_kwargs": flow,
        "raw_model_kwargs_keys": list(mk.keys()),
        "raw_train_model_keys": list(train_model.keys()),
    }


def print_embedding_hierarchy(state_dict: Dict[str, torch.Tensor]) -> None:
    """Print condensed embedding_net module tree with parameter shapes."""
    modules: "OrderedDict[str, List[str]]" = OrderedDict()
    for key, tensor in state_dict.items():
        if not key.startswith("embedding_net."):
            continue
        parts = key.split(".")
        mod = ".".join(parts[:-1])
        pname = parts[-1]
        shape = tuple(tensor.shape)
        modules.setdefault(mod, []).append(f"{pname}{list(shape)}")

    if not modules:
        print("  (no embedding_net.* keys in model_state_dict)")
        return

    print(f"  modules with parameters: {len(modules)}")
    # Group by first three path components for readability.
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for mod in modules:
        bits = mod.split(".")
        group = ".".join(bits[:3]) if len(bits) >= 3 else mod
        groups.setdefault(group, []).append(mod)

    for group, mods in groups.items():
        print(f"\n  [{group}]  ({len(mods)} submodules)")
        for mod in mods:
            params = ", ".join(modules[mod][:6])
            extra = f" (+{len(modules[mod]) - 6} more)" if len(modules[mod]) > 6 else ""
            print(f"    {mod}: {params}{extra}")


def print_noise_psd_head(emb_kwargs: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> None:
    """Interpret input_dims / first projection layer as the noise+PSD head."""
    input_dims = emb_kwargs.get("input_dims")
    print("  embedding_net_kwargs['input_dims'] =", input_dims)
    if isinstance(input_dims, (list, tuple)) and len(input_dims) == 3:
        n_ifo, n_channels, n_freq = input_dims
        print(
            f"  Interpreted as DINGO packaging [n_ifo, n_channels, n_freq] = "
            f"[{n_ifo}, {n_channels}, {n_freq}]"
        )
        print(
            "    channel layout (standard dingo): "
            "[0]=Re(strain), [1]=Im(strain), [2]=1/(ASD·1e23)"
        )
        print(f"    noise/PSD head therefore expects ASD on a {n_freq}-bin frequency grid")
        print(f"    per-IFO flattened width = n_channels * n_freq = {n_channels * n_freq}")
        print(f"    full tensor shape before SVD / RB: ({n_ifo}, {n_channels}, {n_freq})")

    svd = emb_kwargs.get("svd") or {}
    if svd:
        print(f"  SVD / RB compression: {svd}")

    # First reduced-basis linear layers (layers_rb) show the flattened input width.
    rb_keys = [
        k
        for k in state_dict
        if k.startswith("embedding_net.") and "layers_rb" in k and k.endswith(".weight")
    ]
    if rb_keys:
        print("  First RB projection weights (noise/strain head):")
        for k in sorted(rb_keys)[:6]:
            w = state_dict[k]
            print(f"    {k}: weight shape {tuple(w.shape)}  (out_features, in_features)")


def inspect(path: Path) -> int:
    raw = load_ckpt(path)
    print("=" * 72)
    print(f"Checkpoint: {path}")
    print(f"Size:       {path.stat().st_size / (1024 ** 2):.1f} MiB")
    print("=" * 72)

    print("\n[1] Top-level keys")
    for k, v in raw.items():
        if isinstance(v, dict):
            print(f"  {k}: dict ({len(v)} entries)")
        elif torch.is_tensor(v):
            print(f"  {k}: Tensor {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}")

    print("\n[2] model_kwargs / metadata embedding & flow specs")
    kw = _get_kwargs(raw)
    print(f"  model type: {kw['model_type']}")
    print(f"  model_kwargs keys: {kw['raw_model_kwargs_keys']}")
    print(f"  metadata.train_settings.model keys: {kw['raw_train_model_keys']}")
    print("\n  embedding_net_kwargs:")
    print(_pretty(kw["embedding_net_kwargs"]))
    print("\n  flow_kwargs (nsf_kwargs / posterior_kwargs):")
    print(_pretty(kw["flow_kwargs"]))

    sd = raw.get("model_state_dict") or {}
    print("\n[3] embedding_net module hierarchy (from state_dict)")
    n_emb = sum(1 for k in sd if k.startswith("embedding_net."))
    n_flow = sum(1 for k in sd if k.startswith("flow."))
    print(f"  state_dict tensors: total={len(sd)}  embedding_net.*={n_emb}  flow.*={n_flow}")
    print_embedding_hierarchy(sd)

    print("\n[4] Noise / PSD input head dimensions")
    print_noise_psd_head(kw["embedding_net_kwargs"], sd)

    print("\nDone.")
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect DINGO-BNS .pt architecture")
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DEFAULT_DOWNLOADS,
        help=f"Directory containing the BNS .pt (default: {DEFAULT_DOWNLOADS})",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional explicit path to a .pt file (skips search)",
    )
    args = parser.parse_args(argv)

    try:
        path = args.ckpt if args.ckpt is not None else find_pt(args.downloads_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not path.is_file():
        print(f"ERROR: checkpoint not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(inspect(path))


if __name__ == "__main__":
    main()
