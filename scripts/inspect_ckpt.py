#!/usr/bin/env python3
"""Inspect native specifications of models_checkpoint/dingo_t1.pt.

Read-only: never modifies the checkpoint. Prints top-level keys, inference
parameter order, standardization mean/std, domain / window / detector specs,
and embedding / flow architecture summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "models_checkpoint" / "dingo_t1.pt"


def _load_ckpt(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _prefix_counts(keys) -> Dict[str, int]:
    counts: Counter = Counter()
    for k in keys:
        top = k.split(".", 1)[0]
        counts[top] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def inspect(path: Path) -> int:
    if not path.is_file():
        print(f"ERROR: checkpoint not found: {path}", file=sys.stderr)
        return 1

    raw = _load_ckpt(path)
    print("=" * 72)
    print(f"Checkpoint: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.1f} MiB")
    print("=" * 72)

    print("\n[1] Top-level keys")
    for k in raw.keys():
        v = raw[k]
        if isinstance(v, dict):
            print(f"  {k}: dict ({len(v)} entries)")
        elif torch.is_tensor(v):
            print(f"  {k}: Tensor {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}"[:120])

    print("\n[2] Version / training progress")
    print(f"  version   = {raw.get('version')}")
    print(f"  epoch     = {raw.get('epoch')}")
    print(f"  iteration = {raw.get('iteration')}")

    meta = raw.get("metadata") or {}
    train = meta.get("train_settings") or {}
    data = train.get("data") or {}
    model_cfg = train.get("model") or {}
    ds = meta.get("dataset_settings") or {}

    inference_params = list(data.get("inference_parameters") or [])
    print("\n[3] Target parameter ordering (inference_parameters)")
    if not inference_params:
        print("  (missing)")
    else:
        for i, name in enumerate(inference_params):
            print(f"  {i:2d}. {name}")

    std = data.get("standardization") or {}
    means = std.get("mean") or {}
    stds = std.get("std") or {}
    print("\n[4] Standardization (mean / std)")
    if not means:
        print("  (missing)")
    else:
        print(f"  {'parameter':<22} {'mean':>14} {'std':>14}")
        print("  " + "-" * 52)
        for name in inference_params or sorted(means.keys()):
            mu = means.get(name, float("nan"))
            sig = stds.get(name, float("nan"))
            print(f"  {name:<22} {mu:14.6g} {sig:14.6g}")

    print("\n[5] Domain & data conditioning")
    domain = ds.get("domain") or {}
    window = data.get("window") or {}
    detectors = data.get("detectors")
    wfg = ds.get("waveform_generator") or {}
    print(f"  detectors     = {detectors}")
    print(f"  window        = {window}")
    print(f"  ref_time      = {data.get('ref_time')}")
    print(f"  domain        = {json.dumps(domain, indent=4)}")
    print(f"  waveform_gen  = {wfg}")
    print(f"  tokenization  = {json.dumps(data.get('tokenization') or {}, indent=4)}")

    base = domain.get("base_domain") or {}
    print("\n  Derived frequency specs:")
    print(f"    f_min (base)     = {base.get('f_min')}")
    print(f"    f_max (base)     = {base.get('f_max')}")
    print(f"    delta_f (base)   = {base.get('delta_f')}")
    print(f"    MFD nodes        = {domain.get('nodes')}")
    print(f"    delta_f_initial  = {domain.get('delta_f_initial')}")
    print(f"    sample_rate f_s  = {window.get('f_s')}")
    print(f"    duration T       = {window.get('T')}")

    print("\n[6] Model architecture (model_kwargs / train model cfg)")
    mk = raw.get("model_kwargs") or {}
    print(f"  posterior_model_type = {mk.get('posterior_model_type') or model_cfg.get('posterior_model_type')}")
    print(f"  embedding_type       = {mk.get('embedding_type') or model_cfg.get('embedding_type')}")
    ek = mk.get("embedding_kwargs") or model_cfg.get("embedding_kwargs") or {}
    pk = mk.get("posterior_kwargs") or model_cfg.get("posterior_kwargs") or {}
    print(f"  embedding_kwargs     = {json.dumps(ek, indent=4)}")
    print(f"  posterior_kwargs     = {json.dumps(pk, indent=4)}")

    sd = raw.get("model_state_dict") or {}
    print("\n[7] model_state_dict key counts")
    print(f"  total tensors = {len(sd)}")
    for pref, n in _prefix_counts(sd.keys()).items():
        print(f"    {pref}.* : {n}")
    # finer embedding/flow split
    n_emb = sum(1 for k in sd if k.startswith("embedding_net."))
    n_flow = sum(1 for k in sd if k.startswith("flow."))
    print(f"  embedding_net.* = {n_emb}")
    print(f"  flow.*          = {n_flow}")

    print("\nDone.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dingo_t1.pt native specs")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to checkpoint (default: {DEFAULT_CKPT})",
    )
    args = parser.parse_args()
    raise SystemExit(inspect(args.ckpt))


if __name__ == "__main__":
    main()
