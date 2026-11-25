#!/usr/bin/env python3
"""Route B on every damaged Stage 3 analysis, plus the damaged_092 leakage test.

Route B reweights the first 10^4 rows of the R1 clean-data proposal with the
gated likelihood of that cell. It does not draw a new network proposal. The
leakage test, only for the two trusted failures (damaged_092 at both pads),
evaluates the same samples twice: the injected strain, and that strain with the
injected glitch removed on the gate-mask samples.

Writes one JSON per cell under results/revision2/routeb/ and does not write samples.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import resolve_under_repo

logger = logging.getLogger("revision2_routeb")
N_PROP = 10_000


def _ci(x, w):
    import gated_is_stage2 as S2
    return S2._ci(np.asarray(x, float), w)


def _pack(c, g_td, n_freq):
    """Event package of the clean archive plus a time-domain H1 addition."""
    import copy
    from adapt.packaging import package_fd

    data = copy.deepcopy(c.event.data)
    fd = package_fd(g_td, c.fs, roll_off=c.roll_off, f_max=c.f_max, n_freq=n_freq, time_buffer=c.tb)
    data["waveform"]["H1"] = np.asarray(c.event.data["waveform"]["H1"], dtype=np.complex128) + fd
    return data


def _likelihood(c, data):
    """Uniform-grid likelihood on ``data`` without a full proposal draw."""
    from adapt.gated_likelihood import build_ufd_likelihood

    # A one-sample batch hits a zero-dimensional torch.cat in GWSamplerGNPE.
    res = c.session.sample_result(data, c.settings, c.fixed, num_samples=8, batch_size=8)
    return build_ufd_likelihood(res), res


def _gated(c, base, wins, theta):
    from revision_eval import chunked_gated
    return chunked_gated(c, base, wins, theta)


def _one(c, inj, row, half, clean, clean_ci, stage3: Path) -> dict:
    import gated_is_stage2 as S2
    from adapt.gated_likelihood import gate_mask_from_windows, reweight
    from adapt.localize import excess_gates
    from official_control import js_divergence_1d
    from stress_gw170817 import _gated_recovers

    rec = {"id": f"{row['group']}_{int(row['cell_id']):03d}_h{half:g}", "group": str(row["group"]),
           "cell_id": int(row["cell_id"]), "half": float(half), "n_proposal": N_PROP}
    poison, meta = inj(row)
    rho_a = float(row["injected_rho_w"])
    rho_b = float(meta["injected_rho_w"])
    if abs(rho_b - rho_a) / max(rho_a, 1e-12) >= 1e-6:
        rec["error"] = f"injection mismatch: rebuilt {rho_b:.6g} vs archived {rho_a:.6g}"
        return rec
    wins, _z = excess_gates({"H1": meta["td_full"]["H1"]}, {"H1": c.asds["H1"]}, c.fs,
                            f_max=c.f_max, pad_s=half, sigma=8.0, detectors=["H1"])
    if not wins:
        rec["error"] = "localiser did not fire"
        return rec
    g = np.asarray(meta["td_full"]["H1"], float) - np.asarray(c.td["H1"], float)
    n_freq = len(np.asarray(c.event.data["waveform"]["H1"]))
    # The likelihood data are the zero-filled gated package, as in Stage 3.
    from adapt.glitch_excision import rebuild_event_from_gated_td
    gated = rebuild_event_from_gated_td(
        poison, td_by_det={"H1": meta["td_full"]["H1"]}, gates=wins, sample_rate=c.fs,
        roll_off=c.roll_off, f_max=c.f_max, original_asds=c.asds, fill="zero", gate_band=(20.0, 23.0),
    )
    base, res = _likelihood(c, gated.data)
    from adapt.gated_likelihood import theta_frame
    keys = [k for k in theta_frame(res).columns if k in clean.columns]
    # Synthetic phase is not a prior parameter, but the waveform generator requires it.
    # These are the phases stored with the clean-data proposal.
    if "phase" in clean.columns and "phase" not in keys:
        keys.append("phase")
    theta = clean.iloc[:N_PROP][keys].reset_index(drop=True)
    L = _gated(c, base, wins, theta)
    block = clean.iloc[:N_PROP].reset_index(drop=True)
    rw = reweight(block, L)
    d_L = _ci(block["luminosity_distance"], rw["weights"])
    trusted = bool(rw["n_eff"] >= 100.0)
    arch_path = stage3 / f"{rec['id']}_samples.csv.gz"
    arch_json = json.loads((stage3 / "cells" / f"{rec['id']}.json").read_text())
    g0 = arch_json["gated_is"]
    old = pd.read_csv(arch_path)
    js16 = float(js_divergence_1d(
        block["luminosity_distance"].to_numpy(float), old["luminosity_distance"].to_numpy(float),
        wx=rw["weights"], wy=old["weight_gated"].to_numpy(float), bins=16))
    shift = float(d_L["med"] - g0["d_L"]["med"])
    rec.update({
        "n_eff": rw["n_eff"], "efficiency": rw["efficiency"], "trusted": trusted,
        "passes": bool(_gated_recovers(d_L, clean_ci)), "d_L": d_L,
        "stage3_trusted": bool(g0["trusted"]), "stage3_n_eff": float(g0["n_eff"]),
        "stage3_d_L_med": float(g0["d_L"]["med"]),
        "trust_agrees": bool(trusted == bool(g0["trusted"])),
        "d_L_median_shift_Mpc": shift, "js_dL_16bin": js16,
        "nonfinite_log_likelihood": int(np.sum(~np.isfinite(L))),
    })
    if rec["id"].startswith("damaged_092"):
        rec["leakage"] = _leakage(c, g, wins, n_freq, theta, block)
    return rec


def _leakage(c, g, wins, n_freq, theta, block) -> dict:
    """Gated log-likelihood with the glitch, minus the same likelihood after the glitch
    is removed on the gate-mask samples. Outside the mask the two strains match."""
    from adapt.gated_likelihood import gate_mask_from_windows

    mask = gate_mask_from_windows(wins, "H1", c.N, c.fs)
    g_out = np.array(g, copy=True)
    g_out[mask] = 0.0
    base_with, _ = _likelihood(c, _pack(c, g, n_freq))
    base_without, _ = _likelihood(c, _pack(c, g_out, n_freq))
    Lw = _gated(c, base_with, wins, theta)
    Lo = _gated(c, base_without, wins, theta)
    d = Lw - Lo
    finite = d[np.isfinite(d)]
    return {
        "n": int(d.size),
        "n_finite": int(finite.size),
        "n_gate_samples": int(np.sum(mask)),
        "glitch_samples_inside_gate": int(np.sum((np.abs(g) > 0) & mask)),
        "glitch_samples_outside_gate": int(np.sum((np.abs(g) > 0) & ~mask)),
        "glitch_energy_inside_gate": float(np.sum(g[mask] ** 2)),
        "glitch_energy_outside_gate": float(np.sum(g[~mask] ** 2)),
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "p05": float(np.percentile(finite, 5)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "frac_abs_gt_0.1": float(np.mean(np.abs(finite) > 0.1)) if finite.size else None,
        "frac_abs_gt_1": float(np.mean(np.abs(finite) > 1.0)) if finite.size else None,
    }


def main(argv=None):
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    from revision_eval import add_runtime_args

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p, default_outdir="results/revision2/routeb", default_samples=N_PROP)
    p.add_argument("--stage3", type=resolve_under_repo, default="results/gated_is_stage3")
    p.add_argument("--ref", type=resolve_under_repo, default="results/revision/clean_ref_v2")
    p.add_argument("--real-dir", type=resolve_under_repo, default="results/stress_real_glitches_v3_final")
    p.add_argument("--clean-ci", type=resolve_under_repo, default="results/gated_is_stage3/clean_reference.json")
    p.add_argument("--max-cells", type=int, default=None)
    args = p.parse_args(argv)
    args.eps_sensitivity = []
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage3 = Path(args.stage3)
    records = [json.loads(f.read_text()) for f in sorted((stage3 / "cells").glob("*.json"))]
    chosen = [r for r in records if r.get("group") == "damaged" and "gated_is" in r and "error" not in r]
    chosen = sorted(chosen, key=lambda r: r["id"])
    if args.max_cells:
        chosen = chosen[: int(args.max_cells)]
    done = {p.stem for p in outdir.glob("*.json")}
    clean = pd.read_csv(Path(args.ref) / "samples.csv.gz").iloc[:N_PROP].reset_index(drop=True)
    clean_ci = json.loads(Path(args.clean_ci).read_text())["clean_ci"] if "clean_ci" in json.loads(Path(args.clean_ci).read_text()) else json.loads(Path(args.clean_ci).read_text())
    # clean_reference.json is the CI dict itself in Stage 3
    raw_ci = json.loads(Path(args.clean_ci).read_text())
    clean_ci = raw_ci["clean_ci"] if isinstance(raw_ci, dict) and "clean_ci" in raw_ci else raw_ci
    panel = S3.classify_cells(Path(args.real_dir)).set_index("cell_id", drop=False)
    logger.info("route B cells %d, already done %d", len(chosen), len(done & {r['id'] for r in chosen}))
    c = S2.setup(args)
    inj = S3.Injector(c, Path(args.real_dir))
    t0 = time.time()
    for src in chosen:
        if src["id"] in done:
            continue
        row = panel.loc[int(src["cell_id"])]
        logger.info("=== %s ===", src["id"])
        try:
            rec = _one(c, inj, row, float(src["half"]), clean, clean_ci, stage3)
        except Exception as exc:
            rec = {"id": src["id"], "cell_id": int(src["cell_id"]), "half": float(src["half"]),
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            logger.exception("failed %s", src["id"])
        (outdir / f"{rec['id']}.json").write_text(json.dumps(rec, indent=2, default=S2._json_default))
        logger.info("finished %s in %.1f min elapsed", rec["id"], (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
