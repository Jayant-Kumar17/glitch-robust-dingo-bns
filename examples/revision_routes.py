#!/usr/bin/env python3
"""R4: three proposals on the pre-registered eight Stage 3 analyses.

A repeats Stage 3 (zero fill, new 10^4 draw). B reweights the first 2x10^4 rows
of the R1 clean-data proposal. C repeats A with noise fill.
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

logger = logging.getLogger("revision_routes")


def choose_route_cells(records: list[dict]) -> list[dict]:
    """The R4 set from docs/REVISION_PLAN.md. Order: trusted failures, then the
    three lowest-n_eff trusted passing damaged analyses, then the three
    highest-n_eff untrusted damaged ±0.1 s analyses."""
    have = [r for r in records if "gated_is" in r and "id" in r]

    def ne(r):
        return float(r["gated_is"]["n_eff"])

    fail = [r for r in have if r["gated_is"].get("trusted") and not r["gated_is"].get("recovers")]
    fail = sorted(fail, key=lambda r: r["id"])
    passed = [r for r in have if r.get("group") == "damaged" and r["gated_is"].get("trusted") and r["gated_is"].get("recovers")]
    smallest = sorted(passed, key=lambda r: (ne(r), r["id"]))[:3]
    un = [r for r in have if r.get("group") == "damaged" and abs(float(r.get("half", -1)) - 0.1) < 1e-9
          and not r["gated_is"].get("trusted")]
    largest = sorted(un, key=lambda r: (-ne(r), r["id"]))[:3]
    out, seen = [], set()
    for r in fail + smallest + largest:
        if r["id"] not in seen:
            out.append(r)
            seen.add(r["id"])
    return out


def build_parser() -> argparse.ArgumentParser:
    from revision_eval import add_runtime_args

    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p, default_outdir="results/revision/routes", default_samples=10_000)
    p.add_argument("--stage3", type=resolve_under_repo, default="results/gated_is_stage3")
    p.add_argument("--ref", type=resolve_under_repo, default="results/revision/clean_ref_v2")
    p.add_argument("--real-dir", type=resolve_under_repo, default="results/stress_real_glitches_v3_final")
    p.add_argument("--clean-ci", type=resolve_under_repo, default="results/gated_is_stage3/clean_reference.json")
    return p


def _route_record(samples, weights, clean_ci, n: int) -> dict:
    import gated_is_stage2 as S2
    import revision_analysis as ra
    from stress_gw170817 import _gated_recovers

    d_L = S2._ci(samples["luminosity_distance"].to_numpy(float), weights)
    with np.errstate(divide="ignore"):
        khat = ra.psis_khat(np.log(np.maximum(weights, 0.0)))
    ne = float(np.sum(weights) ** 2 / np.sum(np.asarray(weights) ** 2)) if np.any(weights) else 0.0
    return {"n": int(n), "n_eff": ne, "efficiency": ne / int(n) if n else 0.0, "d_L": d_L,
            "recovers": bool(_gated_recovers(d_L, clean_ci)), "khat": khat}


def _save_samples(path: Path, samples: pd.DataFrame, weights) -> None:
    from official_control import INFERENCE_PARAMS

    keep = [p for p in INFERENCE_PARAMS if p in samples.columns]
    out = samples[keep].copy()
    out["weight"] = np.asarray(weights, float)
    out.to_csv(path, index=False)


def _analyse(c, inj, row, half, clean_ci, clean_df, stage3: Path, outdir: Path) -> dict:
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    from adapt.gated_likelihood import build_ufd_likelihood, reweight, theta_frame
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.localize import excess_gates
    from official_control import INFERENCE_PARAMS
    from revision_eval import chunked_gated

    rec = {"id": f"{row['group']}_{int(row['cell_id']):03d}_h{half:g}", "group": str(row["group"]),
           "cell_id": int(row["cell_id"]), "half": float(half)}
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

    def _package(fill):
        return rebuild_event_from_gated_td(
            poison, td_by_det={"H1": meta["td_full"]["H1"]}, gates=wins, sample_rate=c.fs,
            roll_off=c.roll_off, f_max=c.f_max, original_asds=c.asds, fill=fill, gate_band=(20.0, 23.0),
        )

    resA, _rawA = S2.draw(c, _package("zero").data, int(c.args.num_samples))
    baseA = build_ufd_likelihood(resA)
    thA = theta_frame(resA)
    LA = chunked_gated(c, baseA, wins, thA)
    wA = reweight(resA.samples, LA)["weights"]
    routes = {"A": _route_record(resA.samples, wA, clean_ci, int(c.args.num_samples))}
    _save_samples(outdir / f"{rec['id']}_routeA_samples.csv.gz", resA.samples, wA)

    n_b = min(20_000, len(clean_df))
    sub = clean_df.iloc[:n_b].reset_index(drop=True)
    keys = list(thA.columns)
    LB = chunked_gated(c, baseA, wins, sub[keys])
    wB = reweight(sub, LB)["weights"]
    routes["B"] = _route_record(sub, wB, clean_ci, n_b)
    _save_samples(outdir / f"{rec['id']}_routeB_samples.csv.gz", sub, wB)

    resC, _rawC = S2.draw(c, _package("noise").data, int(c.args.num_samples))
    baseC = build_ufd_likelihood(resC)
    LC = chunked_gated(c, baseC, wins, theta_frame(resC))
    wC = reweight(resC.samples, LC)["weights"]
    routes["C"] = _route_record(resC.samples, wC, clean_ci, int(c.args.num_samples))
    _save_samples(outdir / f"{rec['id']}_routeC_samples.csv.gz", resC.samples, wC)

    js = {
        "A-B": S2._js(resA.samples, wA, sub, wB, INFERENCE_PARAMS),
        "A-C": S2._js(resA.samples, wA, resC.samples, wC, INFERENCE_PARAMS),
        "B-C": S2._js(sub, wB, resC.samples, wC, INFERENCE_PARAMS),
    }
    arch = stage3 / f"{rec['id']}_samples.csv.gz"
    if arch.is_file():
        old = pd.read_csv(arch)
        js["A-stage3"] = S2._js(resA.samples, wA, old, old["weight_gated"].to_numpy(float), INFERENCE_PARAMS)
    else:
        js["A-stage3"] = {"error": f"missing {arch}"}
    rec["routes"] = routes
    rec["js"] = js
    return rec


def main(argv=None) -> None:
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    from revision_eval import apply_smoke, over_budget

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    apply_smoke(args, cells=2)
    outdir = Path(args.outdir)
    cells_dir = outdir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    stage3 = Path(args.stage3)
    records = [json.loads(f.read_text()) for f in sorted((stage3 / "cells").glob("*.json"))]
    chosen = choose_route_cells(records)
    if args.smoke:
        chosen = chosen[: int(args.max_cells)]
    print("route cells:", " ".join(r["id"] for r in chosen))
    ref = pd.read_csv(Path(args.ref) / "samples.csv.gz")
    clean_ci = json.loads(Path(args.clean_ci).read_text())["clean_ci"]
    panel = S3.classify_cells(Path(args.real_dir)).set_index("cell_id", drop=False)
    done = {json.loads(f.read_text()).get("id") for f in cells_dir.glob("*.json")}
    t_start = time.time()
    c = S2.setup(args)
    inj = S3.Injector(c, Path(args.real_dir))
    for src in chosen:
        if src["id"] in done:
            continue
        if over_budget(t_start, args.max_hours):
            logger.warning("time budget reached before %s", src["id"])
            break
        row = panel.loc[int(src["cell_id"])]
        logger.info("=== route %s ===", src["id"])
        try:
            rec = _analyse(c, inj, row, float(src["half"]), clean_ci, ref, stage3, outdir)
        except Exception as exc:
            rec = {"id": src["id"], "group": src.get("group"), "cell_id": int(src["cell_id"]),
                   "half": float(src["half"]), "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
            logger.exception("route cell %s failed", src["id"])
        (cells_dir / f"{src['id']}.json").write_text(json.dumps(rec, indent=2, default=S2._json_default))


if __name__ == "__main__":
    main()
