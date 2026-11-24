#!/usr/bin/env python3
"""Stage 3: gate-aware importance sampling on real Gravity Spy glitches.

Cells come from the Phase C real panel (results/stress_real_glitches_v3_final):
  * every DAMAGED cell (poisoned posterior fails the recovery criterion), and
  * every UNDAMAGED cell that the localiser gate broke in Phase C.
Each cell is rebuilt exactly as in Phase C (same cached excerpt, same placement;
the injected whitened SNR must match the archived value to 1e-6 or the cell is
skipped), gated with the whitened-excess localiser (8 sigma) at pad 0.1 s and
0.2 s, and analysed four ways:

  network only   raw samples on the gated package
  naive IS       DINGO IS with the ordinary likelihood on the gated package
  gated IS       the same proposal, reweighted with the gated likelihood
  reference      DINGO IS on clean GW170817 (the target to recover)

Reporting rules are fixed in docs/STAGE3_PLAN.md (committed before this runs).
Outputs: results/gated_is_stage3/{cells/*.json, cells.csv, STAGE3_REPORT.md}.
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

from _repo import REPO_ROOT, resolve_under_repo

logger = logging.getLogger("gated_is_stage3")
TRUST_N_EFF = 100.0
REAL = REPO_ROOT / "results" / "stress_real_glitches_v3_final"


def _recovers(lo, med, hi, clean) -> pd.Series:
    return (med >= 20) & (med <= 50) & ((med - clean["med"]).abs() <= 10) & (hi >= clean["lo"]) & (lo <= clean["hi"])


def select_cells(real_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(real_dir / "results.csv")
    clean = json.loads((real_dir / "clean_reference.json").read_text())
    clean = clean.get("d_L", clean.get("luminosity_distance", clean))
    r = r[r["error"].fillna("") == ""].copy()
    for col in ("poison_collapsed", "gated_recovers", "detector_fired", "is_noise_control"):
        r[col] = r[col].astype(str).str.lower().isin(["true", "1"])
    r = r[(~r["is_noise_control"]) & (r["injected_rho_w"].astype(float) > 1.0)].copy()
    r["damaged"] = ~_recovers(r["poison_lo"], r["poison_med"], r["poison_hi"], clean)
    r["broken"] = (~r["damaged"]) & (~r["gated_recovers"])
    sel = r[r["damaged"] | r["broken"]].copy()
    sel["group"] = np.where(sel["damaged"], "damaged", "broken")
    return sel.sort_values(["group", "cell_id"], ascending=[False, True])


class Injector:
    """Rebuilds a Phase C real-panel cell bit-for-bit from the cached excerpt."""

    def __init__(self, c, real_dir: Path):
        from adapt.dingo_bns_demo import load_bns_checkpoint
        from adapt.glitch_excision import analysis_crop_bounds
        from adapt.stft_context import inband_rms
        from dingo.gw.domains import build_domain_from_model_metadata
        from stress_real_glitches import Transplanter

        meta = load_bns_checkpoint(Path(c.assets["baseline_ckpt"]))["metadata"]
        base = build_domain_from_model_metadata(meta, base=True)
        self.c = c
        self.f_min = float(c.settings.get("f_min", 23.0))
        _, cs0, ce0 = analysis_crop_bounds(duration=c.T, time_buffer=c.tb, sample_rate=c.fs)
        self.tp = Transplanter(
            asd_h1=c.asds["H1"], delta_f=float(base.delta_f), noise_std=float(base.noise_std),
            sample_rate=c.fs, f_min=self.f_min, f_max=c.f_max, clean_crop_h1=c.td["H1"][cs0:ce0],
            rms_inband_h1=float(inband_rms(c.td["H1"], c.fs, f_min=self.f_min, f_max=c.f_max)),
        )
        self.cat = pd.read_csv(real_dir / "glitch_catalogue.csv")

    def __call__(self, row):
        from adapt import gravity_spy_io as gs
        from stress_real_glitches import inject_excerpt_into_event

        cat = self.cat.iloc[(self.cat["event_time"] - float(row["gs_event_time"])).abs().argmin()]
        t = float(cat["event_time"])
        cache = REPO_ROOT / "data" / "gravity_spy" / "excerpts" / f"H1_{t:.4f}_{int(self.c.fs)}.npy"
        if cache.is_file():
            raw = np.load(cache)
        else:
            raw = gs.fetch_h1_excerpt(t, sample_rate=self.c.fs)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, np.asarray(raw, dtype=np.float64))
        exc = gs.extract_glitch_excerpt(raw, sample_rate=self.c.fs, duration_s=float(cat["duration"]),
                                        f_min=self.f_min, f_max=self.c.f_max)
        poison, _td_stft, meta = inject_excerpt_into_event(
            self.c.event, self.c.assets, excerpt=exc, mode="native", severity=float("nan"),
            t_rel=float(row["t_rel"]), td_clean_full=self.c.td, transplanter=self.tp,
            f_max=self.c.f_max, roll_off=self.c.roll_off,
        )
        return poison, meta


def run_cell(c, inj: Injector, row, half: float, clean_ci: dict, clean_res, outdir: Path) -> dict:
    import gated_is_stage2 as S2
    from adapt.gated_likelihood import build_ufd_likelihood, theta_frame
    from adapt.glitch_excision import rebuild_event_from_gated_td
    from adapt.localize import excess_gates
    from official_control import INFERENCE_PARAMS
    from stress_gw170817 import _gated_recovers

    t0 = time.time()
    rec = {
        "id": f"{row['group']}_{int(row['cell_id']):03d}_h{half:g}", "group": row["group"],
        "cell_id": int(row["cell_id"]), "half": half, "label": row["gs_label"], "snr_bin": row["gs_snr_bin"],
        "rho_w_archived": float(row["injected_rho_w"]), "t_rel": float(row["t_rel"]),
        "phaseC_localiser_recovers": bool(row["gated_recovers"]),
        "phaseC_collapsed": bool(row["poison_collapsed"]),
    }
    poison, meta = inj(row)
    rec["rho_w_rebuilt"] = float(meta["injected_rho_w"])
    rel = abs(rec["rho_w_rebuilt"] - rec["rho_w_archived"]) / max(rec["rho_w_archived"], 1e-12)
    rec["injection_match"] = bool(rel < 1e-6)
    if not rec["injection_match"]:
        rec["error"] = f"injection mismatch: rebuilt rho_w {rec['rho_w_rebuilt']:.6g} vs archived {rec['rho_w_archived']:.6g}"
        return rec
    wins, zmax = excess_gates({"H1": meta["td_full"]["H1"]}, {"H1": c.asds["H1"]}, c.fs,
                              f_max=c.f_max, pad_s=half, sigma=8.0, detectors=["H1"])
    rec["zmax"] = zmax.get("H1")
    rec["gates"] = [[g.t_start - c.t_trig, g.t_end - c.t_trig] for g in wins]
    rec["gate_total_s"] = float(sum(g.t_end - g.t_start for g in wins))
    if not wins:
        rec["no_gate"] = True
        rec["elapsed_s"] = time.time() - t0
        return rec
    gated = rebuild_event_from_gated_td(
        poison, td_by_det={"H1": meta["td_full"]["H1"]}, gates=wins, sample_rate=c.fs, roll_off=c.roll_off,
        f_max=c.f_max, original_asds=c.asds, fill="zero", gate_band=(20.0, 23.0),
    )
    resA, rawA = S2.draw(c, gated.data, c.args.num_samples)
    rec["network_only"] = S2._ci(rawA["luminosity_distance"].to_numpy(float))
    rec["network_only"]["recovers"] = _gated_recovers(rec["network_only"], clean_ci)
    wN = resA.samples["weights"].to_numpy(float) if "weights" in resA.samples.columns else None
    rec["naive_is"] = {
        "efficiency": float(resA.sample_efficiency) if resA.sample_efficiency is not None else None,
        "d_L": S2._ci(resA.samples["luminosity_distance"].to_numpy(float), wN),
    }
    rec["naive_is"]["recovers"] = _gated_recovers(rec["naive_is"]["d_L"], clean_ci)
    base = build_ufd_likelihood(resA)
    ops = S2.gate_ops(c, base, wins, [c.args.eps])[c.args.eps]
    payload = S2._Payload(base, S2.unit_ops(c, base, ["H1"]), {"H1": ops["H1"].idx})
    # chunked so that 1e5-sample runs with long gates fit in memory
    theta = theta_frame(resA)
    chunk = int(getattr(c.args, "chunk", 20_000))
    L0_parts, corr_parts = [], []
    for i0 in range(0, len(theta), chunk):
        L0_i, u_i = S2.evaluate(payload, theta.iloc[i0:i0 + chunk], int(c.args.num_processes))
        L0_parts.append(L0_i)
        corr_parts.append(S2.corrections(ops, u_i, payload.windows))
        del u_i
    L0, corr = np.concatenate(L0_parts), np.concatenate(corr_parts)
    G = S2.summarise_route(resA.samples, L0 + corr, "gated")
    rec["gated_is"] = {k: v for k, v in G.items() if not k.startswith("_")}
    rec["gated_is"]["recovers"] = bool(_gated_recovers(G["d_L"], clean_ci))
    rec["gated_is"]["trusted"] = bool(G["n_eff"] >= TRUST_N_EFF)
    rec["gated_is"]["recovers_trusted"] = bool(rec["gated_is"]["recovers"] and rec["gated_is"]["trusted"])
    wC = clean_res.samples["weights"].to_numpy(float)
    rec["js_gated_vs_clean"] = S2._js(resA.samples, G["_weights"], clean_res.samples, wC, INFERENCE_PARAMS)
    if wN is not None:
        rec["js_naive_vs_clean"] = S2._js(resA.samples, wN, clean_res.samples, wC, INFERENCE_PARAMS)
    keep = [p for p in INFERENCE_PARAMS if p in resA.samples.columns]
    out = resA.samples[keep].copy()
    out["weight_gated"] = G["_weights"]
    if wN is not None:
        out["weight_naive"] = wN
    out.to_csv(outdir / f"{rec['id']}_samples.csv.gz", index=False)
    rec["elapsed_s"] = time.time() - t0
    return rec


def untrusted_targets(folder: Path, pads) -> set:
    """(group, cell_id, pad) of every analysed cell in ``folder`` whose gated IS had n_eff < 100."""
    out = set()
    pads = {round(float(p), 6) for p in pads}
    for f in sorted((Path(folder) / "cells").glob("*.json")):
        r = json.loads(f.read_text())
        if "gated_is" in r and not r["gated_is"].get("trusted", True) and round(float(r.get("half", -1)), 6) in pads:
            out.add((r["group"], int(r["cell_id"]), float(r["half"])))
    return out


def _wilson(k, n, z=1.959964):
    if n == 0:
        return "0/0"
    p = k / n
    d = 1 + z * z / n
    cc = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} ({100 * p:.0f}%; {100 * max(cc - h, 0):.0f}–{100 * min(cc + h, 1):.0f}%)"


def report(outdir: Path, rows: list, meta: dict) -> str:
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    L = ["# Stage 3 report: gated importance sampling on real glitches", "",
         f"Clean reference (DINGO IS): d_L median {meta['clean_ci']['med']:.1f}, 90% [{meta['clean_ci']['lo']:.1f}, {meta['clean_ci']['hi']:.1f}] Mpc; "
         f"efficiency {meta['clean_eff']:.4f}.",
         f"A cell counts as recovered by gated IS only if n_eff >= {TRUST_N_EFF:.0f} AND the d_L criterion passes (docs/STAGE3_PLAN.md).", ""]
    L += ["| group | gate pad | cells | Phase C localiser recovers | network only | naive IS | gated IS (trusted) | gated IS untrusted (n_eff<100) | no gate | median eff |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for grp in ("damaged", "broken"):
        for half in sorted({r["half"] for r in ok}):
            sub = [r for r in ok if r["group"] == grp and r["half"] == half]
            if not sub:
                continue
            ran = [r for r in sub if "gated_is" in r]
            n = len(sub)
            effs = [r["gated_is"]["efficiency"] for r in ran]
            L.append(
                f"| {grp} | ±{half:g} s | {n} | {sum(r['phaseC_localiser_recovers'] for r in sub)}/{n} | "
                f"{sum(r['network_only']['recovers'] for r in ran)}/{n} | {sum(r['naive_is']['recovers'] for r in ran)}/{n} | "
                f"{_wilson(sum(r['gated_is']['recovers_trusted'] for r in ran), n)} | "
                f"{sum(not r['gated_is']['trusted'] for r in ran)} | {sum(bool(r.get('no_gate')) for r in sub)} | "
                f"{np.median(effs) if effs else float('nan'):.4f} |"
            )
    # adaptive rule: per cell, use the pad with the larger n_eff (observable without the reference)
    by_cell = {}
    for r in ok:
        if "gated_is" in r:
            by_cell.setdefault((r["group"], r["cell_id"]), []).append(r)
    L += ["", "Adaptive rule (pre-registered): per cell, take the pad with the larger gated-IS n_eff.", ""]
    for grp in ("damaged", "broken"):
        cells = [v for (g, _), v in by_cell.items() if g == grp and len(v) == 2]
        if cells:
            pick = [max(v, key=lambda r: r["gated_is"]["n_eff"]) for v in cells]
            L.append(f"- {grp}: {_wilson(sum(p['gated_is']['recovers_trusted'] for p in pick), len(pick))} recovered (trusted), "
                     f"{sum(not p['gated_is']['trusted'] for p in pick)} untrusted, over {len(pick)} cells with both pads.")
    # damaged, by morphology and SNR bin, at each pad
    for half in sorted({r["half"] for r in ok}):
        dmg = [r for r in ok if r["group"] == "damaged" and r["half"] == half and "gated_is" in r]
        if not dmg:
            continue
        L += ["", f"Damaged cells, ±{half:g} s, by morphology / Omicron SNR bin (gated IS trusted recoveries; Phase C localiser in brackets):", ""]
        for key in ("label", "snr_bin"):
            groups = sorted({r[key] for r in dmg})
            L.append("- " + "; ".join(
                f"{gname}: {sum(r['gated_is']['recovers_trusted'] for r in dmg if r[key] == gname)}/{sum(1 for r in dmg if r[key] == gname)} "
                f"[{sum(r['phaseC_localiser_recovers'] for r in dmg if r[key] == gname)}]" for gname in groups))
        js = [r["js_gated_vs_clean"].get("luminosity_distance") for r in dmg if r["gated_is"]["trusted"]]
        js = [x for x in js if x is not None and np.isfinite(x)]
        if js:
            L.append(f"- trusted cells: d_L JS vs clean median {np.median(js):.3f}, max {np.max(js):.3f}")
    n_mis = sum(1 for r in rows if r.get("injection_match") is False)
    L += ["", f"Injection reproduction: {sum(1 for r in rows if r.get('injection_match'))} matched, {n_mis} mismatched (skipped)."]
    if errs:
        L += ["", "Errors:"] + [f"- {r['id']}: {r['error'][:200]}" for r in errs]
    text = "\n".join(L)
    (outdir / "STAGE3_REPORT.md").write_text(text + "\n")
    return text


def main(argv=None) -> None:
    import gated_is_stage2 as S2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default="results/gated_is_stage3")
    p.add_argument("--real-dir", type=resolve_under_repo, default=str(REAL))
    p.add_argument("--pads", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="300 samples, 2 cells, pad 0.1 only")
    p.add_argument("--clean-samples", type=int, default=10_000, help="samples for the clean reference")
    p.add_argument("--chunk", type=int, default=20_000, help="likelihood evaluations per chunk")
    p.add_argument("--only-untrusted-from", type=resolve_under_repo, default=None,
                   help="a previous Stage 3 output folder: rerun exactly its untrusted cells (pads in --pads only)")
    args = p.parse_args(argv)
    args.eps_sensitivity = []
    if args.smoke:
        args.num_samples, args.batch_size, args.max_cells, args.pads = 300, 300, 2, [0.1]
        args.clean_samples = 300
    outdir = Path(args.outdir)
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    cells = select_cells(Path(args.real_dir))
    only = None
    if args.only_untrusted_from:
        only = untrusted_targets(Path(args.only_untrusted_from), args.pads)
        logger.info("rerunning %d untrusted (group, cell, pad) combinations", len(only))
        keep = {(g, cid) for g, cid, _ in only}
        cells = cells[[(g, int(cid)) in keep for g, cid in zip(cells["group"], cells["cell_id"])]]
    if args.max_cells:
        cells = cells.groupby("group", sort=False).head(int(args.max_cells))
    logger.info("selected %d damaged and %d broken cells", int((cells["group"] == "damaged").sum()), int((cells["group"] == "broken").sum()))

    c = S2.setup(args)
    inj = Injector(c, Path(args.real_dir))
    clean_res, _ = S2.draw(c, c.event.data, args.clean_samples)
    wC = clean_res.samples["weights"].to_numpy(float)
    clean_ci = S2._ci(clean_res.samples["luminosity_distance"].to_numpy(float), wC)
    meta = {"clean_ci": clean_ci, "clean_eff": float(clean_res.sample_efficiency)}
    (outdir / "clean_reference.json").write_text(json.dumps(meta, indent=2, default=float))

    rows = [json.loads(f.read_text()) for f in sorted((outdir / "cells").glob("*.json"))]
    done = {r["id"] for r in rows}
    plan = [(grp, half) for grp in ("damaged", "broken") for half in args.pads]
    for grp, half in plan:
        for _, row in cells[cells["group"] == grp].iterrows():
            cid = f"{grp}_{int(row['cell_id']):03d}_h{half:g}"
            if cid in done:
                continue
            if only is not None and (grp, int(row["cell_id"]), float(half)) not in only:
                continue
            if (time.time() - t_start) / 3600.0 > args.max_hours:
                logger.warning("time budget reached before %s", cid)
                report(outdir, rows, meta)
                return
            logger.info("=== %s ===", cid)
            try:
                rec = run_cell(c, inj, row, half, clean_ci, clean_res, outdir)
            except Exception as exc:
                rec = {"id": cid, "group": grp, "half": half, "cell_id": int(row["cell_id"]),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                logger.exception("cell %s failed", cid)
            (outdir / "cells" / f"{cid}.json").write_text(json.dumps(rec, indent=2, default=S2._json_default))
            rows.append(rec)
            done.add(cid)
            report(outdir, rows, meta)
    pd.json_normalize([{k: v for k, v in r.items() if k != "traceback"} for r in rows]).to_csv(outdir / "cells.csv", index=False)
    logger.info("\n%s", report(outdir, rows, meta))
    logger.info("done in %.2f h", (time.time() - t_start) / 3600.0)


if __name__ == "__main__":
    main()
