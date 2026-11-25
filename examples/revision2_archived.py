#!/usr/bin/env python3
"""Archived-result checks for the revision-2 report. No new sampling.

Writes results/revision2/archived.json. The 0.26 residual diagnostic is
optional (--residual) because it reads the GW170817 frames.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import REPO_ROOT, resolve_under_repo

PARAMS = ["chirp_mass", "mass_ratio", "theta_jn", "luminosity_distance", "lambda_1", "lambda_2",
          "lambda_tilde", "geocent_time"]
GROUP_NAME = {"damaged": "damaged", "broken": "broken", "fine": "unaffected"}


def _load_cells(folder: Path):
    return [json.loads(p.read_text()) for p in sorted((folder / "cells").glob("*.json"))]


def _q(xs, p):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


def section_018(stage3: Path) -> dict:
    """The median printed by gated_is_stage3.report for damaged ±0.1 s."""
    rows = []
    for r in _load_cells(stage3):
        if r.get("group") != "damaged" or abs(float(r.get("half", -1)) - 0.1) > 1e-9:
            continue
        if "gated_is" not in r or "js_gated_vs_clean" not in r:
            continue
        js = r["js_gated_vs_clean"].get("luminosity_distance")
        g = r["gated_is"]
        rows.append({"id": r["id"], "trusted": bool(g["trusted"]), "passes": bool(g["recovers"]),
                     "js": None if js is None else float(js), "n_eff": float(g["n_eff"])})
    trusted = [r for r in rows if r["trusted"] and r["js"] is not None]
    passes = [r for r in trusted if r["passes"]]
    js_t = np.array([r["js"] for r in trusted], float)
    js_p = np.array([r["js"] for r in passes], float)
    return {
        "source": "examples/gated_is_stage3.py report(): damaged, half==0.1, gated_is.trusted, "
                  "js_gated_vs_clean['luminosity_distance'] against the pre-registered 10^4 clean reference. "
                  "Printed with np.median(...):.3f.",
        "n_cells": len(rows),
        "n_trusted": len(trusted),
        "n_trusted_and_pass": len(passes),
        "trusted_failures": [r for r in trusted if not r["passes"]],
        "median_trusted": float(np.median(js_t)),
        "median_trusted_printed": f"{np.median(js_t):.3f}",
        "max_trusted": float(np.max(js_t)),
        "median_passes_only": float(np.median(js_p)),
        "median_passes_only_printed": f"{np.median(js_p):.3f}",
        "subset_used": "all trusted (33), not the 32 passes",
    }


def section_table(summary_path: Path) -> dict:
    S = json.loads(summary_path.read_text())
    floor = {k: float(v) for k, v in S["js_floor"].items()}
    rows = []
    for j in S["js"]:
        if j["method"] == "GIS" and j.get("trusted"):
            kind = "gated trusted"
        elif j["method"] == "OIS" and j.get("trusted"):
            kind = "ordinary trusted"
        elif j["method"] == "NET":
            kind = "network"
        else:
            continue
        if j["group"] not in GROUP_NAME:
            continue
        rows.append(j | {"kind": kind, "group_name": GROUP_NAME[j["group"]]})
    table = []
    damaged_vs_floor = {}
    for group in ("damaged", "broken", "unaffected"):
        for kind in ("gated trusted", "ordinary trusted", "network"):
            sub = [r for r in rows if r["group_name"] == group and r["kind"] == kind]
            if not sub:
                continue
            cell = {"group": group, "kind": kind, "n": len(sub)}
            for p in PARAMS:
                vals = [r["js"].get(p) for r in sub]
                cell[p] = {"median": _q(vals, 50), "p90": _q(vals, 90)}
            mx = [r.get("js_max_all") for r in sub]
            cell["max_over_parameters"] = {"median": _q(mx, 50), "p90": _q(mx, 90)}
            table.append(cell)
            if group == "damaged" and kind == "gated trusted":
                for p in PARAMS:
                    med = cell[p]["median"]
                    fl = floor.get(p)
                    damaged_vs_floor[p] = {
                        "median": med, "floor": fl,
                        "at_or_below_floor": bool(med <= fl) if fl is not None and np.isfinite(med) else None,
                    }
    return {"floor_pooled_trusted_gated": floor, "rows": table,
            "damaged_gated_trusted_vs_floor": damaged_vs_floor,
            "all_shown_parameters_at_or_below_floor": all(
                v["at_or_below_floor"] for v in damaged_vs_floor.values())}


def section_permutation(stage3: Path, real_csv: Path, n_perm: int, seed: int) -> dict:
    """Glitch-clustered sign flip of the GIS-trusted-and-pass vs OIS-pass McNemar comparison.

    Within each glitch, one fair coin swaps the two methods' outcomes on every cell of
    that glitch. Cells from different glitches are not mixed. Statistic is
    (GIS only) - (OIS only). Two-sided p = (1 + #{|S*| >= |S|}) / (n_perm + 1).
    """
    import revision_analysis as ra

    cfg = json.loads((stage3 / "config.json").read_text())
    gmap, gcol = ra.glitch_map(real_csv)
    rows = ra.analyses([r for r in _load_cells(stage3) if r.get("injection_match", True)], [], [], float(cfg.get("num_samples", 10000)))
    out = []
    rng = np.random.default_rng(seed)
    for pad in (0.1, 0.2):
        gis, ois = {}, {}
        for r in rows:
            if r["group"] != "damaged" or abs(float(r["pad"]) - pad) > 1e-9:
                continue
            if r["method"] == "GIS":
                gis[r["cell"]] = bool(r["passes"] and r.get("trusted"))
            elif r["method"] == "OIS":
                ois[r["cell"]] = bool(r["passes"])
        cells = sorted(set(gis) & set(ois))
        by = defaultdict(list)
        for c in cells:
            by[gmap.get(c, str(c))].append((gis[c], ois[c]))
        clusters = [by[k] for k in sorted(by, key=str)]

        def stat(swapped_flags):
            a = b = 0
            for flags, pairs in zip(swapped_flags, clusters):
                for (g, o) in pairs:
                    if flags:
                        g, o = o, g
                    a += int(g and not o)
                    b += int(o and not g)
            return a - b

        obs_flags = [False] * len(clusters)
        obs = stat(obs_flags)
        # one bit per cluster
        bits = rng.random((n_perm, len(clusters))) < 0.5
        n_ge = 0
        for i in range(n_perm):
            if abs(stat(bits[i])) >= abs(obs):
                n_ge += 1
        a_only = sum(gis[c] and not ois[c] for c in cells)
        b_only = sum(ois[c] and not gis[c] for c in cells)
        out.append({
            "group": "damaged", "pad": pad, "n_cells": len(cells), "n_glitches": len(clusters),
            "cluster_column": gcol, "gis_only": a_only, "ois_only": b_only,
            "gis_total": sum(gis[c] for c in cells), "ois_total": sum(ois[c] for c in cells),
            "statistic": obs, "n_perm": n_perm, "n_ge": n_ge,
            "p": (1 + n_ge) / (n_perm + 1),
            "mcnemar_exact": ra.mcnemar_exact(a_only, b_only),
            "definition": "GIS outcome is trusted and passes; OIS outcome is passes (trust not required), "
                          "matching examples/revision_analysis.py mcnemar_rows.",
        })
    return {"seed": seed, "pads": out}


def _weight_counts(paths_and_cols):
    """paths_and_cols: list of (label, path, column or None)."""
    out = {}
    for label, path, col in paths_and_cols:
        if col is None or not path.is_file():
            continue
        w = pd.read_csv(path, usecols=[col])[col].to_numpy(float)
        key = label
        slot = out.setdefault(key, {"n": 0, "nonfinite": 0, "exact_zero": 0, "files": 0})
        slot["n"] += int(w.size)
        slot["nonfinite"] += int(np.sum(~np.isfinite(w)))
        slot["exact_zero"] += int(np.sum(w == 0.0))
        slot["files"] += 1
    return out


def section_weights(stage3: Path, fine: Path, ungated: Path, clean: Path) -> dict:
    jobs = []
    for p in (stage3).glob("*_samples.csv.gz"):
        jobs.append(("GIS", p, "weight_gated"))
        jobs.append(("OIS", p, "weight_naive"))
    for p in fine.glob("*_samples.csv.gz"):
        jobs.append(("GIS", p, "weight_gated"))
        jobs.append(("OIS", p, "weight_naive"))
    for p in ungated.glob("*_samples.csv.gz"):
        jobs.append(("UIS", p, "weight"))
    stored = _weight_counts(jobs)
    clean_df = pd.read_csv(clean / "samples.csv.gz")
    log_w = (clean_df["log_prior"].to_numpy(float) + clean_df["log_likelihood"].to_numpy(float)
             - clean_df["log_prob"].to_numpy(float) + clean_df["delta_log_prob_target"].to_numpy(float))
    return {
        "stored_weights": stored,
        "note": "GIS, OIS and UIS sample files store the normalised weights, not the log weights. "
                "reweight() maps a non-finite log weight to weight 0. An exact zero can also be underflow "
                "of a finite log weight (exp(log_w - max) == 0). Non-finite entries in the stored weight "
                "column are counted separately.",
        "clean_ref_v2_log_weights": {
            "n": int(log_w.size),
            "nonfinite_log_weight": int(np.sum(~np.isfinite(log_w))),
            "nonfinite_log_likelihood": int(np.sum(~np.isfinite(clean_df["log_likelihood"].to_numpy(float)))),
            "nonfinite_log_prior": int(np.sum(~np.isfinite(clean_df["log_prior"].to_numpy(float)))),
            "nonfinite_log_prob": int(np.sum(~np.isfinite(clean_df["log_prob"].to_numpy(float)))),
            "nonfinite_stored_weight": int(np.sum(~np.isfinite(clean_df["weights"].to_numpy(float)))),
            "exact_zero_stored_weight": int(np.sum(clean_df["weights"].to_numpy(float) == 0.0)),
        },
    }


def section_seeds(stage3: Path, routes: Path) -> dict:
    s3 = {r["id"]: r for r in _load_cells(stage3) if "gated_is" in r}
    redraws = []
    for p in sorted((routes / "cells").glob("*.json")):
        r = json.loads(p.read_text())
        a = (r.get("routes") or {}).get("A") or {}
        old = s3.get(r.get("id"), {})
        g = old.get("gated_is") or {}
        if not g or "n_eff" not in a:
            continue
        ne = float(g["n_eff"])
        trusted0 = bool(g["trusted"])
        trusted1 = bool(a["n_eff"] >= 100.0)
        redraws.append({"id": r["id"], "n_eff_stage3": ne, "trusted_stage3": trusted0,
                        "n_eff_routeA": float(a["n_eff"]), "trusted_routeA": trusted1,
                        "flipped": trusted0 != trusted1, "below_300": ne < 300.0})
    below = [r for r in redraws if r["below_300"]]
    n_below = sum(1 for r in s3.values() if float(r["gated_is"]["n_eff"]) < 300.0)
    return {
        "n_stage3_gis_neff_below_300": n_below,
        "second_draw": "R4 route A is one new network draw of the same gated package (draw() sets no seed). "
                       "No other multi-seed archive of these cells exists.",
        "routeA_cells": redraws,
        "routeA_below_300": len(below),
        "routeA_below_300_flips": sum(r["flipped"] for r in below),
        "flips": [r for r in below if r["flipped"]],
    }


def section_residual() -> dict:
    """Whitened residual of package_fd versus the archived event package, and versus DINGO.

    Equation (1) is package_fd. The archived package is whatever was stored in the
    event dataset. data_to_domain is DINGO's own transform of the same reloaded strain,
    including domain.update_data (zero below f_min, truncate above f_max).
    """
    from adapt.dingo_bns_demo import discover_assets, load_bns_checkpoint, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from adapt.packaging import package_fd
    from dingo.gw.data.data_preparation import data_to_domain
    from dingo.gw.domains import build_domain_from_model_metadata
    from leakage_check import whitened_norm

    assets = discover_assets()
    event = load_event_dataset(assets)
    settings = dict(event.settings)
    f_min = float(settings.get("f_min", 23.0))
    f_max = float(settings.get("f_max", 1535.3046875))
    meta = load_bns_checkpoint(Path(assets["baseline_ckpt"]))["metadata"]
    domain = build_domain_from_model_metadata(meta, base=True)
    window = {"type": "tukey", "roll_off": float(settings.get("roll_off", 0.4)),
              "T": float(settings.get("T", 128.0)), "f_s": float(settings.get("f_s", 4096.0))}
    per = {}
    for det in ("H1", "L1", "V1"):
        td, fs, tb = load_full_event_td(assets, settings, det)
        archived = np.asarray(event.data["waveform"][det], dtype=np.complex128)
        asd = np.asarray(event.data["asds"][det], dtype=np.float64)
        ours = package_fd(td, fs, roll_off=window["roll_off"], f_max=f_max, n_freq=archived.size, time_buffer=tb)
        raw = {"strain": {det: np.asarray(td, dtype=np.float64)},
               "psd": {det: np.ones(td.size // 2 + 1, dtype=np.float64)}}
        ding = np.asarray(data_to_domain(raw, {"f_s": fs, "time_buffer": tb}, domain, window=window)["waveform"][det],
                          dtype=np.complex128)
        df = fs / float(td.size)
        n = min(ours.size, archived.size, asd.size)
        m = min(n, ding.size)
        scale = _complex_scale(ours[:n], archived[:n], asd[:n], df, f_min, f_max)
        per[det] = {
            "n_td": int(td.size),
            "n_archived": int(archived.size),
            "n_data_to_domain": int(ding.size),
            "rho_package_fd_vs_archive": whitened_norm(ours[:n] - archived[:n], asd[:n], df, f_min, f_max),
            "rho_archive": whitened_norm(archived[:n], asd[:n], df, f_min, f_max),
            "rho_data_to_domain_vs_archive": whitened_norm(ding[:m] - archived[:m], asd[:m], df, f_min, f_max),
            "rho_package_fd_vs_data_to_domain": whitened_norm(ours[:m] - ding[:m], asd[:m], df, f_min, f_max),
            "rel_l2_inband_package_fd": _rel_l2(ours[:n], archived[:n], df, f_min, f_max),
            "whitened_complex_scale_package_fd_over_archive": {"real": float(scale.real), "imag": float(scale.imag),
                                                                "abs": float(abs(scale))},
        }
    return {"f_min": f_min, "f_max": f_max, "domain_f_min": float(domain.f_min), "domain_f_max": float(domain.f_max),
            "detectors": per}


def _band(n, df, f_lo, f_hi, asd):
    freqs = np.arange(n) * float(df)
    a = np.asarray(asd, dtype=np.float64)[:n]
    return (freqs >= f_lo) & (freqs <= f_hi) & (a > 0.0) & (a < 1e-6)


def _rel_l2(a, b, df, f_lo, f_hi):
    n = min(a.size, b.size)
    # unweighted, analysis band only (placeholder ASD is not available here)
    freqs = np.arange(n) * float(df)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.linalg.norm((a[:n] - b[:n])[band]) / np.linalg.norm(b[:n][band]))


def _complex_scale(ours, archived, asd, df, f_lo, f_hi):
    """Least-squares scale ours ≈ scale * archived, whitened in band. 1 means no scale error."""
    n = min(ours.size, archived.size, asd.size)
    band = _band(n, df, f_lo, f_hi, asd)
    w = 1.0 / np.asarray(asd[:n], float)[band]
    u = np.asarray(ours[:n])[band] * w
    v = np.asarray(archived[:n])[band] * w
    return complex(np.vdot(v, u) / np.vdot(v, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", type=resolve_under_repo, default="results/gated_is_stage3")
    ap.add_argument("--fine", type=resolve_under_repo, default="results/revision/stage3_fine")
    ap.add_argument("--ungated", type=resolve_under_repo, default="results/revision/ungated_is")
    ap.add_argument("--summary", type=resolve_under_repo, default="results/revision/revision_summary.json")
    ap.add_argument("--real-csv", type=resolve_under_repo, default="results/stress_real_glitches_v3_final/results.csv")
    ap.add_argument("--routes", type=resolve_under_repo, default="results/revision/routes")
    ap.add_argument("--clean", type=resolve_under_repo, default="results/revision/clean_ref_v2")
    ap.add_argument("--out", type=resolve_under_repo, default="results/revision2/archived.json")
    ap.add_argument("--perms", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--residual", action="store_true")
    args = ap.parse_args()
    out = {
        "median_0018": section_018(Path(args.stage3)),
        "table_iv": section_table(Path(args.summary)),
        "permutation": section_permutation(Path(args.stage3), Path(args.real_csv), args.perms, args.seed),
        "weights": section_weights(Path(args.stage3), Path(args.fine), Path(args.ungated), Path(args.clean)),
        "seeds": section_seeds(Path(args.stage3), Path(args.routes)),
    }
    if args.residual:
        out["residual"] = section_residual()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
