#!/usr/bin/env python3
"""Numbers for the referee revision, written straight into LaTeX macros.

Run from the REPOSITORY root (read-only on results/):

    python paper/scripts/paper_numbers.py            # writes paper/generated/numbers.tex

It reads
  results/gated_is_stage3/cells/*.json             per-analysis JS, n_eff, runtime, gates
  results/gated_is_stage3/<id>_samples.csv.gz      weighted proposal samples (for the JS floor)
  results/gated_is_stage3/clean_reference.json     clean efficiency (for the JS floor)
  results/gated_is_stage3/config.json              proposal and clean sample sizes
  results/gated_is_stage2/cells/*.json             eps sensitivity of n_eff
and prints a human-readable summary as well as writing the macros.
Nothing here is fitted or tuned; the JS divergence is the same 64-bin estimator used
by the analysis drivers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FS = 4096.0
# parameters shown in the table, in order, with their LaTeX names
SHOW = [
    ("chirp_mass", r"$\mathcal{M}$"),
    ("mass_ratio", r"$q$"),
    ("theta_jn", r"$\theta_{JN}$"),
    ("luminosity_distance", r"$d_L$"),
    ("lambda_1", r"$\Lambda_1$"),
    ("lambda_2", r"$\Lambda_2$"),
    ("geocent_time", r"$t_c$"),
]


def js_divergence_1d(x, y, *, wx=None, wy=None, bins=64):
    """Copy of examples/official_control.js_divergence_1d (64-bin histogram JS, nats)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = np.isfinite(x), np.isfinite(y)
    x, y = x[mx], y[my]
    if wx is not None:
        wx = np.maximum(np.asarray(wx, dtype=np.float64)[mx], 0.0)
        wx = wx / (wx.sum() + 1e-30)
    if wy is not None:
        wy = np.maximum(np.asarray(wy, dtype=np.float64)[my], 0.0)
        wy = wy / (wy.sum() + 1e-30)
    lo, hi = float(min(np.min(x), np.min(y))), float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges, weights=wx, density=True)
    py, _ = np.histogram(y, bins=edges, weights=wy, density=True)
    px, py = px + 1e-12, py + 1e-12
    px, py = px / px.sum(), py / py.sum()
    m = 0.5 * (px + py)
    return 0.5 * (float(np.sum(px * np.log(px / m))) + float(np.sum(py * np.log(py / m))))


try:  # prefer the analysis code itself when it imports cleanly
    import sys

    sys.path[:0] = ["src", "examples"]
    from official_control import js_divergence_1d as _js_repo  # type: ignore

    _rng = np.random.default_rng(0)
    _a, _b = _rng.normal(size=500), _rng.normal(0.3, 1, size=700)
    assert abs(_js_repo(_a, _b) - js_divergence_1d(_a, _b)) < 1e-12
    js_divergence_1d = _js_repo  # noqa: F811
    JS_SOURCE = "examples/official_control.js_divergence_1d"
except Exception:  # pragma: no cover
    JS_SOURCE = "local copy of examples/official_control.js_divergence_1d"


def q(xs, p):
    xs = sorted(x for x in xs if x is not None and np.isfinite(x))
    return float(np.percentile(xs, p)) if xs else float("nan")


def fmt(x, nd=3):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def js_floor(samples_csv: Path, params, n_eff_cell: float, n_eff_clean: float, rng, repeats=5):
    """JS between two independent resamples of the cell's OWN gated posterior, of sizes
    n_eff(cell) and n_eff(clean): the value a perfect match would still show."""
    import pandas as pd

    df = pd.read_csv(samples_csv)
    w = df["weight_gated"].to_numpy(float)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    n1, n2 = max(2, int(round(n_eff_cell))), max(2, int(round(n_eff_clean)))
    out = {p: [] for p in params}
    for _ in range(repeats):
        i1 = rng.choice(len(df), size=n1, replace=True, p=w)
        i2 = rng.choice(len(df), size=n2, replace=True, p=w)
        for p in params:
            if p in df.columns:
                out[p].append(js_divergence_1d(df[p].to_numpy(float)[i1], df[p].to_numpy(float)[i2]))
    return {p: float(np.median(v)) for p, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", type=Path, default=Path("results/gated_is_stage3"))
    ap.add_argument("--stage2", type=Path, default=Path("results/gated_is_stage2"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated/numbers.tex"))
    ap.add_argument("--no-floor", action="store_true", help="skip the resampling floor (fast)")
    a = ap.parse_args()

    rows = [json.loads(f.read_text()) for f in sorted((a.stage3 / "cells").glob("*.json"))]
    rows = [r for r in rows if "gated_is" in r and "js_gated_vs_clean" in r]
    if len(rows) != 110:
        print(f"WARNING: expected 110 analysed cells, found {len(rows)}")
    tr = [r for r in rows if r["gated_is"]["trusted"]]
    un = [r for r in rows if not r["gated_is"]["trusted"]]
    params_all = sorted(set().union(*[r["js_gated_vs_clean"].keys() for r in rows]))
    M = {}

    # ---------------- multi-parameter JS ----------------
    def col(sub, key, p):
        return [r.get(key, {}).get(p) for r in sub]

    cfg = json.loads((a.stage3 / "config.json").read_text()) if (a.stage3 / "config.json").exists() else {}
    clean = json.loads((a.stage3 / "clean_reference.json").read_text())
    n_clean = float(cfg.get("clean_samples", 10_000)) * float(clean["clean_eff"])
    floors = {}
    if not a.no_floor:
        rng = np.random.default_rng(12345)
        per = []
        for r in tr:
            f = a.stage3 / f"{r['id']}_samples.csv.gz"
            if f.exists():
                per.append(js_floor(f, params_all, r["gated_is"]["n_eff"], n_clean, rng))
        floors = {p: float(np.median([d[p] for d in per if p in d])) for p in params_all} if per else {}
        M["MPfloorCells"] = str(len(per))

    lines = []
    for p, tex in SHOW:
        lines.append(" & ".join([
            tex,
            fmt(q(col(tr, "js_gated_vs_clean", p), 50)), fmt(q(col(tr, "js_gated_vs_clean", p), 90)),
            fmt(q(col(un, "js_gated_vs_clean", p), 50)), fmt(q(col(un, "js_gated_vs_clean", p), 90)),
            fmt(q(col(rows, "js_naive_vs_clean", p), 50)), fmt(q(col(rows, "js_naive_vs_clean", p), 90)),
            fmt(floors.get(p)),
        ]) + r" \\")
    # max over all compared parameters, per analysis
    def cellmax(sub, key):
        return [max(v for v in r.get(key, {}).values() if v is not None and np.isfinite(v)) for r in sub if r.get(key)]
    fl_max = max(floors.values()) if floors else float("nan")
    lines.append(r"\midrule")
    lines.append(" & ".join([
        r"max over all", fmt(q(cellmax(tr, "js_gated_vs_clean"), 50)), fmt(q(cellmax(tr, "js_gated_vs_clean"), 90)),
        fmt(q(cellmax(un, "js_gated_vs_clean"), 50)), fmt(q(cellmax(un, "js_gated_vs_clean"), 90)),
        fmt(q(cellmax(rows, "js_naive_vs_clean"), 50)), fmt(q(cellmax(rows, "js_naive_vs_clean"), 90)),
        fmt(fl_max),
    ]) + r" \\")
    M["MProws"] = "\n".join(lines)

    tr_med = {p: q(col(tr, "js_gated_vs_clean", p), 50) for p in params_all}
    tr_p90 = {p: q(col(tr, "js_gated_vs_clean", p), 90) for p in params_all}
    worst90 = max(tr_p90, key=lambda p: tr_p90[p])
    M["MPnparams"] = str(len(params_all))
    M["MPtrustedMaxMedian"] = fmt(max(tr_med.values()))
    M["MPtrustedMaxMedianParam"] = dict(SHOW).get(max(tr_med, key=tr_med.get), max(tr_med, key=tr_med.get).replace("_", r"\_"))
    M["MPtrustedMaxPninety"] = fmt(tr_p90[worst90])
    M["MPtrustedMaxPninetyParam"] = dict(SHOW).get(worst90, worst90.replace("_", r"\_"))
    tmax = cellmax(tr, "js_gated_vs_clean")
    M["MPtrustedAnyOverTenth"] = str(sum(1 for v in tmax if v > 0.1))
    M["MPtrustedN"] = str(len(tr))
    M["MPuntrustedN"] = str(len(un))
    M["MPfloorMax"] = fmt(fl_max)
    M["MPnClean"] = f"{n_clean:.0f}"
    M["MPtrustedCellMaxMedian"] = fmt(q(tmax, 50))
    M["MPuntrustedCellMaxMedian"] = fmt(q(cellmax(un, "js_gated_vs_clean"), 50))

    # ---------------- runtime ----------------
    el = [r["elapsed_s"] for r in rows if "elapsed_s" in r]
    slow = max(rows, key=lambda r: r.get("elapsed_s", -1))
    gl = [r.get("gate_total_s", float("nan")) for r in rows]
    M["RTmedianMin"] = f"{q(el, 50) / 60:.1f}"
    M["RTpninetyMin"] = f"{q(el, 90) / 60:.1f}"
    M["RTpninetyfiveMin"] = f"{q(el, 95) / 60:.1f}"
    M["RTmaxH"] = f"{max(el) / 3600:.1f}"
    M["RTslowId"] = slow["id"].replace("_", r"\_")
    M["RTslowGate"] = f"{slow.get('gate_total_s', float('nan')):.2f}"
    M["RTslowM"] = f"{int(round(slow.get('gate_total_s', 0) * FS)):,}".replace(",", r"{,}")
    M["RTslowGateRank"] = str(1 + sum(1 for g in gl if g > slow.get("gate_total_s", 0)))
    M["RTlongestGate"] = f"{np.nanmax(gl):.2f}"
    M["RTmedianGate"] = f"{np.nanmedian(gl):.2f}"
    M["RTspearman"] = f"{spearman([r.get('elapsed_s', np.nan) for r in rows], gl):.2f}"
    M["RTnOverHour"] = str(sum(1 for e in el if e > 3600))

    # ---------------- eps sensitivity of n_eff (Stage 2) ----------------
    s2 = [json.loads(f.read_text()) for f in sorted((a.stage2 / "cells").glob("*.json"))] if (a.stage2 / "cells").exists() else []
    rel, flips, ncell = [], 0, 0
    for r in s2:
        es = r.get("eps_sensitivity_A")
        A = r.get("gated_is_A")
        if not es or not A or not A.get("efficiency"):
            continue
        N = A["n_eff"] / A["efficiency"]
        ref = es.get("0.001", {}).get("efficiency")
        if not ref:
            continue
        ncell += 1
        trust_ref = ref * N >= 100
        for k, v in es.items():
            if k == "0.001" or v.get("efficiency") is None:
                continue
            rel.append(abs(v["efficiency"] - ref) / ref)
            flips += int((v["efficiency"] * N >= 100) != trust_ref)
    M["EPScells"] = str(ncell)
    M["EPSmaxRelPct"] = f"{100 * max(rel):.1f}" if rel else "--"
    M["EPSmedRelPct"] = f"{100 * float(np.median(rel)):.2f}" if rel else "--"
    M["EPSflips"] = str(flips)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    body = ["% generated by paper/scripts/paper_numbers.py -- do not edit by hand", f"% JS estimator: {JS_SOURCE}"]
    for k, v in M.items():
        body.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    a.out.write_text("\n".join(body) + "\n")

    print(f"JS estimator: {JS_SOURCE}")
    print(f"analyses {len(rows)} (trusted {len(tr)}, untrusted {len(un)}); compared parameters: {len(params_all)}")
    print("param | trusted med p90 | untrusted med p90 | ordinary-IS med p90 | floor")
    for ln in lines:
        print("  ", ln)
    print(f"trusted analyses with any parameter JS > 0.1: {M['MPtrustedAnyOverTenth']} of {len(tr)}")
    print(f"runtime: median {M['RTmedianMin']} min, p90 {M['RTpninetyMin']} min, p95 {M['RTpninetyfiveMin']} min, max {M['RTmaxH']} h "
          f"({slow['id']}, gate {M['RTslowGate']} s, rank {M['RTslowGateRank']} by gate length); "
          f"longest gate {M['RTlongestGate']} s, median gate {M['RTmedianGate']} s; Spearman(runtime, gate) {M['RTspearman']}; "
          f"{M['RTnOverHour']} analyses over 1 h")
    print(f"eps: {ncell} Stage 2 cells; n_eff change vs eps=1e-3: median {M['EPSmedRelPct']}%, max {M['EPSmaxRelPct']}%; trust flips {flips}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
