#!/usr/bin/env python3
"""Referee-revision analysis: flag baselines, cluster bootstrap, McNemar, PSIS k-hat,
multi-parameter JS against a larger clean reference, route agreement.

Read-only on everything it is given. Run from the repository root AFTER the Phase 2
runs of docs/REVISION_PLAN.md have finished:

    python examples/revision_analysis.py            # writes results/revision/REVISION_REPORT.md
                                                    #    and results/revision/revision_summary.json
    python examples/revision_analysis.py --smoke    # quick pass: 200 bootstrap draws, no JS floor

Every definition used here is fixed in docs/REVISION_PLAN.md (committed before the runs).
Inputs (all paths overridable):
  results/gated_is_stage3/                 cells/*.json, <id>_samples.csv.gz, config.json, clean_reference.json
  results/revision/stage3_fine/            same layout, group "fine", pad 0.1 only
  results/revision/ungated_is/             cells/NNN.json, NNN_samples.csv.gz  (column "weight")
  results/revision/clean_ref_v2/           samples.csv.gz (column "weights"), meta.json
  results/revision/routes/                 cells/*.json
  results/stress_real_glitches_v3_final/   results.csv  (glitch identity for the cluster bootstrap)
  results/gated_is_stage3b/  (optional)    cells/*.json at 1e5 samples, substituted in a separate table
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TRUST = 100.0
THRESHOLDS = [25, 50, 100, 200, 500]
JS_PARAMS = ["chirp_mass", "mass_ratio", "theta_jn", "luminosity_distance", "lambda_1", "lambda_2",
             "lambda_tilde", "geocent_time"]


# ---------------------------------------------------------------- estimators
def js_divergence_1d(x, y, *, wx=None, wy=None, bins=64):
    """Identical to examples/official_control.js_divergence_1d (64-bin histogram JS, nats)."""
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
    if x.size == 0 or y.size == 0:
        return float("nan")
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


def _gpdfit(ary):
    """Generalised Pareto fit (Zhang & Stephens 2009) with the PSIS prior, as in arviz."""
    prior_bs, prior_k = 3, 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]
    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    real = weights >= 10 * np.finfo(float).eps
    weights, b_ary = weights[real], b_ary[real]
    weights /= weights.sum()
    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * ary).mean()
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post)


def psis_khat(log_w):
    """Pareto-k-hat of a vector of log importance weights (Vehtari et al. 2024)."""
    lw = np.asarray(log_w, float)
    lw = lw[np.isfinite(lw)]
    n = lw.size
    if n < 10:
        return float("nan")
    x = lw - lw.max()
    order = np.argsort(x)
    cutoff_ind = -int(np.ceil(min(0.2 * n, 3 * np.sqrt(n)))) - 1
    xcut = max(x[order[cutoff_ind]], np.log(np.finfo(float).tiny))
    tail = x[x > xcut]
    if tail.size <= 4:
        return float("inf")
    tail = np.sort(np.exp(tail) - np.exp(xcut))
    return _gpdfit(tail)


def lambda_tilde(df):
    mc, q = df["chirp_mass"].to_numpy(float), df["mass_ratio"].to_numpy(float)
    l1, l2 = df["lambda_1"].to_numpy(float), df["lambda_2"].to_numpy(float)
    m1 = mc * (1 + q) ** 0.2 / q ** 0.6
    m2 = q * m1
    return 16.0 / 13.0 * ((m1 + 12 * m2) * m1 ** 4 * l1 + (m2 + 12 * m1) * m2 ** 4 * l2) / (m1 + m2) ** 5


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    from scipy.stats import binom

    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def cluster_ratio_ci(records, num_key, den_key, cluster_key="glitch", B=10000, seed=0):
    """Percentile CI of sum(num)/sum(den) resampling clusters (glitches) with replacement."""
    by = defaultdict(lambda: [0, 0])
    for r in records:
        by[r[cluster_key]][0] += int(r[num_key])
        by[r[cluster_key]][1] += int(r[den_key])
    keys = list(by)
    if not keys:
        return (float("nan"), float("nan"), 0)
    num = np.array([by[k][0] for k in keys], float)
    den = np.array([by[k][1] for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    N, D = num[idx].sum(1), den[idx].sum(1)
    ok = D > 0
    vals = N[ok] / D[ok]
    if vals.size == 0:
        return (float("nan"), float("nan"), len(keys))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(keys))


# ---------------------------------------------------------------- loading
def load_cells(folder: Path):
    if not (folder / "cells").exists():
        return []
    return [json.loads(f.read_text()) for f in sorted((folder / "cells").glob("*.json"))]


def glitch_map(real_csv: Path):
    r = pd.read_csv(real_csv)
    for col in ("gs_event_time", "src", "gs_id", "glitch_id"):
        if col in r.columns:
            return {int(c): str(v) for c, v in zip(r["cell_id"], r[col])}, col
    raise KeyError(f"no glitch identity column in {real_csv}")


def analyses(stage3, fine, ungated, n_prop):
    """One row per (cell, pad, method) with flag and pass; methods GIS, OIS, NET, UIS."""
    U = {int(u["cell_id"]): u for u in ungated if u.get("injection_match", True) and "efficiency" in u}
    rows = []
    for r in stage3 + fine:
        cid, grp, pad = int(r["cell_id"]), r["group"], float(r["half"])
        base = dict(cell=cid, group=grp, pad=pad, id=r["id"], label=r.get("label"), snr_bin=r.get("snr_bin"),
                    rho_w=r.get("rho_w_archived"), gate_s=r.get("gate_total_s", 0.0))
        if r.get("no_gate"):
            u = U.get(cid)
            if u is None:
                continue
            rows.append({**base, "method": "GIS", "no_gate": True, "n_eff": u["n_eff"], "trusted": u["n_eff"] >= TRUST,
                         "passes": bool(u["recovers"]), "d_L": u["d_L"]})
            rows.append({**base, "method": "OIS", "no_gate": True, "n_eff": u["n_eff"], "trusted": u["n_eff"] >= TRUST,
                         "passes": bool(u["recovers"]), "d_L": u["d_L"]})
            rows.append({**base, "method": "NET", "no_gate": True, "passes": bool(u["network_only"]["recovers"])})
            continue
        if "gated_is" not in r:
            continue
        g = r["gated_is"]
        rows.append({**base, "method": "GIS", "n_eff": g["n_eff"], "trusted": bool(g["trusted"]),
                     "passes": bool(g["recovers"]), "d_L": g.get("d_L")})
        o = r.get("naive_is", {})
        if o.get("efficiency") is not None:
            ne = float(o["efficiency"]) * n_prop
            rows.append({**base, "method": "OIS", "n_eff": ne, "trusted": ne >= TRUST, "passes": bool(o["recovers"]),
                         "d_L": o.get("d_L")})
        if "network_only" in r:
            rows.append({**base, "method": "NET", "passes": bool(r["network_only"]["recovers"])})
    seen_u = set()
    for r in stage3 + fine:  # ungated DINGO-IS once per cell, on the cells analysed
        cid = int(r["cell_id"])
        if cid in seen_u or cid not in U:
            continue
        seen_u.add(cid)
        u = U[cid]
        rows.append(dict(cell=cid, group=r["group"], pad=float("nan"), id=f"{r['group']}_{cid:03d}_ungated",
                         label=r.get("label"), snr_bin=r.get("snr_bin"), rho_w=r.get("rho_w_archived"), gate_s=0.0,
                         method="UIS", n_eff=u["n_eff"], trusted=u["n_eff"] >= TRUST, passes=bool(u["recovers"]),
                         d_L=u["d_L"]))
    return rows


# ---------------------------------------------------------------- tables
def flag_table(rows, gmap):
    out = []
    for method in ("GIS", "OIS", "UIS"):
        for grp in ("damaged", "broken", "fine"):
            pads = sorted({r["pad"] for r in rows if r["method"] == method and r["group"] == grp and not np.isnan(r["pad"])})
            pads = pads if pads else [float("nan")]
            for pad in pads + ([None] if len(pads) > 1 else []):
                sub = [r for r in rows if r["method"] == method and r["group"] == grp
                       and (pad is None or (np.isnan(pad) and np.isnan(r["pad"])) or r["pad"] == pad)]
                if not sub:
                    continue
                net = {(r["cell"], r["pad"]): r["passes"] for r in rows if r["method"] == "NET"}
                tp = sum(r["trusted"] and r["passes"] for r in sub)
                tf = sum(r["trusted"] and not r["passes"] for r in sub)
                up = sum((not r["trusted"]) and r["passes"] for r in sub)
                uf = sum((not r["trusted"]) and not r["passes"] for r in sub)
                un = [r for r in sub if not r["trusted"]]
                un_net = [net.get((r["cell"], r["pad"])) for r in un]
                un_net = [x for x in un_net if x is not None]
                recs = [dict(glitch=gmap.get(r["cell"], str(r["cell"])), tp=int(r["trusted"] and r["passes"]),
                             tr=int(r["trusted"]), un=int(not r["trusted"])) for r in sub]
                ci = cluster_ratio_ci(recs, "tp", "tr", B=ARGS.B)
                refuse = cluster_ratio_ci([dict(glitch=x["glitch"], un=x["un"], one=1) for x in recs], "un", "one", B=ARGS.B)
                out.append(dict(method=method, group=grp, pad=("all" if pad is None else ("-" if np.isnan(pad) else pad)),
                                n=len(sub), trusted_pass=tp, trusted_fail=tf, untrusted_pass=up, untrusted_fail=uf,
                                precision=(tp / (tp + tf) if tp + tf else float("nan")),
                                precision_wilson=wilson(tp, tp + tf), precision_cluster=ci[:2], clusters=ci[2],
                                refusal=(len(un) / len(sub)), refusal_cluster=refuse[:2],
                                net_pass_in_untrusted=(sum(un_net), len(un_net))))
    return out


def mcnemar_rows(rows):
    out = []
    for grp in ("damaged", "broken", "fine"):
        for pad in sorted({r["pad"] for r in rows if r["group"] == grp and not np.isnan(r["pad"])}):
            def get(m, trusted_only):
                d = {}
                for r in rows:
                    if r["method"] == m and r["group"] == grp and r["pad"] == pad:
                        d[r["cell"]] = bool(r["passes"] and (r.get("trusted", True) if trusted_only else True))
                return d
            G, O, N = get("GIS", True), get("OIS", False), get("NET", False)
            for a_name, A, b_name, Bm in (("GIS trusted&pass", G, "NET pass", N), ("OIS pass", O, "NET pass", N),
                                          ("GIS trusted&pass", G, "OIS pass", O)):
                common = sorted(set(A) & set(Bm))
                b = sum(A[c] and not Bm[c] for c in common)
                c = sum(Bm[c_] and not A[c_] for c_ in common)
                out.append(dict(group=grp, pad=pad, a=a_name, b=b_name, n=len(common), a_only=b, b_only=c,
                                a_total=sum(A[c_] for c_ in common), b_total=sum(Bm[c_] for c_ in common),
                                p_exact=mcnemar_exact(b, c)))
    return out


def threshold_table(rows):
    out = []
    sub = [r for r in rows if r["method"] == "GIS" and r["group"] in ("damaged", "broken")]
    for t in THRESHOLDS:
        tr = [r for r in sub if r["n_eff"] >= t]
        tp = sum(r["passes"] for r in tr)
        out.append(dict(threshold=t, trusted=len(tr), trusted_pass=tp, precision=(tp / len(tr) if tr else float("nan")),
                        damaged_trusted_pass=sum(r["passes"] for r in tr if r["group"] == "damaged")))
    return out


def samples_path(folder: Path, rid: str):
    p = folder / f"{rid}_samples.csv.gz"
    return p if p.exists() else None


def per_analysis_samples(rows, stage3_dir, fine_dir, ungated_dir):
    """Yield (row, df, weight_array) for every GIS/OIS/NET/UIS row with archived samples."""
    for r in rows:
        if r["method"] == "UIS" or r.get("no_gate"):
            p = ungated_dir / f"{r['cell']:03d}_samples.csv.gz"
            col = "weight"
        else:
            p = samples_path(stage3_dir if r["group"] != "fine" else fine_dir, r["id"])
            col = {"GIS": "weight_gated", "OIS": "weight_naive", "NET": None}[r["method"]]
        if p is None or not Path(p).exists():
            continue
        df = pd.read_csv(p)
        if r["method"] == "NET" and r.get("no_gate"):
            w = None  # network alone on the unchanged package: unweighted
        elif col is None:
            w = None
        else:
            if col not in df.columns:
                continue
            w = df[col].to_numpy(float)
        yield r, df, w


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", type=Path, default=Path("results/gated_is_stage3"))
    ap.add_argument("--stage3b", type=Path, default=Path("results/gated_is_stage3b"))
    ap.add_argument("--fine", type=Path, default=Path("results/revision/stage3_fine"))
    ap.add_argument("--ungated", type=Path, default=Path("results/revision/ungated_is"))
    ap.add_argument("--ref", type=Path, default=Path("results/revision/clean_ref_v2"))
    ap.add_argument("--routes", type=Path, default=Path("results/revision/routes"))
    ap.add_argument("--real-csv", type=Path, default=Path("results/stress_real_glitches_v3_final/results.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/revision"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.smoke:
        ARGS.B = 200
    ARGS.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ARGS.stage3 / "config.json").read_text())
    n_prop = float(cfg.get("num_samples", 10000))
    gmap, gcol = glitch_map(ARGS.real_csv)
    stage3, fine, ungated = load_cells(ARGS.stage3), load_cells(ARGS.fine), load_cells(ARGS.ungated)
    stage3 = [r for r in stage3 if r.get("injection_match", True)]
    rows = analyses(stage3, fine, ungated, n_prop)
    S = {"inputs": {"stage3": len(stage3), "fine": len(fine), "ungated": len(ungated), "cluster_column": gcol,
                    "n_proposal": n_prop}}
    S["flag_table"] = flag_table(rows, gmap)
    S["mcnemar"] = mcnemar_rows(rows)
    S["threshold"] = threshold_table(rows)

    # 3b substitution (separate table; never mixed into the 1e4 tables above)
    s3b = load_cells(ARGS.stage3b)
    if s3b:
        sub = {(r["group"], int(r["cell_id"]), float(r["half"])): r for r in s3b if "gated_is" in r}
        merged = [sub.get((r["group"], int(r["cell_id"]), float(r["half"])), r) for r in stage3]
        S["flag_table_with_3b"] = [x for x in flag_table(analyses(merged, [], [], n_prop), gmap) if x["method"] == "GIS"]
        s3 = {(r["group"], int(r["cell_id"]), float(r["half"])): r for r in stage3 if "gated_is" in r}
        S["stage3b_cells"] = []
        for r in s3b:
            if "gated_is" not in r:
                continue
            old = s3.get((r["group"], int(r["cell_id"]), float(r["half"])))
            ne_old = old["gated_is"]["n_eff"] if old else float("nan")
            # n_eff should grow ~10x with 10x samples if the proposal covers the target; much less suggests missing mass
            S["stage3b_cells"].append(dict(id=r["id"], n_eff=r["gated_is"]["n_eff"], n_eff_1e4=ne_old,
                                           growth=(r["gated_is"]["n_eff"] / ne_old if ne_old else float("nan")),
                                           efficiency=r["gated_is"]["efficiency"], trusted=r["gated_is"]["trusted"],
                                           passes=r["gated_is"]["recovers"], d_L=r["gated_is"]["d_L"],
                                           elapsed_s=r.get("elapsed_s")))

    # reference v2
    ref = None
    if (ARGS.ref / "samples.csv.gz").exists():
        ref = pd.read_csv(ARGS.ref / "samples.csv.gz")
        ref["lambda_tilde"] = lambda_tilde(ref)
        wref = ref["weights"].to_numpy(float)
        meta = json.loads((ARGS.ref / "meta.json").read_text())
        S["ref_v2"] = meta
        from_pass = []
        for r in rows:
            if r["method"] in ("GIS", "OIS", "UIS") and r.get("d_L"):
                from_pass.append((r, _recovers(r["d_L"], meta["d_L"])))
        S["pass_changes_vs_ref_v2"] = [dict(id=r["id"], method=r["method"], before=r["passes"], after=a)
                                       for r, a in from_pass if bool(a) != bool(r["passes"])]

    # PSIS k-hat and JS (needs archived samples)
    khat, js_rows, floors = [], [], defaultdict(list)
    rng = np.random.default_rng(7)
    n_ref_eff = None
    if ref is not None:
        n_ref_eff = float(wref.sum() ** 2 / np.sum(wref ** 2))
    for r, df, w in per_analysis_samples(rows, ARGS.stage3, ARGS.fine, ARGS.ungated):
        if w is not None and r["method"] in ("GIS", "OIS", "UIS"):
            with np.errstate(divide="ignore"):
                khat.append(dict(id=r["id"], method=r["method"], group=r["group"], pad=r["pad"], n_eff=r.get("n_eff"),
                                 trusted=r.get("trusted"), passes=r["passes"], khat=psis_khat(np.log(np.maximum(w, 0)))))
        if ref is None:
            continue
        if {"chirp_mass", "mass_ratio", "lambda_1", "lambda_2"} <= set(df.columns):
            df = df.copy()
            df["lambda_tilde"] = lambda_tilde(df)
        js = {p: js_divergence_1d(df[p].to_numpy(float), ref[p].to_numpy(float), wx=w, wy=wref)
              for p in JS_PARAMS if p in df.columns}
        allp = [c for c in df.columns if c in ref.columns and not c.startswith("weight") and c not in ("weights",)]
        jall = [js_divergence_1d(df[p].to_numpy(float), ref[p].to_numpy(float), wx=w, wy=wref) for p in allp]
        js_rows.append(dict(id=r["id"], method=r["method"], group=r["group"], pad=r["pad"],
                            trusted=r.get("trusted"), passes=r["passes"], js=js, js_max_all=float(np.nanmax(jall)),
                            n_params_all=len(allp)))
        if r["method"] == "GIS" and r.get("trusted") and not ARGS.smoke and w is not None:
            ww = np.maximum(w, 0) / np.maximum(w, 0).sum()
            n1 = max(2, int(round(r["n_eff"])))
            for _ in range(3):
                i1 = rng.choice(len(df), size=n1, replace=True, p=ww)
                i2 = rng.choice(len(df), size=int(round(n_ref_eff)), replace=True, p=ww)
                for p in js:
                    floors[p].append(js_divergence_1d(df[p].to_numpy(float)[i1], df[p].to_numpy(float)[i2]))
    S["khat"] = khat
    S["js"] = js_rows
    S["js_floor"] = {p: float(np.median(v)) for p, v in floors.items()}

    # trusted failures
    S["trusted_failures"] = [r for r in rows if r["method"] == "GIS" and r.get("trusted") and not r["passes"]]
    # routes
    S["routes"] = load_cells(ARGS.routes)

    (ARGS.out / "revision_summary.json").write_text(json.dumps(S, indent=1, default=_jd))
    (ARGS.out / "REVISION_REPORT.md").write_text(render(S))
    print(render(S))


def _recovers(ci, clean):
    if not (20.0 <= ci["med"] <= 50.0):
        return False
    if ci["hi"] < clean["lo"] or clean["hi"] < ci["lo"]:
        return False
    return abs(ci["med"] - clean["med"]) <= 10.0


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def q(xs, p):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


def render(S):
    L = ["# Referee-revision report", "", f"Inputs: {json.dumps(S['inputs'])}", ""]
    L += ["## 1. Flag tables (method's own n_eff >= 100 flag)", "",
          "GIS = gated IS (this paper); OIS = ordinary DINGO-IS on the gated package; UIS = ordinary DINGO-IS on the UNGATED contaminated package.",
          "precision = trusted & pass / trusted. Cluster intervals resample glitches.", "",
          "| method | group | pad | n | T&P | T&F | U&P | U&F | precision | Wilson | cluster CI | refusal | refusal cluster CI | net passes among untrusted |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda x: "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"
    for t in S["flag_table"]:
        L.append(f"| {t['method']} | {t['group']} | {t['pad']} | {t['n']} | {t['trusted_pass']} | {t['trusted_fail']} | "
                 f"{t['untrusted_pass']} | {t['untrusted_fail']} | {f(t['precision'])} | {f(t['precision_wilson'][0])}–{f(t['precision_wilson'][1])} | "
                 f"{f(t['precision_cluster'][0])}–{f(t['precision_cluster'][1])} ({t['clusters']} glitches) | {f(t['refusal'])} | "
                 f"{f(t['refusal_cluster'][0])}–{f(t['refusal_cluster'][1])} | {t['net_pass_in_untrusted'][0]}/{t['net_pass_in_untrusted'][1]} |")
    L += ["", "## 2. Paired comparisons (exact McNemar)", "", "| group | pad | A | B | n | A only | B only | A total | B total | p |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in S["mcnemar"]:
        L.append(f"| {m['group']} | {m['pad']} | {m['a']} | {m['b']} | {m['n']} | {m['a_only']} | {m['b_only']} | {m['a_total']} | {m['b_total']} | {m['p_exact']:.3g} |")
    L += ["", "## 3. Trust threshold sensitivity (GIS, damaged+broken, both pads; exploratory)", "",
          "| n_eff >= | trusted | trusted & pass | precision | damaged trusted & pass |", "| --- | --- | --- | --- | --- |"]
    for t in S["threshold"]:
        L.append(f"| {t['threshold']} | {t['trusted']} | {t['trusted_pass']} | {f(t['precision'])} | {t['damaged_trusted_pass']} |")
    if "flag_table_with_3b" in S:
        L += ["", "## 3b. GIS flag table with Stage 3b cells substituted (1e5 samples where rerun)", ""]
        for t in S["flag_table_with_3b"]:
            L.append(f"- {t['group']} pad {t['pad']}: n {t['n']}, T&P {t['trusted_pass']}, T&F {t['trusted_fail']}, U&P {t['untrusted_pass']}, U&F {t['untrusted_fail']}")
        L += ["", "Stage 3b cells:", ""] + [f"- {c['id']}: n_eff {c['n_eff']:.1f} (was {c['n_eff_1e4']:.1f} at 1e4; growth x{c['growth']:.1f}), trusted {c['trusted']}, passes {c['passes']}, d_L {c['d_L']}, {c.get('elapsed_s')} s" for c in S["stage3b_cells"]]
    L += ["", "## 4. PSIS k-hat", ""]
    kh = S["khat"]
    for m in ("GIS", "OIS", "UIS"):
        for tr in (True, False):
            v = [k["khat"] for k in kh if k["method"] == m and bool(k["trusted"]) == tr]
            if v:
                L.append(f"- {m} {'trusted' if tr else 'untrusted'}: n {len(v)}, median {q(v, 50):.2f}, 90% {q(v, 90):.2f}, k>0.7: {sum(x > 0.7 for x in v)}")
    g = [k for k in kh if k["method"] == "GIS"]
    if g:
        L.append("")
        L.append("GIS crosstab (trusted by n_eff) x (k-hat <= 0.7) x pass:")
        for tr in (True, False):
            for good in (True, False):
                s = [k for k in g if bool(k["trusted"]) == tr and (k["khat"] <= 0.7) == good]
                L.append(f"- n_eff {'>=' if tr else '<'}100, k {'<=' if good else '>'}0.7: {len(s)} analyses, {sum(k['passes'] for k in s)} pass")
    L += ["", "## 5. Trusted failures", ""]
    for r in S["trusted_failures"]:
        kk = [k["khat"] for k in kh if k["id"] == r["id"] and k["method"] == "GIS"]
        L.append(f"- {r['id']}: {r['label']} {r['snr_bin']}, rho_w {r['rho_w']}, gate {r['gate_s']} s, n_eff {r['n_eff']:.1f}, "
                 f"k-hat {kk[0] if kk else 'n/a'}, d_L {r['d_L']}")
    if "ref_v2" in S:
        L += ["", "## 6. JS against the 1e5 clean reference", "", f"Reference: {json.dumps(S['ref_v2'])}", "",
              "| method | trust | n | " + " | ".join(JS_PARAMS) + " | max over all |", "| " + " | ".join(["---"] * (len(JS_PARAMS) + 4)) + " |"]
        for m in ("GIS", "OIS", "NET", "UIS"):
            for tr in ((True, False) if m != "NET" else (None,)):
                s = [j for j in S["js"] if j["method"] == m and (tr is None or bool(j["trusted"]) == tr)]
                if not s:
                    continue
                cells = [f"{q([j['js'].get(p) for j in s], 50):.3f} / {q([j['js'].get(p) for j in s], 90):.3f}" for p in JS_PARAMS]
                L.append(f"| {m} | {tr} | {len(s)} | " + " | ".join(cells) + f" | {q([j['js_max_all'] for j in s], 50):.3f} / {q([j['js_max_all'] for j in s], 90):.3f} |")
        L += ["", "(cells are median / 90th percentile)", "", f"Floor (median over trusted GIS, resampling): {json.dumps({k: round(v, 4) for k, v in S['js_floor'].items()})}"]
        L += ["", f"Pass/fail changes when judged against the 1e5 reference: {len(S['pass_changes_vs_ref_v2'])}"]
        L += [f"- {c}" for c in S["pass_changes_vs_ref_v2"]]
    if S["routes"]:
        L += ["", "## 7. Independent-proposal routes", ""]
        for r in S["routes"]:
            L.append(f"- {r.get('id')}: " + json.dumps({k: {kk: v.get(kk) for kk in ('n_eff', 'efficiency', 'd_L', 'khat')} for k, v in r.get('routes', {}).items()}) +
                     " JS(d_L): " + json.dumps({k: round(v.get('luminosity_distance', float('nan')), 4) for k, v in r.get('js', {}).items()}))
    return "\n".join(L) + "\n"


ARGS = None

if __name__ == "__main__":
    main()
