#!/usr/bin/env python3
"""Final numbers for the audit paper, written to results/PHASE_C_REPORT.md.

Poisoned and gated posteriors use the SAME recovery criterion: d_L median in
[20, 50] Mpc, within 10 Mpc of the clean median, 90% intervals overlap.
"Damaged" = poisoned posterior fails that criterion. "Collapsed" = d_L 90%
interval upper edge < 15 Mpc or lower edge > 90 Mpc. Intervals: Wilson 95% on
cells, plus a trigger-level cluster bootstrap (10,000 resamples of glitches).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "results"
REAL = RES / "stress_real_glitches_v3_final"
REAL_OLD = RES / "stress_real_glitches_v2"
SWEEP = RES / "collapse_threshold_v3"
RAWL1 = RES / "gw170817_raw_l1_v3_final"
MULTI = RES / "multi_param_validation_v3"
LOCAUD = RES / "localizer_audit_v3"
GATETOL = RES / "gate_tolerance_v3"


def wilson(k: int, n: int, z: float = 1.959964) -> str:
    if n == 0:
        return "0/0"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} ({100 * p:.0f}%; {100 * max(c - h, 0):.0f}–{100 * min(c + h, 1):.0f}%)"


def cluster_boot(df: pd.DataFrame, col: str, key: str = "gs_event_time", n_boot: int = 10_000, seed: int = 0) -> str:
    if len(df) == 0:
        return "n/a"
    g = df.groupby(key)[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s = g["sum"].to_numpy()[idx].sum(1)
    c = g["count"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(s / c, [2.5, 97.5])
    return f"{100 * lo:.0f}–{100 * hi:.0f}% ({len(g)} glitches)"


def recovers(lo, med, hi, clean) -> pd.Series:
    return (med >= 20) & (med <= 50) & ((med - clean["med"]).abs() <= 10) & (hi >= clean["lo"]) & (lo <= clean["hi"])


def md(rows: list[dict]) -> str:
    if not rows:
        return "_none_"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    out += ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _clean_ref(path: Path) -> dict:
    c = json.loads(path.read_text())
    return c.get("d_L", c.get("luminosity_distance", c))


def section_localizer() -> list[str]:
    f = LOCAUD / "summary.json"
    if not f.is_file():
        return ["_localizer audit missing_"]
    s = json.loads(f.read_text())
    rows = []
    for det, r in s.get("gw170817_segment", {}).items():
        rows.append({"population": f"GW170817 segment {det}", "n": 1, "max |z|": round(r["zmax"], 2),
                     **{f"fires at {x:g}σ": int(r[f"fires_{x:g}"]) for x in s["sigmas"]}})
    for det, r in (s.get("offsource") or {}).items():
        rows.append({"population": f"off-source {det}", "n": r["n"], "max |z|": f"median {r['zmax_median']}",
                     **{f"fires at {x:g}σ": (None if r[f"fire_rate_{x:g}"] is None else round(r[f"fire_rate_{x:g}"] * r["n"]))
                        for x in s["sigmas"]}})
    for det, r in s.get("gaussian", {}).items():
        rows.append({"population": f"Gaussian {det}", "n": r["n"], "max |z|": f"median {r['zmax_median']:.2f}, max {r['zmax_max']:.2f}",
                     **{f"fires at {x:g}σ": r[f"fires_{x:g}"] for x in s["sigmas"]}})
    out = [md(rows)]
    if s.get("offsource_error"):
        out.append(f"\nOff-source error: `{s['offsource_error']}`")
    return out


def section_real() -> list[str]:
    f, cf = REAL / "results.csv", REAL / "clean_reference.json"
    if not (f.is_file() and cf.is_file()):
        return ["_real panel missing_"]
    clean = _clean_ref(cf)
    r = pd.read_csv(f)
    r = r[r["error"].fillna("") == ""]
    for c in ("poison_collapsed", "gated_recovers", "oracle_recovers", "detector_fired", "is_noise_control"):
        r[c] = r[c].astype(str).str.lower().isin(["true", "1"])
    noise = r[r["is_noise_control"]]
    g = r[(~r["is_noise_control"]) & (r["injected_rho_w"].astype(float) > 1.0)].copy()
    g["damaged"] = ~recovers(g["poison_lo"], g["poison_med"], g["poison_hi"], clean)
    out = [f"Clean reference d_L: median {clean['med']:.1f}, 90% [{clean['lo']:.1f}, {clean['hi']:.1f}] Mpc. "
           f"{len(g)} cells from {g['gs_event_time'].nunique()} glitches; {len(noise)} noise-only controls.", ""]

    def block(sub: pd.DataFrame, label: str) -> dict:
        dmg, col, und = sub[sub["damaged"]], sub[sub["poison_collapsed"]], sub[~sub["damaged"]]
        return {
            "subset": label,
            "cells/glitches": f"{len(sub)}/{sub['gs_event_time'].nunique()}",
            "median ρ_w": round(float(sub["injected_rho_w"].median()), 0) if len(sub) else None,
            "collapsed": wilson(int(sub["poison_collapsed"].sum()), len(sub)),
            "collapsed glitches": int(col["gs_event_time"].nunique()),
            "damaged": wilson(int(sub["damaged"].sum()), len(sub)),
            "localiser fired": wilson(int(sub["detector_fired"].sum()), len(sub)),
            "localiser: damaged recovered": wilson(int(dmg["gated_recovers"].sum()), len(dmg)),
            "localiser: collapsed recovered": wilson(int(col["gated_recovers"].sum()), len(col)),
            "localiser: undamaged broken": f"{int((~und['gated_recovers']).sum())}/{len(und)}",
            "oracle: damaged recovered": wilson(int(dmg["oracle_recovers"].sum()), len(dmg)),
            "oracle: collapsed recovered": wilson(int(col["oracle_recovers"].sum()), len(col)),
            "oracle: undamaged broken": f"{int((~und['oracle_recovers']).sum())}/{len(und)}",
        }

    rows = [block(sub, f"Omicron SNR {b}") for b, sub in g.groupby("gs_snr_bin", sort=False)]
    rows.append(block(g, "all"))
    rows += [block(sub, f"{m}") for m, sub in g.groupby("gs_label")]
    out.append(md(rows))
    dmg = g[g["damaged"]]
    out += ["", "Trigger-level (cluster bootstrap) 95% intervals:",
            f"- collapse rate, all cells: {cluster_boot(g.assign(x=g['poison_collapsed'].astype(int)), 'x')}",
            f"- localiser recovery of damaged cells: {cluster_boot(dmg.assign(x=dmg['gated_recovers'].astype(int)), 'x')}",
            f"- oracle recovery of damaged cells: {cluster_boot(dmg.assign(x=dmg['oracle_recovers'].astype(int)), 'x')}",
            "", f"Noise-only controls: localiser fired {int(noise['detector_fired'].sum())}/{len(noise)}, "
            f"recovered {int(noise['gated_recovers'].sum())}/{len(noise)}."]
    # collapse vs loudness for the collapsing morphology(ies)
    bins = [0, 100, 170, 250, 350, 500, 750, 1e9]
    g["rho_bin"] = pd.cut(g["injected_rho_w"].astype(float), bins)
    lr = [{"ρ_w bin": str(b), "cells": len(s), "collapsed": wilson(int(s["poison_collapsed"].sum()), len(s)),
           "damaged": int(s["damaged"].sum())} for b, s in g.groupby("rho_bin", observed=True)]
    out += ["", "Collapse versus injected ρ_w (all morphologies):", "", md(lr)]
    # placement dependence: same glitch, different placement
    per = g.groupby("gs_event_time")["poison_collapsed"].agg(["sum", "count"])
    mixed = int(((per["sum"] > 0) & (per["sum"] < per["count"])).sum())
    out += ["", f"Glitches that collapse at some placements but not others: {mixed} of {int((per['sum'] > 0).sum())} collapsing glitches."]
    return out


def section_catalogue_diff() -> list[str]:
    a, b = REAL_OLD / "glitch_catalogue.csv", REAL / "glitch_catalogue.csv"
    if not (a.is_file() and b.is_file()):
        return ["_catalogue diff unavailable_"]
    old, new = pd.read_csv(a), pd.read_csv(b)
    key = "gravityspy_id"
    so = set(old.loc[old.get("excerpt_native_snr", pd.Series(dtype=float)).fillna(0) > 1, key]) if "excerpt_native_snr" in old else set(old[key])
    sn = set(new.loc[new.get("excerpt_native_snr", pd.Series(dtype=float)).fillna(0) > 1, key]) if "excerpt_native_snr" in new else set(new[key])
    cols = [c for c in ("gravityspy_id", "ml_label", "snr", "duration", "fetched", "excerpt_native_snr", "excerpt_excess_energy_w", "excerpt_background_rms_w") if c in new.columns]
    out = [f"Usable excerpts: v2 {len(so)}, v3 {len(sn)}. Only in v3: {len(sn - so)}. Only in v2: {len(so - sn)}.", ""]
    rows = []
    for gid in sorted(sn ^ so):
        o = old[old[key] == gid]
        n = new[new[key] == gid]
        rows.append({"id": gid, "label": (n if len(n) else o)["ml_label"].iloc[0],
                     "v2 native SNR": None if not len(o) else o["excerpt_native_snr"].iloc[0] if "excerpt_native_snr" in o else None,
                     "v3 native SNR": None if not len(n) else n["excerpt_native_snr"].iloc[0] if "excerpt_native_snr" in n else None,
                     "v2 background rms": None if not len(o) else o.get("excerpt_background_rms_w", pd.Series([None])).iloc[0],
                     "v3 background rms": None if not len(n) else n.get("excerpt_background_rms_w", pd.Series([None])).iloc[0]})
    out.append(md(rows))
    return out


def section_sweep() -> list[str]:
    f, cf = SWEEP / "results.csv", SWEEP / "clean_reference.json"
    if not (f.is_file() and cf.is_file()):
        return ["_sweep missing_"]
    clean = _clean_ref(cf)
    r = pd.read_csv(f)
    r = r[r["error"].fillna("") == ""]
    for c in ("poison_collapsed", "gated_recovers", "detector_fired"):
        r[c] = r[c].astype(str).str.lower().isin(["true", "1"])
    r["damaged"] = ~recovers(r["poison_lo"], r["poison_med"], r["poison_hi"], clean)
    rows = []
    for (fam, rho), s in r.groupby(["family", "rho_w_target"]):
        d = s[s["damaged"]]
        rows.append({"family": fam, "ρ_w": int(round(rho)), "collapsed": wilson(int(s["poison_collapsed"].sum()), len(s)),
                     "damaged": int(len(d)), "localiser fired": f"{int(s['detector_fired'].sum())}/{len(s)}",
                     "gated recovers (all)": f"{int(s['gated_recovers'].sum())}/{len(s)}",
                     "gated recovers (damaged)": f"{int(d['gated_recovers'].sum())}/{len(d)}"})
    out = [md(rows)]
    pooled = r.groupby("rho_w_target")["poison_collapsed"].mean()
    x, y = np.log10(pooled.index.to_numpy(float)), pooled.to_numpy(float)
    cross = None
    for i in range(len(y) - 1):
        if y[i] < 0.5 <= y[i + 1]:
            cross = 10 ** (x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
            break
    out.append(f"\nPooled 50% collapse crossing (log-linear interpolation): {'none' if cross is None else f'ρ_w ≈ {cross:.0f}'}. "
               f"Lowest ρ_w with any collapse: {r.loc[r['poison_collapsed'], 'rho_w_target'].min() if r['poison_collapsed'].any() else 'none'}.")
    return out


def section_raw_l1() -> list[str]:
    f = RAWL1 / "summary.json"
    if not f.is_file():
        return ["_raw L1 run missing_"]
    s = json.loads(f.read_text())
    rows = [{"package": k, "efficiency": v.get("sample_efficiency"), "n_eff": v.get("n_eff"), "log Z": v.get("log_evidence")}
            for k, v in s.get("efficiency", {}).items()]
    out = [md(rows), "", f"Gates: {json.dumps(s.get('detector', {}).get('gates'))}",
           f"Localiser: {json.dumps(s.get('detector', {}).get('localizer'))}"]
    for key in ("raw_l1_poisoned_vs_ref", "raw_l1_adapt_full_vs_ref"):
        c = s.get(key, {})
        dl = c.get("medians", {}).get("luminosity_distance", {})
        out.append(f"- {key}: JS median {c.get('js_median')}, JS max {c.get('js_max')}, "
                   f"all 90% intervals overlap: {all(c.get('ci90_overlap', {}).values()) if c.get('ci90_overlap') else None}, "
                   f"d_L ref {dl.get('reference')} vs {dl.get('other')}")
        if c.get("js_nats"):
            worst = sorted(c["js_nats"].items(), key=lambda kv: -kv[1])[:4]
            out.append(f"  largest JS: {worst}; non-overlapping: {[k for k, v in c.get('ci90_overlap', {}).items() if not v]}")
    return out


def section_multi() -> list[str]:
    f = MULTI / "summary.json"
    if not f.is_file():
        return ["_multi-parameter run missing_"]
    s = json.loads(f.read_text())
    blk = s.get("real_koi_fish", {})
    return ["```json", json.dumps(blk, indent=1)[:6000], "```"]


def main() -> None:
    out = ["# Phase C report: final numbers for the audit paper", ""]
    out += ["## 1. Localiser false-trigger audit", ""] + section_localizer() + [""]
    out += ["## 2. Real Gravity Spy panel (band-limited, zero fill, ±0.1 s)", ""] + section_real() + [""]
    out += ["### 2a. Excerpts that differ between the v2 and v3 panels", ""] + section_catalogue_diff() + [""]
    out += ["## 3. Loudness sweep", ""] + section_sweep() + [""]
    out += ["## 4. Unsubtracted GW170817 L1", ""] + section_raw_l1() + [""]
    out += ["## 5. Fifteen-parameter importance-sampled comparison (collapsed real cells)", ""] + section_multi() + [""]
    out += [f"## 6. Gate tolerance on clean data", "", f"Unchanged from Phase B: see {GATETOL.relative_to(REPO_ROOT)}."]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
