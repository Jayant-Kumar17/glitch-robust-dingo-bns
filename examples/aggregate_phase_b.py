#!/usr/bin/env python3
"""Collect the Phase B (honest-gate) results into one markdown report.

Reads results/gate_tolerance_v3/*, results/journal_method_hardening_v3_*,
results/stress_real_glitches_v3_* and results/gw170817_raw_l1_v3_*.
Poisoned and gated posteriors are scored with the SAME recovery criterion
(d_L median in [20, 50] Mpc, within 10 Mpc of the clean median, 90% intervals overlap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _recovers(lo, med, hi, clean) -> pd.Series:
    return (
        (med >= 20.0)
        & (med <= 50.0)
        & ((med - clean["med"]).abs() <= 10.0)
        & (hi >= clean["lo"])
        & (lo <= clean["hi"])
    )


def _wilson(k: int, n: int, z: float = 1.959964) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} [{100 * max(c - h, 0):.0f}%, {100 * min(c + h, 1):.0f}%]"


def _md(rows: list[dict], transpose: bool = False) -> str:
    """Plain markdown table (no tabulate dependency)."""
    if transpose:
        keys = list(rows[0].keys())
        rows = [{"metric": k, **{f"run {i + 1}": r[k] for i, r in enumerate(rows)}} for k in keys]
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def gate_tolerance(root: Path) -> list[str]:
    rows = []
    for d in sorted(root.glob("*")):
        cfg_f, res_f = d / "config.json", d / "results.csv"
        if not (cfg_f.is_file() and res_f.is_file()):
            continue
        cfg = json.loads(cfg_f.read_text())
        df = pd.read_csv(res_f)
        ok = df[df["error"].fillna("") == ""]
        h1 = ok[ok["detectors"] == "H1"]
        rows.append(
            {
                "run": d.name,
                "t_rel range [s]": "%g..%g" % tuple(cfg.get("t_rel_range", [np.nan, np.nan])),
                "half-width [s]": cfg.get("gate_half_s"),
                "fill": cfg.get("fill", "zero"),
                "band [Hz]": cfg.get("gate_band_hz") or "raw",
                "H1 recovered": _wilson(int(h1["gated_recovers"].astype(bool).sum()), len(h1)),
                "H1 collapsed": int(h1["poison_style_collapse"].astype(bool).sum()),
                "mean abs d_L median shift [Mpc]": round(float(h1["delta_dL_med"].abs().mean()), 2) if len(h1) else None,
                "max JS [nat]": round(float(h1["js_max"].max()), 3) if len(h1) else None,
                "JS floor [nat]": round(float(cfg.get("js_null_max_nat") or np.nan), 3),
            }
        )
    if not rows:
        return ["_no gate-tolerance runs found_"]
    return [_md(rows)]


def hardening(results: Path) -> list[str]:
    rows = []
    for d in sorted(results.glob("journal_method_hardening_v3_*")):
        f = d / "ablation_summary.json"
        if not f.is_file():
            continue
        s = json.loads(f.read_text())
        for arm, v in s.get("by_arm", {}).items():
            n = int(v["n"])
            rows.append(
                {
                    "run": d.name.replace("journal_method_hardening_v3_", ""),
                    "arm": arm,
                    "recovers": _wilson(int(round(v["recovers_like_clean"] * n)), n),
                    "collapsed": f"{int(round(v['poison_collapsed'] * n))}/{n}",
                    "median d_L [Mpc]": round(float(v["median_med_d_L"]), 1),
                }
            )
    if not rows:
        return ["_no hardening runs found_"]
    return [_md(rows)]


def real_panel(results: Path) -> list[str]:
    rows = []
    for d in sorted(results.glob("stress_real_glitches_v3_*")):
        res_f, clean_f = d / "results.csv", d / "clean_reference.json"
        if not (res_f.is_file() and clean_f.is_file()):
            continue
        clean = json.loads(clean_f.read_text())
        clean = clean.get("d_L", clean.get("luminosity_distance", clean))
        r = pd.read_csv(res_f)
        r = r[r["error"].fillna("") == ""]
        g = r[(~r["is_noise_control"].astype(bool)) & (r["injected_rho_w"].astype(float) > 1.0)].copy()
        if not len(g):
            continue
        for col in ("poison_collapsed", "gated_recovers", "oracle_recovers", "detector_fired"):
            g[col] = g[col].astype(bool)
        g["poison_ok"] = _recovers(g["poison_lo"], g["poison_med"], g["poison_hi"], clean)
        dam, col, und = g[~g["poison_ok"]], g[g["poison_collapsed"]], g[g["poison_ok"]]
        trig = col.groupby("gs_event_time")
        rows.append(
            {
                "run": d.name.replace("stress_real_glitches_v3_", ""),
                "cells / triggers": f"{len(g)} / {g['gs_event_time'].nunique()}",
                "damaged (fail criterion un-gated)": len(dam),
                "collapsed (cells / triggers)": f"{len(col)} / {col['gs_event_time'].nunique()}",
                "detector gate: damaged recovered": _wilson(int(dam["gated_recovers"].sum()), len(dam)),
                "oracle gate: damaged recovered": _wilson(int(dam["oracle_recovers"].sum()), len(dam)),
                "detector gate: collapsed triggers fully recovered": _wilson(
                    int(trig["gated_recovers"].all().sum()), int(trig.ngroups)
                ),
                "oracle gate: collapsed triggers fully recovered": _wilson(
                    int(trig["oracle_recovers"].all().sum()), int(trig.ngroups)
                ),
                "detector gate breaks undamaged": f"{int((~und['gated_recovers']).sum())}/{len(und)}",
                "oracle gate breaks undamaged": f"{int((~und['oracle_recovers']).sum())}/{len(und)}",
            }
        )
    if not rows:
        return ["_no real-glitch runs found_"]
    return [_md(rows, transpose=True)]


def raw_l1(results: Path) -> list[str]:
    rows = []
    for d in sorted(results.glob("gw170817_raw_l1_v3_*")):
        f = d / "summary.json"
        if not f.is_file():
            continue
        s = json.loads(f.read_text())
        for label, e in s.get("efficiency", {}).items():
            rows.append(
                {
                    "run": d.name.replace("gw170817_raw_l1_v3_", ""),
                    "package": label,
                    "efficiency": e.get("sample_efficiency"),
                    "n_eff": e.get("n_eff"),
                    "log Z": e.get("log_evidence"),
                }
            )
    if not rows:
        return ["_no raw-L1 runs found_"]
    return [_md(rows)]


def main() -> None:
    results = REPO_ROOT / "results"
    out = ["# Phase B report: honest gates after the packaging-convention fix", ""]
    out += ["## 1. Forced gates on clean GW170817 (H1 only cells)", ""] + gate_tolerance(results / "gate_tolerance_v3") + [""]
    out += ["## 2. Hardening cells, detector-driven gates", ""] + hardening(results) + [""]
    out += ["## 3. Real Gravity Spy panel", ""] + real_panel(results) + [""]
    out += ["## 4. Unsubtracted GW170817 L1", ""] + raw_l1(results) + [""]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
