#!/usr/bin/env python3
"""Cluster-adjusted recovery statistics from archived result CSVs.

No parameter estimation. Reads:

  results/stress_real_glitches_v2/results.csv
  results/collapse_threshold_v1/results.csv
  results/stress_test_excision_v1/results.csv
  results/stress_test_synthetic_bns_v1/results.csv

The real-glitch panel is aggregated at the Gravity Spy trigger
(``gs_event_time``): 61 glitches, three time placements each, after dropping
noise-only controls and excerpts with no whitened excess (``injected_snr_w``
<= 1). Recovery and collapse flags are the archived Section 4.4 columns.

Writes ``results/stress_real_glitches_v2/cluster_summary.json`` and
``cluster_summary.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import REPO_ROOT, resolve_under_repo

Z95 = 1.959963984540054
N_BOOT = 10_000
RNG_SEED = 0

SNR_ORDER = ["[8,30)", "[30,100)", "[100,300)", "[300,inf)"]
LABEL_PRETTY = {
    "Blip": "Blip",
    "Koi_Fish": "Koi Fish",
    "Scattered_Light": "Scattered Light",
    "Tomte": "Tomte",
    "Whistle": "Whistle",
}


def wilson_interval(k: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. ``k`` and ``n`` may be effective (non-integer) counts."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = float(np.clip(k / n, 0.0, 1.0))
    z2 = z * z
    den = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / den
    half = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / den
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def design_effect(cluster_outcomes: list[np.ndarray]) -> dict:
    """One-way ANOVA intraclass correlation for equal or unequal binary clusters."""
    sizes = np.asarray([len(y) for y in cluster_outcomes], dtype=np.float64)
    k = int(sizes.size)
    n = float(sizes.sum())
    if k < 2 or n <= k:
        return {"icc": 0.0, "deff": 1.0, "n_eff": n, "m_bar": float(sizes.mean()) if k else 0.0}
    means = np.asarray([float(np.mean(y)) for y in cluster_outcomes], dtype=np.float64)
    p = float(np.sum(sizes * means) / n)
    ssb = float(np.sum(sizes * (means - p) ** 2))
    ssw = 0.0
    for y, m in zip(cluster_outcomes, means):
        ssw += float(np.sum((y.astype(np.float64) - m) ** 2))
    df_b = k - 1
    df_w = n - k
    msb = ssb / df_b
    msw = ssw / df_w if df_w > 0 else 0.0
    # Average cluster size used by the ICC estimator (unequal n).
    m0 = float((n - np.sum(sizes ** 2) / n) / df_b)
    if m0 <= 1.0 or (msb <= 0 and msw <= 0):
        icc = 0.0
    else:
        icc = (msb - msw) / (msb + (m0 - 1.0) * msw)
    icc = float(np.clip(icc, 0.0, 1.0))
    m_bar = float(sizes.mean())
    deff = 1.0 + (m_bar - 1.0) * icc
    n_eff = n / deff if deff > 0 else n
    return {"icc": icc, "deff": float(deff), "n_eff": float(n_eff), "m_bar": m_bar}


def cluster_bootstrap_rate(
    clusters: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    reducer=None,
) -> dict:
    """Percentile interval for a statistic of resampled trigger IDs.

    Default statistic is the cell-level mean (each resampled trigger contributes
    all of its time placements). ``reducer`` maps a list of per-cluster boolean
    arrays to a scalar (used for strict / majority glitch rates).
    """
    ids = np.asarray(list(clusters.keys()))
    k = ids.size
    if k == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan")}

    def cell_rate(chosen: np.ndarray) -> float:
        parts = [clusters[i] for i in chosen]
        return float(np.concatenate(parts).mean())

    stat = reducer or cell_rate
    point = stat(ids)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(ids, size=k, replace=True)
        draws[b] = stat(chosen)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi)}


def _strict(chosen: np.ndarray, clusters: dict[str, np.ndarray]) -> float:
    ok = [bool(np.all(clusters[i])) for i in chosen]
    return float(np.mean(ok)) if ok else float("nan")


def _majority(chosen: np.ndarray, clusters: dict[str, np.ndarray]) -> float:
    ok = [bool(np.mean(clusters[i]) >= 2.0 / 3.0) for i in chosen]
    return float(np.mean(ok)) if ok else float("nan")


def summarize_panel(frame: pd.DataFrame, cluster_col: str, rng: np.random.Generator) -> dict:
    grouped = {
        str(key): grp["gated_recovers"].to_numpy(dtype=bool)
        for key, grp in frame.groupby(cluster_col, sort=False)
    }
    outcomes = list(grouped.values())
    n_cells = int(sum(len(y) for y in outcomes))
    n_rec = int(sum(int(y.sum()) for y in outcomes))
    de = design_effect(outcomes)
    cell_boot = cluster_bootstrap_rate(grouped, rng=rng)
    strict_boot = cluster_bootstrap_rate(
        grouped, rng=rng, reducer=lambda chosen: _strict(chosen, grouped)
    )
    maj_boot = cluster_bootstrap_rate(
        grouped, rng=rng, reducer=lambda chosen: _majority(chosen, grouped)
    )
    n_strict = int(sum(bool(np.all(y)) for y in outcomes))
    n_majority = int(sum(bool(np.mean(y) >= 2.0 / 3.0) for y in outcomes))
    n_glitches = len(outcomes)
    w_lo, w_hi = wilson_interval(n_rec / n_cells * de["n_eff"], de["n_eff"]) if n_cells else (float("nan"), float("nan"))
    return {
        "n_glitches": n_glitches,
        "n_cells": n_cells,
        "n_recovered_cells": n_rec,
        "cell_recovery": n_rec / n_cells if n_cells else float("nan"),
        "cell_bootstrap_95": [cell_boot["lo"], cell_boot["hi"]],
        "design_effect": de,
        "wilson_deff_95": [w_lo, w_hi],
        "n_strict_3of3": n_strict,
        "strict_recovery": n_strict / n_glitches if n_glitches else float("nan"),
        "strict_bootstrap_95": [strict_boot["lo"], strict_boot["hi"]],
        "n_majority_ge2of3": n_majority,
        "majority_recovery": n_majority / n_glitches if n_glitches else float("nan"),
        "majority_bootstrap_95": [maj_boot["lo"], maj_boot["hi"]],
    }


def conditional_recovery(frame: pd.DataFrame, label: str) -> dict:
    collapsed = frame[frame["poison_collapsed"].astype(bool)]
    n = int(len(collapsed))
    k = int(collapsed["gated_recovers"].astype(bool).sum()) if n else 0
    lo, hi = wilson_interval(k, n) if n else (float("nan"), float("nan"))
    return {
        "panel": label,
        "n_collapsed": n,
        "n_recovered_given_collapse": k,
        "proportion": (k / n) if n else float("nan"),
        "wilson_95": [lo, hi],
    }


def load_real_native(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    native = df["severity_mode"].astype(str).eq("native")
    noise = df["is_noise_control"].astype(bool)
    usable = df["injected_snr_w"].astype(float) > 1.0
    out = df[native & ~noise & usable].copy()
    out["cluster_id"] = out["gs_event_time"].map(lambda x: f"{float(x):.5f}")
    out["morphology"] = out["gs_label"].map(lambda s: LABEL_PRETTY.get(str(s), str(s)))
    return out


def row_from_summary(scope: str, stratum: str, summary: dict) -> dict:
    de = summary["design_effect"]
    return {
        "scope": scope,
        "stratum": stratum,
        "n_glitches": summary["n_glitches"],
        "n_cells": summary["n_cells"],
        "n_recovered_cells": summary["n_recovered_cells"],
        "cell_recovery": summary["cell_recovery"],
        "cell_boot_lo": summary["cell_bootstrap_95"][0],
        "cell_boot_hi": summary["cell_bootstrap_95"][1],
        "wilson_deff_lo": summary["wilson_deff_95"][0],
        "wilson_deff_hi": summary["wilson_deff_95"][1],
        "icc": de["icc"],
        "deff": de["deff"],
        "n_eff": de["n_eff"],
        "n_strict_3of3": summary["n_strict_3of3"],
        "strict_recovery": summary["strict_recovery"],
        "strict_boot_lo": summary["strict_bootstrap_95"][0],
        "strict_boot_hi": summary["strict_bootstrap_95"][1],
        "n_majority_ge2of3": summary["n_majority_ge2of3"],
        "majority_recovery": summary["majority_recovery"],
        "majority_boot_lo": summary["majority_bootstrap_95"][0],
        "majority_boot_hi": summary["majority_bootstrap_95"][1],
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    real_path = resolve_under_repo("results/stress_real_glitches_v2/results.csv")
    real = load_real_native(real_path)

    by_stratum: dict[str, dict] = {"all": summarize_panel(real, "cluster_id", rng)}
    snr_rows = []
    for snr in SNR_ORDER:
        sub = real[real["gs_snr_bin"].astype(str) == snr]
        key = f"snr:{snr}"
        by_stratum[key] = summarize_panel(sub, "cluster_id", rng)
        snr_rows.append(row_from_summary("real_native_snr", snr, by_stratum[key]))
    morph_rows = []
    for morph in ["Blip", "Koi Fish", "Scattered Light", "Tomte", "Whistle"]:
        sub = real[real["morphology"] == morph]
        key = f"morph:{morph}"
        by_stratum[key] = summarize_panel(sub, "cluster_id", rng)
        morph_rows.append(row_from_summary("real_native_morphology", morph, by_stratum[key]))

    sweep = pd.read_csv(resolve_under_repo("results/collapse_threshold_v1/results.csv"))
    grid = pd.read_csv(resolve_under_repo("results/stress_test_excision_v1/results.csv"))
    synth = pd.read_csv(resolve_under_repo("results/stress_test_synthetic_bns_v1/results.csv"))
    conditional = [
        conditional_recovery(real, "stress_real_glitches_v2"),
        conditional_recovery(sweep, "collapse_threshold_v1"),
        conditional_recovery(grid, "stress_test_excision_v1"),
        conditional_recovery(synth, "stress_test_synthetic_bns_v1"),
    ]

    payload = {
        "criterion": {
            "collapse": "archived poison_collapsed: d_L 90% CI hi<15 or lo>90",
            "recovery": "archived gated_recovers: Section 4.4 d_L median in [20,50], CI overlap, |median shift|<=10 Mpc",
            "real_panel_filter": "severity_mode==native, not noise control, injected_snr_w>1",
            "cluster": "gs_event_time (Gravity Spy trigger), 3 time placements",
            "bootstrap": f"{N_BOOT} resamples of trigger IDs, seed {RNG_SEED}, percentile 2.5/97.5",
            "wilson_deff": "Wilson interval at n_eff = n_cells / (1+(m_bar-1)*ICC), ICC from one-way ANOVA, floored at 0",
        },
        "real_native": by_stratum["all"],
        "by_omicron_snr": {k: by_stratum[k] for k in by_stratum if k.startswith("snr:")},
        "by_morphology": {k: by_stratum[k] for k in by_stratum if k.startswith("morph:")},
        "conditional_on_collapse": conditional,
    }
    outdir = resolve_under_repo("results/stress_real_glitches_v2")
    (outdir / "cluster_summary.json").write_text(json.dumps(payload, indent=2))

    rows = [row_from_summary("real_native", "all", by_stratum["all"]), *snr_rows, *morph_rows]
    for item in conditional:
        rows.append(
            {
                "scope": "conditional_on_collapse",
                "stratum": item["panel"],
                "n_glitches": "",
                "n_cells": item["n_collapsed"],
                "n_recovered_cells": item["n_recovered_given_collapse"],
                "cell_recovery": item["proportion"],
                "cell_boot_lo": "",
                "cell_boot_hi": "",
                "wilson_deff_lo": item["wilson_95"][0],
                "wilson_deff_hi": item["wilson_95"][1],
                "icc": "",
                "deff": "",
                "n_eff": item["n_collapsed"],
                "n_strict_3of3": "",
                "strict_recovery": "",
                "strict_boot_lo": "",
                "strict_boot_hi": "",
                "n_majority_ge2of3": "",
                "majority_recovery": "",
                "majority_boot_lo": "",
                "majority_boot_hi": "",
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "cluster_summary.csv", index=False)

    overall = by_stratum["all"]
    print(
        f"real cells {overall['n_recovered_cells']}/{overall['n_cells']} "
        f"strict {overall['n_strict_3of3']}/{overall['n_glitches']} "
        f"majority {overall['n_majority_ge2of3']}/{overall['n_glitches']}"
    )
    for item in conditional:
        print(
            f"conditional {item['panel']}: "
            f"{item['n_recovered_given_collapse']}/{item['n_collapsed']} "
            f"wilson {item['wilson_95']}"
        )


if __name__ == "__main__":
    main()
