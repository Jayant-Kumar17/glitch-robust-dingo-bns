#!/usr/bin/env python3
"""Fig. 5 (optional): n_eff against the luminosity-distance JS divergence for every Stage 3 cell.

Run this from the REPOSITORY root (it reads the per-cell JSON files written by
examples/gated_is_stage3.py), then copy the PDF into the paper's figures/ folder:

    python <paper>/scripts/make_trust_scatter.py \
        --stage3 results/gated_is_stage3 [--stage3b results/gated_is_stage3b] \
        --out <paper>/figures/fig5_trust_scatter.pdf

The manuscript includes figures/fig5_trust_scatter.pdf automatically if it exists.
It also prints the counts the caption quotes, so they can be checked against the text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 8.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def load(folder: Path):
    rows = []
    for f in sorted((folder / "cells").glob("*.json")):
        r = json.loads(f.read_text())
        if "gated_is" not in r or "js_gated_vs_clean" not in r:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", type=Path, default=Path("results/gated_is_stage3"))
    ap.add_argument("--stage3b", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("fig5_trust_scatter.pdf"))
    a = ap.parse_args()
    rows = load(a.stage3)
    if a.stage3b is not None and a.stage3b.exists():
        newer = {(r["group"], r["cell_id"], r["half"]): r for r in load(a.stage3b)}
        rows = [newer.get((r["group"], r["cell_id"], r["half"]), r) for r in rows]
    if not rows:
        raise SystemExit(f"no analysed cells under {a.stage3}/cells")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for grp, col, mk in (("damaged", BLUE, "o"), ("broken", ORANGE, "s")):
        for ok in (True, False):
            sub = [r for r in rows if r["group"] == grp and bool(r["gated_is"]["recovers"]) == ok]
            if not sub:
                continue
            x = [max(r["gated_is"]["n_eff"], 1.0) for r in sub]
            y = [max(r["js_gated_vs_clean"].get("luminosity_distance", float("nan")), 1e-4) for r in sub]
            ax.scatter(x, y, s=16, marker=mk, facecolors=col if ok else "none", edgecolors=col, linewidths=0.9,
                       label=f"{grp}, $d_L$ {'passes' if ok else 'fails'}", zorder=3)
    ax.axvline(100, color=MUTED, ls=":", lw=1.0)
    ax.text(110, 0.6, "trusted", color=MUTED, fontsize=7)
    ax.text(90, 0.6, "untrusted", color=MUTED, fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gated-IS effective sample size $n_{\\rm eff}$")
    ax.set_ylabel("$d_L$ JS divergence vs clean (nat)")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.savefig(a.out)

    tr = [r for r in rows if r["gated_is"]["trusted"]]
    un = [r for r in rows if not r["gated_is"]["trusted"]]
    print(f"cells {len(rows)}; trusted {len(tr)} (d_L passes {sum(r['gated_is']['recovers'] for r in tr)}); "
          f"untrusted {len(un)} (gated d_L passes {sum(r['gated_is']['recovers'] for r in un)}, "
          f"network alone passes {sum(r['network_only']['recovers'] for r in un)})")
    js_tr = sorted(r["js_gated_vs_clean"].get("luminosity_distance") for r in tr)
    print(f"trusted d_L JS: median {js_tr[len(js_tr) // 2]:.4f}, max {js_tr[-1]:.4f}")
    worst = sorted(rows, key=lambda r: -r["js_gated_vs_clean"].get("luminosity_distance", 0))[:5]
    for r in worst:
        print(f"  {r['id']}: JS {r['js_gated_vs_clean']['luminosity_distance']:.3f}, n_eff {r['gated_is']['n_eff']:.1f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
