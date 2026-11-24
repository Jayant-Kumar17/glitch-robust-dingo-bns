#!/usr/bin/env python3
"""Why did the real panel gain quiet excerpts between v2 and v3?

For every glitch that was unusable in v2 (native SNR 0) but usable in v3, report the
excerpt's window excess energy and denoised energy from the cached strain. If the
window excess is <= 0 and the denoised energy is > 0, the excerpt is usable only
through the denoised-energy fallback in GlitchExcerpt.native_energy_w, which the v2
code did not have. Writes results/excerpt_diagnosis.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _repo import REPO_ROOT


def main() -> None:
    from adapt import gravity_spy_io as gs

    old = pd.read_csv(REPO_ROOT / "results/stress_real_glitches_v2/glitch_catalogue.csv")
    new = pd.read_csv(REPO_ROOT / "results/stress_real_glitches_v3_final/glitch_catalogue.csv")
    o = old.set_index("gravityspy_id")["excerpt_native_snr"].fillna(0)
    n = new.set_index("gravityspy_id")
    ids = [i for i in n.index if float(n.loc[i, "excerpt_native_snr"] or 0) > 1 and float(o.get(i, 0) or 0) <= 1]
    out = []
    for gid in ids:
        row = n.loc[gid]
        t = float(row["event_time"])
        cache = REPO_ROOT / "data" / "gravity_spy" / "excerpts" / f"H1_{t:.4f}_4096.npy"
        rec = {"id": gid, "label": row["ml_label"], "omicron_snr": float(row["snr"]), "cached": cache.is_file()}
        if cache.is_file():
            exc = gs.extract_glitch_excerpt(np.load(cache), sample_rate=4096.0, duration_s=float(row["duration"]),
                                            f_min=23.0, f_max=1535.3046875)
            rec.update(excess_energy_w=float(exc.excess_energy_w), denoised_energy_w=float(exc.denoised_energy_w),
                       native_snr_now=float(np.sqrt(exc.native_energy_w())))
            rec["explained_by_fallback"] = bool(exc.excess_energy_w <= 0 < exc.denoised_energy_w)
        out.append(rec)
    summary = {"n": len(out), "explained_by_fallback": sum(bool(r.get("explained_by_fallback")) for r in out), "cells": out}
    (REPO_ROOT / "results" / "excerpt_diagnosis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}))
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
