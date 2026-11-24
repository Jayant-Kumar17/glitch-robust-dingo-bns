"""Tests for examples/revision_analysis.py on synthetic inputs (no DINGO needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import revision_analysis as ra  # noqa: E402

P = ["delta_chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn",
     "luminosity_distance", "geocent_time", "psi", "lambda_1", "lambda_2", "chirp_mass"]


def test_khat_recovers_pareto_shape():
    for a in (3.0, 1.5, 0.8):
        ks = [ra.psis_khat(np.log(np.random.default_rng(s).pareto(a, 10000) + 1)) for s in range(20)]
        assert abs(np.mean(ks) - 1 / a) < 0.06, (a, np.mean(ks))


def test_lambda_tilde_equal_mass():
    df = pd.DataFrame({"chirp_mass": [1.2], "mass_ratio": [1.0], "lambda_1": [400.0], "lambda_2": [400.0]})
    assert abs(ra.lambda_tilde(df)[0] - 400.0) < 1e-9


def test_mcnemar_and_wilson():
    assert abs(ra.mcnemar_exact(0, 5) - 0.0625) < 1e-12
    lo, hi = ra.wilson(32, 41)
    assert round(100 * lo) == 63 and round(100 * hi) == 88


def _fake(root: Path, rng):
    def samp(n=1500):
        d = {p: rng.normal(size=n) for p in P}
        d["chirp_mass"] = 1.1975 + 1e-4 * rng.normal(size=n)
        d["mass_ratio"] = rng.uniform(0.7, 1, n)
        d["lambda_1"], d["lambda_2"] = rng.uniform(0, 800, n), rng.uniform(0, 1200, n)
        d["luminosity_distance"] = rng.normal(34, 8, n)
        return pd.DataFrame(d)
    (root / "real").mkdir(parents=True)
    cells = list(range(24))
    pd.DataFrame({"cell_id": cells, "gs_event_time": [c // 3 for c in cells]}).to_csv(root / "real/results.csv", index=False)
    for d in ("s3/cells", "fine/cells", "ung/cells", "routes/cells", "ref", "s3b/cells"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "s3/config.json").write_text(json.dumps({"num_samples": 10000}))
    for cid in cells:
        grp, folder = ("damaged", "s3") if cid < 8 else (("broken", "s3") if cid < 12 else ("fine", "fine"))
        for h in ((0.1, 0.2) if folder == "s3" else (0.1,)):
            rid = f"{grp}_{cid:03d}_h{h:g}"
            ne = float(rng.uniform(1, 900))
            r = {"id": rid, "group": grp, "cell_id": cid, "half": h, "label": "Blip", "snr_bin": "x",
                 "rho_w_archived": 1.0, "injection_match": True, "gate_total_s": 0.3,
                 "gated_is": {"n_eff": ne, "efficiency": ne / 1e4, "trusted": ne >= 100, "recovers": True,
                              "d_L": {"lo": 20, "med": 34, "hi": 46}},
                 "naive_is": {"efficiency": 0.01, "recovers": False, "d_L": {"lo": 1, "med": 2, "hi": 3}},
                 "network_only": {"recovers": bool(cid % 2)}}
            (root / folder / "cells" / f"{rid}.json").write_text(json.dumps(r))
            df = samp()
            df["weight_gated"], df["weight_naive"] = rng.exponential(size=len(df)), rng.exponential(size=len(df))
            df.to_csv(root / folder / f"{rid}_samples.csv.gz", index=False)
        (root / "ung/cells" / f"{cid:03d}.json").write_text(json.dumps(
            {"cell_id": cid, "injection_match": True, "efficiency": 0.02, "n_eff": 200.0,
             "d_L": {"lo": 20, "med": 34, "hi": 46}, "recovers": True, "network_only": {"recovers": True}}))
        df = samp()
        df["weight"] = rng.exponential(size=len(df))
        df.to_csv(root / "ung" / f"{cid:03d}_samples.csv.gz", index=False)
    ref = samp(5000)
    ref["weights"] = rng.exponential(size=len(ref))
    ref.to_csv(root / "ref/samples.csv.gz", index=False)
    (root / "ref/meta.json").write_text(json.dumps({"n": 5000, "d_L": {"lo": 19, "med": 34.2, "hi": 46}}))


def test_end_to_end(tmp_path):
    _fake(tmp_path, np.random.default_rng(1))
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "examples/revision_analysis.py"), "--smoke",
           "--stage3", str(tmp_path / "s3"), "--stage3b", str(tmp_path / "s3b"), "--fine", str(tmp_path / "fine"),
           "--ungated", str(tmp_path / "ung"), "--ref", str(tmp_path / "ref"), "--routes", str(tmp_path / "routes"),
           "--real-csv", str(tmp_path / "real/results.csv"), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    text = (out / "REVISION_REPORT.md").read_text()
    for sec in ("## 1. Flag tables", "## 2. Paired comparisons", "## 4. PSIS k-hat", "## 6. JS against"):
        assert sec in text
    S = json.loads((out / "revision_summary.json").read_text())
    gis_d = [t for t in S["flag_table"] if t["method"] == "GIS" and t["group"] == "damaged" and t["pad"] == 0.1][0]
    assert gis_d["n"] == 8 and gis_d["trusted_fail"] == 0
    assert all(np.isfinite(j["js"]["lambda_tilde"]) for j in S["js"])
