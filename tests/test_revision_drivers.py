"""Unit tests for the referee-revision drivers. No DINGO sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_phase_b_placements_match_stage2_draws():
    from leakage_check import phase_b_trels

    rng = np.random.default_rng(0)
    expected = [float(rng.uniform(-2.0, -0.2)) for _ in range(10)]
    assert phase_b_trels(10, 0) == expected
    assert all(-2.0 <= t <= -0.2 for t in expected)


def test_whitened_norm_one_bin():
    from leakage_check import whitened_norm

    df = 1.0
    delta = np.zeros(200, dtype=complex)
    asd = np.full(200, 1e-23)
    delta[100] = 1.0
    rho = whitened_norm(delta, asd, df, 23.0, 150.0)
    expect = np.sqrt(4.0 * df * (1.0 / 1e-23 ** 2))
    assert rho == pytest.approx(expect, rel=1e-12)
    # a placeholder ASD (the 1.0 DINGO writes below f_min) is not in the sum
    asd[:] = 1.0
    assert np.isnan(whitened_norm(delta, asd, df, 23.0, 150.0))


def test_classify_cells_and_default_groups():
    import gated_is_stage3 as S3

    real = REPO / "results" / "stress_real_glitches_v3_final"
    if not (real / "results.csv").is_file():
        pytest.skip("real panel not archived")
    allc = S3.classify_cells(real)
    assert int((allc["group"] == "damaged").sum()) == 41
    assert int((allc["group"] == "broken").sum()) == 14
    assert int((allc["group"] == "fine").sum()) == 140
    sel = S3.select_cells(real)
    sub = allc[allc["group"].isin(["damaged", "broken"])]
    assert list(sub["cell_id"]) == list(sel["cell_id"])
    assert list(sub["group"]) == list(sel["group"])
    args = S3.build_parser().parse_args([])
    assert args.groups == ["damaged", "broken"]


def test_route_cell_rule():
    from revision_routes import choose_route_cells

    rows = []
    # two trusted failures
    for i, grp in ((1, "damaged"), (2, "broken")):
        rows.append({"id": f"{grp}_{i:03d}_h0.1", "group": grp, "half": 0.1, "cell_id": i,
                     "gated_is": {"trusted": True, "recovers": False, "n_eff": 500}})
    # five trusted passing damaged, smallest n_eff should be 10, 20, 30
    for i, ne in enumerate((10, 20, 30, 40, 50), start=10):
        rows.append({"id": f"damaged_{i:03d}_h0.2", "group": "damaged", "half": 0.2, "cell_id": i,
                     "gated_is": {"trusted": True, "recovers": True, "n_eff": ne}})
    # four untrusted damaged 0.1, largest n_eff 90, 80, 70
    for i, ne in enumerate((90, 80, 70, 5), start=20):
        rows.append({"id": f"damaged_{i:03d}_h0.1", "group": "damaged", "half": 0.1, "cell_id": i,
                     "gated_is": {"trusted": False, "recovers": True, "n_eff": ne}})
    ids = [r["id"] for r in choose_route_cells(rows)]
    assert ids == [
        "broken_002_h0.1", "damaged_001_h0.1",
        "damaged_010_h0.2", "damaged_011_h0.2", "damaged_012_h0.2",
        "damaged_020_h0.1", "damaged_021_h0.1", "damaged_022_h0.1",
    ]


def test_r7_pass_rule_and_distance_scale():
    from synthetic_panel import r7_passes, scale_distance

    clean = {"lo": 20.0, "med": 40.0, "hi": 60.0}  # width 40, tolerance 10
    assert r7_passes({"lo": 30.0, "med": 50.0, "hi": 70.0}, clean)
    assert not r7_passes({"lo": 30.0, "med": 50.1, "hi": 70.0}, clean)
    assert not r7_passes({"lo": 61.0, "med": 40.0, "hi": 80.0}, clean)
    assert scale_distance(40.0, 64.0, 32.0) == pytest.approx(80.0)


def test_smoke_flags_are_opt_in():
    import clean_ref_v2
    import revision_routes
    import ungated_is_baseline
    from revision_eval import apply_smoke

    for mod, dest in (
        (clean_ref_v2, "clean_ref_v2"),
        (ungated_is_baseline, "ungated_is"),
        (revision_routes, "routes"),
    ):
        full = mod.build_parser().parse_args([])
        assert full.smoke is False
        assert full.num_samples in (10_000, 100_000)
        assert dest in str(full.outdir)
        smoke = mod.build_parser().parse_args(["--smoke", "--num-processes", "1"])
        apply_smoke(smoke)
        assert smoke.num_samples == 300 and smoke.num_processes == 1
    stage3 = __import__("gated_is_stage3")
    assert stage3.build_parser().parse_args([]).groups == ["damaged", "broken"]
    assert stage3.build_parser().parse_args(["--groups", "fine"]).groups == ["fine"]


def test_chunked_gated_concatenates(monkeypatch):
    import pandas as pd

    import revision_eval as rv

    class Args:
        eps = 1e-3
        chunk = 3
        num_processes = 1

    calls = []

    def fake_gate_ops(c, base, wins, eps_list):
        return {eps_list[0]: {"H1": type("Op", (), {"idx": np.array([0])})()}}

    def fake_unit(c, base, ifos):
        return {}

    def fake_eval(payload, theta, nproc):
        calls.append(len(theta))
        return np.ones(len(theta)), {"H1": np.zeros((len(theta), 1))}

    def fake_corr(ops, u, windows):
        return np.full(len(next(iter(u.values()))), 2.0)

    import gated_is_stage2 as S2

    class _P:
        def __init__(self, base, units, windows):
            self.windows = windows

    monkeypatch.setattr(S2, "_Payload", _P)
    monkeypatch.setattr(S2, "gate_ops", fake_gate_ops)
    monkeypatch.setattr(S2, "unit_ops", fake_unit)
    monkeypatch.setattr(S2, "evaluate", fake_eval)
    monkeypatch.setattr(S2, "corrections", fake_corr)
    c = type("C", (), {"args": Args()})()
    theta = pd.DataFrame({"x": np.arange(7)})
    out = rv.chunked_gated(c, base=None, wins=[], theta=theta)
    assert calls == [3, 3, 1]
    assert np.allclose(out, 3.0)
