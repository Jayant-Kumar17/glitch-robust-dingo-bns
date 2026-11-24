"""Stage 3 driver logic on archived tables and a fake model (no DINGO needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("bilby")


def _real_dir():
    for name in ("stress_real_glitches_v3_final", "stress_real_glitches_v2"):
        d = REPO / "results" / name
        if (d / "results.csv").is_file() and (d / "clean_reference.json").is_file():
            return d
    pytest.skip("no real-panel results archived")


def test_select_cells_matches_definitions():
    import gated_is_stage3 as S3

    sel = S3.select_cells(_real_dir())
    assert set(sel["group"]) <= {"damaged", "broken"}
    assert (sel.loc[sel["group"] == "broken", "gated_recovers"] == False).all()  # noqa: E712
    if _real_dir().name == "stress_real_glitches_v2":
        assert int((sel["group"] == "damaged").sum()) == 41
        assert int((sel["group"] == "broken").sum()) == 2


def test_run_cell_and_report_on_fake_model(tmp_path, monkeypatch):
    import gated_is_stage2 as S2
    import gated_is_stage3 as S3
    import adapt.gated_likelihood as G
    from test_gated_is_driver import ASD, F_MAX, FS, N, N_FREQ, T, TB, FakeBase, FakeResult, _ctx

    c = _ctx()
    c.args.num_samples = 120
    c.asds = {"H1": ASD.copy()}
    rng = np.random.default_rng(4)
    x = np.fft.irfft((rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1))
                     * np.concatenate([np.where(ASD < 1e-6, ASD, 0.0), np.zeros(N // 2 + 1 - N_FREQ)]) * np.sqrt(N * FS) / 2, n=N)
    t = np.arange(N) / FS
    tc = T - TB - 1.0
    burst = 40 * np.std(x) * np.sin(2 * np.pi * 120 * (t - tc)) * np.exp(-((t - tc) / 0.02) ** 2)
    base = FakeBase()
    clean = FakeResult(base, n=120, seed=5)
    monkeypatch.setattr(S2, "draw", lambda ctx, data, n: (lambda r: (r, r.samples.copy()))(FakeResult(base, n=n, seed=11)))
    monkeypatch.setattr(G, "build_ufd_likelihood", lambda res: res._base)

    def inj(row):
        return ({"waveform": {"H1": np.zeros(N_FREQ, complex)}, "asds": {"H1": ASD.copy()}},
                {"injected_rho_w": float(row["injected_rho_w"]), "td_full": {"H1": x + burst}})

    row = {"group": "damaged", "cell_id": 7, "gs_label": "Blip", "gs_snr_bin": "[100,300)", "injected_rho_w": 200.0,
           "t_rel": -1.0, "gated_recovers": False, "poison_collapsed": True}
    clean_ci = S2._ci(clean.samples["luminosity_distance"].to_numpy(float), clean.samples["weights"].to_numpy(float))
    rec = S3.run_cell(c, inj, row, 0.1, clean_ci, clean, tmp_path)
    assert rec["injection_match"] and rec["gates"], rec
    assert "gated_is" in rec and rec["gated_is"]["trusted"] in (True, False)
    # a mismatched injection must be refused, not analysed
    bad = S3.run_cell(c, lambda r: (inj(r)[0], {"injected_rho_w": 999.0, "td_full": {"H1": x}}), row, 0.1, clean_ci, clean, tmp_path)
    assert "error" in bad and "mismatch" in bad["error"]
    text = S3.report(tmp_path, [rec, dict(rec, id="damaged_007_h0.2", half=0.2), bad],
                     {"clean_ci": clean_ci, "clean_eff": 0.1})
    assert "gated IS (trusted)" in text and "Adaptive rule" in text and "1 mismatched" in text
