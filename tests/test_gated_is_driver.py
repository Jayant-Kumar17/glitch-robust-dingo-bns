"""Driver logic of examples/gated_is_stage2.py on a small fake model (no DINGO needed).

Checks the real-data likelihood checks, the route-B cache, multiprocessing, and the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bilby = pytest.importorskip("bilby")

FS, T, TB, F_MAX = 1024.0, 16.0, 2.0, 400.0
N = int(FS * T)
N_FREQ = int(F_MAX * T) + 1
F = np.arange(N_FREQ) / T
ASD = 1e-23 * (1 + (30 / np.maximum(F, 1)) ** 4) * (1 + (F / 200) ** 2)
ASD[F < 23] = 1.0
NS = 1.0 / np.sqrt(4.0 / T)


def _chirp(dl, phase):
    from adapt.packaging import package_fd

    t = np.arange(N) / FS
    tc = T - TB
    tau = np.clip(tc - t, 1e-3, None)
    fr = np.clip(40 * (tau / 4) ** (-3 / 8), 0, 350)
    x = (40.0 / dl) * 2e-22 * (fr / 40) ** (2 / 3) * np.cos(np.cumsum(2 * np.pi * fr / FS) + phase) * (t < tc - 0.002)
    return np.where(ASD < 1e-6, package_fd(x, FS, f_max=F_MAX, n_freq=N_FREQ, time_buffer=TB) / (ASD * NS), 0.0)


class FakeBase:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        from adapt.packaging import package_fd

        x = np.fft.irfft(
            (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1))
            * np.concatenate([np.where(ASD < 1e-6, ASD, 0.0), np.zeros(N // 2 + 1 - N_FREQ)]) * np.sqrt(N * FS) / 2,
            n=N,
        )
        noise = np.where(ASD < 1e-6, package_fd(x, FS, f_max=F_MAX, n_freq=N_FREQ, time_buffer=TB) / (ASD * NS), 0.0)
        self.whitened_strains = {"H1": noise + _chirp(40.0, 0.3)}
        self.asd = {"H1": ASD.copy()}
        self.data_domain = SimpleNamespace(noise_std=NS)

    def signal(self, theta):
        return {"waveform": {"H1": _chirp(float(theta["luminosity_distance"]), float(theta["phase"]))}}

    def log_likelihood(self, theta):
        mu = self.signal(theta)["waveform"]["H1"]
        return float(-0.5 * np.sum(np.abs(self.whitened_strains["H1"] - mu) ** 2))


class FakeResult:
    def __init__(self, base, n=200, seed=1):
        from bilby.core.prior import Uniform

        rng = np.random.default_rng(seed)
        dl = rng.normal(40.0, 3.0, n)
        ph = rng.normal(0.3, 0.2, n)
        self.prior = {"luminosity_distance": Uniform(10, 100), "phase": Uniform(0, 2 * np.pi)}
        log_prob = -0.5 * ((dl - 40) / 3) ** 2 - 0.5 * ((ph - 0.3) / 0.2) ** 2
        self.samples = pd.DataFrame({"luminosity_distance": dl, "phase": ph, "log_prob": log_prob,
                                     "log_prior": np.zeros(n)})
        L = np.array([base.log_likelihood(r) for r in self.samples.to_dict("records")])
        self.samples["log_likelihood"] = L
        lw = L - log_prob
        w = np.exp(lw - lw.max())
        self.samples["weights"] = w / w.mean()
        self.context = {"waveform": dict(base.whitened_strains)}
        self.sample_efficiency = float(w.sum() ** 2 / (w ** 2).sum() / n)
        self._base = base

    def _build_likelihood(self, decimate=False, phase_heterodyning=False):
        self.likelihood = self._base


def _ctx():
    import gated_is_stage2 as S  # noqa: F401

    c = SimpleNamespace(fs=FS, T=T, tb=TB, N=N, t_trig=T - TB, roll_off=0.4, f_max=F_MAX,
                        args=SimpleNamespace(eps=1e-3, eps_sensitivity=[1e-2, 1e-4], num_processes=1))
    return c


def test_likelihood_checks_pass_on_consistent_model():
    import gated_is_stage2 as S

    base = FakeBase()
    res = FakeResult(base)
    out = S.likelihood_checks(_ctx(), res, base)
    assert out["B_pass"], out
    assert out["A_max_rel_diff_vs_dingo_ufd"] < 1e-10


def test_route_b_cache_equals_direct_correction():
    import gated_is_stage2 as S
    from adapt.gated_likelihood import GatedLikelihood, gate_mask_from_windows, theta_frame
    from adapt.glitch_excision import GateWindow

    c = _ctx()
    base = FakeBase()
    res = FakeResult(base, n=30)
    cache = S.build_cacheB(c, res, base)
    wins = [GateWindow("H1", c.t_trig - 1.3, c.t_trig - 0.9)]
    ops = S.gate_ops(c, base, wins, [1e-3])[1e-3]
    corr_cache = S.corrections(ops, {"H1": cache["u"]}, {"H1": np.arange(cache["lo"], cache["hi"])})
    lik = GatedLikelihood(base, {"H1": gate_mask_from_windows(wins, "H1", N, FS)},
                          sample_rate=FS, duration=T, time_buffer=TB, eps=1e-3)
    direct = np.array([lik.components(th)[1] for th in theta_frame(res).to_dict("records")])
    assert np.allclose(corr_cache, direct, rtol=1e-9, atol=1e-9)
    L0_direct = np.array([lik.components(th)[0] for th in theta_frame(res).to_dict("records")])
    assert np.allclose(cache["L0"], L0_direct, rtol=1e-12)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork-only pickling of the fake model")
def test_multiprocessing_matches_serial():
    import gated_is_stage2 as S
    from adapt.gated_likelihood import theta_frame

    c = _ctx()
    base = FakeBase()
    res = FakeResult(base, n=24)
    payload = S._Payload(base, S.unit_ops(c, base, ["H1"]), {"H1": np.arange(1000, 1200)})
    L1, u1 = S.evaluate(payload, theta_frame(res), 1)
    L2, u2 = S.evaluate(payload, theta_frame(res), 2)
    assert np.allclose(L1, L2) and np.allclose(u1["H1"], u2["H1"])


def test_verdict_go_and_nogo(tmp_path):
    import gated_is_stage2 as S

    def row(i, eff, rec, js):
        return {"id": f"c{i}", "kind": "clean", "half": 0.4,
                "network_only": {"recovers": False}, "naive_is": {"recovers": False},
                "gated_is_A": {"efficiency": eff, "n_eff": eff * 1e4, "recovers": rec, "d_L": {"med": 34.0}},
                "gated_is_B": {"n_eff": 500.0}, "js_A_vs_B": {"luminosity_distance": js},
                "js_A_vs_clean": {"luminosity_distance": 0.01},
                "eps_sensitivity_A": {"0.001": {"d_L_med": 34.0}, "0.01": {"d_L_med": 34.2}}}

    checks = {"B_pass": True, "A_max_rel_diff_vs_dingo_ufd": 0.0, "B": {}}
    good = [row(i, 0.05, True, 0.01) for i in range(10)]
    assert "## Decision: GO" in S.verdict(tmp_path, good, checks)
    bad = [row(i, 0.001, False, 0.3) for i in range(10)]
    assert "## Decision: NO-GO" in S.verdict(tmp_path, bad, checks)


def test_run_cell_end_to_end_on_fake_model(tmp_path, monkeypatch):
    """run_cell plumbing with the network draw and DINGO IS replaced by the fake model."""
    import gated_is_stage2 as S
    import adapt.gated_likelihood as G

    c = _ctx()
    c.args.num_samples = 150
    rng = np.random.default_rng(3)
    x = np.fft.irfft(
        (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1))
        * np.concatenate([np.where(ASD < 1e-6, ASD, 0.0), np.zeros(N // 2 + 1 - N_FREQ)]) * np.sqrt(N * FS) / 2,
        n=N,
    )
    c.td = {"H1": x}
    c.asds = {"H1": ASD.copy()}
    c.event = SimpleNamespace(data={"waveform": {"H1": np.zeros(N_FREQ, complex)}, "asds": {"H1": ASD.copy()}})
    base = FakeBase()
    clean = FakeResult(base, n=150, seed=5)

    def fake_draw(ctx, data, n):
        res = FakeResult(base, n=n, seed=7)
        return res, res.samples.copy()

    monkeypatch.setattr(S, "draw", fake_draw)
    monkeypatch.setattr(G, "build_ufd_likelihood", lambda res: res._base)
    cache = S.build_cacheB(c, clean, base)
    clean_ci = S._ci(clean.samples["luminosity_distance"].to_numpy(float), clean.samples["weights"].to_numpy(float))
    cell = {"id": "clean_h0.4_00", "kind": "clean", "half": 0.4, "t_rel": -1.1}
    rec = S.run_cell(c, cell, clean_ci, clean, cache, tmp_path)
    for k in ("network_only", "naive_is", "gated_is_A", "gated_is_B", "js_A_vs_B", "js_A_vs_clean", "eps_sensitivity_A"):
        assert k in rec, k
    assert 0 < rec["gated_is_A"]["efficiency"] <= 1
    # same proposal family and same gated likelihood -> both routes agree closely here
    assert abs(rec["gated_is_A"]["d_L"]["med"] - rec["gated_is_B"]["d_L"]["med"]) < 3.0
    assert (tmp_path / "clean_h0.4_00_samples.csv.gz").is_file()


def test_raw_cell_path_and_empty_gate(tmp_path, monkeypatch):
    import gated_is_stage2 as S
    import adapt.gated_likelihood as G
    from adapt.glitch_excision import GateWindow

    c = _ctx()
    c.args.num_samples = 120
    c.td = {"H1": np.zeros(N)}
    c.asds = {"H1": ASD.copy()}
    c.event = SimpleNamespace(data={"waveform": {"H1": np.zeros(N_FREQ, complex)}, "asds": {"H1": ASD.copy()}})
    base = FakeBase()
    clean = FakeResult(base, n=120, seed=5)
    monkeypatch.setattr(S, "draw", lambda ctx, data, n: (lambda r: (r, r.samples.copy()))(FakeResult(base, n=n, seed=9)))
    monkeypatch.setattr(G, "build_ufd_likelihood", lambda res: res._base)
    clean_ci = S._ci(clean.samples["luminosity_distance"].to_numpy(float), clean.samples["weights"].to_numpy(float))
    cell = {"id": "rawL1_h0.1", "kind": "raw_l1", "half": 0.1, "data": c.event.data, "td": c.td,
            "windows": [GateWindow("H1", c.t_trig - 1.2, c.t_trig - 1.0)]}
    rec = S.run_cell(c, cell, clean_ci, clean, None, tmp_path)
    assert "gated_is_A" in rec and "gated_is_B" not in rec
    cell_empty = dict(cell, id="rawL1_none", windows=[])
    with pytest.raises(ValueError):
        S.run_cell(c, cell_empty, clean_ci, clean, None, tmp_path)
