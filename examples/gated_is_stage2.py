#!/usr/bin/env python3
"""Stage 1 + 2: gate-aware importance sampling for the frozen DINGO-BNS GW170817 network.

Order of work (each step writes its outputs before the next starts, so a partial
overnight run still yields a decision):

  1. Likelihood checks on real data (hard stop if they fail):
       A  empty gate: our uniform-grid log L equals DINGO's own log L;
       B  a loud burst added inside the gate leaves the gated log L unchanged,
          while a gate moved by the 2 s cyclic buffer does not (negative control).
  2. Clean reference: network draw on clean GW170817 + DINGO importance sampling.
     These samples are also proposal "route B" for every cell.
  3. Cells: forced H1 gates on clean data at the Phase B placements, width 0.4 s first
     (the decisive set), then the raw unsubtracted Livingston frame, then 0.2 s, 0.8 s.
     Per cell:
       network only      raw samples on the gated package
       naive IS          DINGO IS with the ordinary likelihood on the gated package
       gated IS, route A network samples on the gated package, gated likelihood
       gated IS, route B clean-data samples, same gated likelihood (independent check)
  4. STAGE2_VERDICT.md after every cell, judged against STAGE2_CRITERIA.md.

Gate recipe (fixed from Phase B): band-limited strain (20-23 Hz ramp), zero fill.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from _repo import resolve_under_repo

logger = logging.getLogger("gated_is_stage2")
T_REL_RANGE = (-2.0, -0.2)
CACHE_WINDOW = (-3.0, 1.0)  # seconds relative to trigger, for the route-B cache
JS_N = 5000


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _wq(x, w, qs=(0.05, 0.5, 0.95)):
    from official_control import _weighted_quantiles

    return _weighted_quantiles(np.asarray(x, float), None if w is None else np.asarray(w, float), list(qs))


def _ci(x, w=None) -> dict:
    lo, med, hi = _wq(x, w)
    return {"lo": lo, "med": med, "hi": hi}


def _js(a_df, wa, b_df, wb, params) -> dict:
    from official_control import js_divergence_1d

    out = {}
    for p in params:
        if p in a_df.columns and p in b_df.columns:
            try:
                out[p] = float(
                    js_divergence_1d(a_df[p].to_numpy(float), b_df[p].to_numpy(float), wx=wa, wy=wb)
                )
            except Exception:
                out[p] = float("nan")
    return out


class _Payload:
    """What each worker needs: the uniform-grid likelihood and per-IFO index windows."""

    def __init__(self, base, units: dict, windows: dict):
        self.base = base
        self.units = units  # ifo -> GateOperator with empty gate (computes u~)
        self.windows = windows  # ifo -> np.ndarray of sample indices to return
        self.d_w = {k: np.asarray(v, dtype=np.complex128) for k, v in base.whitened_strains.items()}

    def __call__(self, theta):
        from adapt.gated_likelihood import whittle_log_likelihood

        mu = self.base.signal(theta)["waveform"]
        L0 = whittle_log_likelihood(self.d_w, mu)
        u = {}
        for ifo, unit in self.units.items():
            u[ifo] = unit.u_tilde(self.d_w[ifo] - np.asarray(mu[ifo]))[self.windows[ifo]].astype(np.float64)
        return L0, u


_P: dict = {}


def _init(payload):
    import threadpoolctl

    threadpoolctl.threadpool_limits(1)
    _P["p"] = payload


def _call(theta):
    try:
        return _P["p"](theta)
    except Exception:
        return float("nan"), None


def evaluate(payload: _Payload, theta_df: pd.DataFrame, num_processes: int):
    """Return L0 (n,) and {ifo: u~ window (n, len(window))} for every sample."""
    rows = [r.to_dict() for _, r in theta_df.iterrows()]
    t0 = time.time()
    if num_processes <= 1:
        res = []
        _P["p"] = payload
        for th in rows:
            res.append(_call(th))
    else:
        import multiprocessing as mp

        with mp.Pool(int(num_processes), initializer=_init, initargs=(payload,)) as pool:
            res = pool.map(_call, rows, chunksize=max(1, len(rows) // (8 * int(num_processes))))
    dt = time.time() - t0
    logger.info("uniform-grid likelihood: %d samples in %.1f s (%.1f ms/sample wall, %d processes)",
                len(rows), dt, 1e3 * dt / max(len(rows), 1), int(num_processes))
    L0 = np.array([r[0] for r in res], dtype=np.float64)
    n_bad = int(np.sum(~np.isfinite(L0)))
    if n_bad:
        logger.warning("%d of %d likelihood evaluations failed (weight set to zero)", n_bad, len(rows))
    u = {}
    for ifo, idx in payload.windows.items():
        arr = np.full((len(res), len(idx)), np.nan, dtype=np.float64)
        for i, r in enumerate(res):
            if r[1] is not None:
                arr[i] = r[1][ifo]
        u[ifo] = arr
    return L0, u


def corrections(ops: dict, u: dict, windows: dict) -> np.ndarray:
    """Sum over IFOs of the gate correction, from cached u~ windows.

    ``windows[ifo]`` holds the absolute sample indices of the columns of ``u[ifo]``.
    """
    n = next(iter(u.values())).shape[0]
    total = np.zeros(n)
    for ifo, op in ops.items():
        win = np.asarray(windows[ifo])
        pos = np.searchsorted(win, op.idx)
        if np.any(pos >= win.size) or not np.array_equal(win[np.minimum(pos, win.size - 1)], op.idx):
            raise ValueError(f"gate samples for {ifo} are not inside the cached window")
        block = u[ifo][:, pos]
        for i in range(n):
            if np.all(np.isfinite(block[i])):
                total[i] += op.correction_from_ug(block[i])
            else:
                total[i] = np.nan
    return total


# ---------------------------------------------------------------------------
# main pieces
# ---------------------------------------------------------------------------


class Ctx:
    pass


def setup(args) -> Ctx:
    from adapt.dingo_bns_demo import discover_assets, load_event_dataset
    from adapt.event_glitch_io import load_full_event_td
    from pe_session import PESession

    c = Ctx()
    c.args = args
    c.assets = discover_assets()
    c.event = load_event_dataset(c.assets)
    c.settings = dict(c.event.settings)
    c.fixed = dict(c.assets["fixed_context"])
    c.fs = float(c.settings.get("f_s", 4096.0))
    c.T = float(c.settings.get("T", 128.0))
    c.tb = float(c.settings.get("time_buffer", 2.0))
    c.roll_off = float(c.settings.get("roll_off", 0.4))
    c.f_max = float(c.settings.get("f_max", 1535.3046875))
    c.N = int(round(c.T * c.fs))
    c.t_trig = c.T - c.tb
    c.dets = ("H1", "L1", "V1")
    c.asds = {d: np.asarray(c.event.data["asds"][d]).copy() for d in c.dets}
    c.td = {}
    for d in c.dets:
        c.td[d], _, _ = load_full_event_td(c.assets, c.settings, d)
    c.session = PESession(Path(c.assets["baseline_ckpt"]))
    return c


def draw(c: Ctx, data: dict, n: int):
    from official_control import run_importance_sampling

    res = c.session.sample_result(data, c.settings, c.fixed, num_samples=n, batch_size=c.args.batch_size)
    raw = res.samples.copy()
    run_importance_sampling(res, num_processes=int(c.args.num_processes))
    return res, raw


def gate_ops(c: Ctx, base, windows, eps_list):
    from adapt.gated_likelihood import GateOperator, gate_mask_from_windows

    ns = float(base.data_domain.noise_std)
    out = {}
    for eps in eps_list:
        ops = {}
        for ifo in sorted({g.detector for g in windows}):
            mask = gate_mask_from_windows(windows, ifo, c.N, c.fs)
            ops[ifo] = GateOperator(np.asarray(base.asd[ifo]), ns, c.fs, c.T, c.tb, mask, eps=eps)
        out[eps] = ops
    return out


def unit_ops(c: Ctx, base, ifos):
    from adapt.gated_likelihood import GateOperator

    ns = float(base.data_domain.noise_std)
    return {
        ifo: GateOperator(np.asarray(base.asd[ifo]), ns, c.fs, c.T, c.tb, np.zeros(c.N, bool))
        for ifo in ifos
    }


def likelihood_checks(c: Ctx, clean_res, base) -> dict:
    """Checks A and B on real GW170817 data and real waveforms. Raises on failure."""
    from adapt.gated_likelihood import GatedLikelihood, theta_frame
    from adapt.glitch_excision import GateWindow
    from adapt.packaging import package_fd

    theta = theta_frame(clean_res)
    w = clean_res.samples.get("weights")
    order = np.argsort(-np.asarray(w)) if w is not None else np.arange(len(theta))
    picks = [theta.iloc[i].to_dict() for i in order[:12]]
    out = {"n_theta": len(picks)}

    # A: empty gate equals DINGO's own uniform-grid likelihood
    lik0 = GatedLikelihood(base, {}, sample_rate=c.fs, duration=c.T, time_buffer=c.tb)
    dA = []
    for th in picks:
        L0, corr = lik0.components(th)
        ref = float(base.log_likelihood(th))
        dA.append(abs(L0 - ref) / max(1.0, abs(ref)))
        assert corr == 0.0
    out["A_max_rel_diff_vs_dingo_ufd"] = float(max(dA))
    # also report the decimated/heterodyned likelihood DINGO used for IS (expected close, not exact)
    if "log_likelihood" in clean_res.samples.columns:
        L_dec = clean_res.samples["log_likelihood"].to_numpy(float)[order[:12]]
        L_ufd = np.array([lik0.components(th)[0] for th in picks])
        diff = (L_ufd - L_ufd.mean()) - (L_dec - L_dec.mean())
        out["A_decimated_vs_ufd_spread"] = float(np.max(np.abs(diff)))
    if out["A_max_rel_diff_vs_dingo_ufd"] > 1e-8:
        raise AssertionError(f"Check A failed: {out}")

    # B: invariance to a loud burst inside the gate, and the misplaced-gate control
    tc = c.t_trig - 1.0
    t = np.arange(c.N) / c.fs
    gate = [GateWindow("H1", tc - 0.2, tc + 0.2)]
    bad = [GateWindow("H1", tc - 0.2 + c.tb, tc + 0.2 + c.tb)]
    # compact 150 Hz burst, entirely inside the gate; scaled below to whitened SNR 1000
    burst = np.sin(2 * np.pi * 150.0 * (t - tc)) * np.exp(-(((t - tc) / 0.03) ** 2)) * (np.abs(t - tc) < 0.15)
    from adapt.gated_likelihood import gate_mask_from_windows

    res = {}
    for label, wins in (("gate", gate), ("misplaced", bad)):
        lik = GatedLikelihood(
            base, {"H1": gate_mask_from_windows(wins, "H1", c.N, c.fs)}, sample_rate=c.fs, duration=c.T, time_buffer=c.tb
        )
        d0 = dict(lik.d_w)
        n_freq = len(d0["H1"])
        g_fd = package_fd(burst, c.fs, roll_off=c.roll_off, f_max=c.f_max, n_freq=n_freq, time_buffer=c.tb)
        a = np.asarray(base.asd["H1"]) * float(base.data_domain.noise_std)
        g_w = np.where(np.asarray(base.asd["H1"]) < 1e-6, g_fd / a, 0.0)
        g_w = g_w * (1000.0 / max(float(np.sqrt(np.sum(np.abs(g_w) ** 2))), 1e-300))
        dG, dL = [], []
        for th in picks[:6]:
            lik.d_w = d0
            L0a, ca = lik.components(th)
            lik.d_w = dict(d0)
            lik.d_w["H1"] = d0["H1"] + g_w
            L0b, cb = lik.components(th)
            dG.append(abs((L0b + cb) - (L0a + ca)))
            dL.append(abs(L0b - L0a))
        lik.d_w = d0
        res[label] = {"max_delta_gated": float(max(dG)), "min_delta_ungated": float(min(dL))}
    out["B"] = res
    ok_gate = res["gate"]["min_delta_ungated"] > 100.0 and res["gate"]["max_delta_gated"] < 1e-3 * res["gate"]["min_delta_ungated"]
    ok_bad = res["misplaced"]["max_delta_gated"] > 0.1 * res["misplaced"]["min_delta_ungated"]
    out["B_pass"] = bool(ok_gate and ok_bad)
    if not out["B_pass"]:
        raise AssertionError(f"Check B failed: {out}")
    return out


def summarise_route(samples: pd.DataFrame, L: np.ndarray, label: str) -> dict:
    from adapt.gated_likelihood import reweight

    rw = reweight(samples, L)
    return {
        "label": label,
        "efficiency": rw["efficiency"],
        "n_eff": rw["n_eff"],
        "log_evidence": rw["log_evidence"],
        "d_L": _ci(samples["luminosity_distance"].to_numpy(float), rw["weights"]),
        "_weights": rw["weights"],
    }


def run_cell(c: Ctx, cell: dict, clean_ref_ci: dict, clean_res, cacheB, outdir: Path) -> dict:
    from adapt.gated_likelihood import build_ufd_likelihood, theta_frame
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td
    from official_control import INFERENCE_PARAMS
    from stress_gw170817 import _gated_recovers, _poison_collapsed

    t0 = time.time()
    rec = dict(cell)
    if cell["kind"] == "clean":
        t_peak = c.t_trig + cell["t_rel"]
        wins = [GateWindow("H1", t_peak - cell["half"], t_peak + cell["half"])]
        base_data, td_by_det = c.event.data, {"H1": c.td["H1"]}
    else:  # raw L1
        wins = cell["windows"]
        base_data, td_by_det = cell["data"], cell["td"]
    if not wins:
        raise ValueError("no gate for this cell (localiser did not fire)")
    gated = rebuild_event_from_gated_td(
        base_data, td_by_det=td_by_det, gates=wins, sample_rate=c.fs, roll_off=c.roll_off,
        f_max=c.f_max, original_asds=c.asds, fill="zero", gate_band=(20.0, 23.0),
    )
    rec["gates"] = [[g.detector, g.t_start - c.t_trig, g.t_end - c.t_trig] for g in wins]

    # route A proposal (network on the gated package) + naive IS
    resA, rawA = draw(c, gated.data, c.args.num_samples)
    rec["network_only"] = _ci(rawA["luminosity_distance"].to_numpy(float))
    rec["network_only"]["recovers"] = _gated_recovers(rec["network_only"], clean_ref_ci)
    rec["network_only"]["collapsed"] = _poison_collapsed(rec["network_only"])
    wN = resA.samples.get("weights")
    rec["naive_is"] = {
        "efficiency": float(resA.sample_efficiency) if resA.sample_efficiency is not None else None,
        "d_L": _ci(resA.samples["luminosity_distance"].to_numpy(float), None if wN is None else wN.to_numpy(float)),
    }
    rec["naive_is"]["recovers"] = _gated_recovers(rec["naive_is"]["d_L"], clean_ref_ci)

    baseA = build_ufd_likelihood(resA)
    eps_all = [c.args.eps] + [e for e in c.args.eps_sensitivity if e != c.args.eps]
    opsA = gate_ops(c, baseA, wins, eps_all)
    ifos = sorted(opsA[c.args.eps].keys())
    payload = _Payload(baseA, unit_ops(c, baseA, ifos), {i: opsA[c.args.eps][i].idx for i in ifos})
    thA = theta_frame(resA)
    L0A, uA = evaluate(payload, thA, int(c.args.num_processes))
    routeA = {}
    for eps in eps_all:
        corr = corrections(opsA[eps], uA, payload.windows)
        routeA[eps] = summarise_route(resA.samples, L0A + corr, f"A eps={eps:g}")
    A = routeA[c.args.eps]
    rec["gated_is_A"] = {k: v for k, v in A.items() if not k.startswith("_")}
    rec["gated_is_A"]["recovers"] = _gated_recovers(A["d_L"], clean_ref_ci)
    rec["eps_sensitivity_A"] = {
        f"{e:g}": {"d_L_med": routeA[e]["d_L"]["med"], "efficiency": routeA[e]["efficiency"]} for e in eps_all
    }

    # route B (clean-data proposal, same gated likelihood); only for gates inside the cache window
    if cell["kind"] == "clean" and cacheB is not None and all(g.detector == "H1" for g in wins):
        opsB = gate_ops(c, cacheB["base"], wins, [c.args.eps])[c.args.eps]
        if all(cacheB["lo"] <= op.idx.min() and op.idx.max() < cacheB["hi"] for op in opsB.values()):
            corrB = corrections(opsB, {"H1": cacheB["u"]}, {"H1": np.arange(cacheB["lo"], cacheB["hi"])})
            B = summarise_route(clean_res.samples, cacheB["L0"] + corrB, "B")
            rec["gated_is_B"] = {k: v for k, v in B.items() if not k.startswith("_")}
            rec["js_A_vs_B"] = _js(resA.samples, A["_weights"], clean_res.samples, B["_weights"], INFERENCE_PARAMS)
    # agreement with the clean reference posterior
    wC = clean_res.samples["weights"].to_numpy(float)
    rec["js_A_vs_clean"] = _js(resA.samples, A["_weights"], clean_res.samples, wC, INFERENCE_PARAMS)
    rec["elapsed_s"] = time.time() - t0
    # keep the weighted samples for later plots
    keep = [p for p in INFERENCE_PARAMS if p in resA.samples.columns]
    out = resA.samples[keep].copy()
    out["weight_gated"] = A["_weights"]
    if "weights" in resA.samples.columns:
        out["weight_naive"] = resA.samples["weights"].to_numpy(float)
    out.to_csv(outdir / f"{cell['id']}_samples.csv.gz", index=False)
    return rec


def build_cacheB(c: Ctx, clean_res, base):
    from adapt.gated_likelihood import theta_frame

    lo = int(round((c.t_trig + CACHE_WINDOW[0]) * c.fs))
    hi = int(round((c.t_trig + CACHE_WINDOW[1]) * c.fs))
    payload = _Payload(base, unit_ops(c, base, ["H1"]), {"H1": np.arange(lo, hi)})
    L0, u = evaluate(payload, theta_frame(clean_res), int(c.args.num_processes))
    return {"base": base, "L0": L0, "u": u["H1"], "lo": lo, "hi": hi}


def raw_l1_cells(c: Ctx) -> list:
    from adapt.glitch_excision import GateWindow  # noqa: F401
    from adapt.localize import excess_gates
    from adapt.dingo_bns_demo import load_bns_checkpoint
    from dingo.gw.domains import build_domain_from_model_metadata
    from dingo_conditioning import base_frequency_domain, data_to_domain_package, splice_package
    from gw170817_raw_l1 import _fetch_raw_l1

    trigger = float(c.assets["trigger_time"])
    start, end = trigger - (c.T - c.tb), trigger + c.tb
    raw = _fetch_raw_l1(start, end, Path(c.args.raw_l1_hdf5))
    meta = load_bns_checkpoint(Path(c.assets["baseline_ckpt"]))["metadata"]
    domain = base_frequency_domain(build_domain_from_model_metadata(meta, base=True))
    pkg = data_to_domain_package({"L1": raw}, {"L1": c.asds["L1"]}, domain)
    data = splice_package(c.event.data, pkg, {"L1": c.asds["L1"]})
    td = {"H1": c.td["H1"], "L1": raw, "V1": c.td["V1"]}
    cells = []
    for half in c.args.raw_l1_halfwidths:
        wins, zmax = excess_gates(td, c.asds, c.fs, f_max=c.f_max, pad_s=half, sigma=8.0)
        cells.append({"id": f"rawL1_h{half:g}", "kind": "raw_l1", "half": half, "windows": wins,
                      "data": data, "td": td, "zmax": zmax})
    return cells


def verdict(outdir: Path, rows: list, checks: dict) -> str:
    """Apply STAGE2_CRITERIA.md to the decisive width (0.4 s) and write the verdict."""
    dec = [r for r in rows if r.get("kind") == "clean" and abs(r.get("half", 0) - 0.4) < 1e-9 and "error" not in r]
    n = len(dec)
    lines = ["# Stage 2 verdict", ""]
    lines.append(f"Likelihood checks: A max rel diff {checks.get('A_max_rel_diff_vs_dingo_ufd')}, "
                 f"B {json.dumps(checks.get('B'), default=_json_default)} -> {'PASS' if checks.get('B_pass') else 'FAIL'}")
    if n == 0:
        lines.append("\nNo decisive (0.4 s) cells finished yet.")
        text = "\n".join(lines)
        (outdir / "STAGE2_VERDICT.md").write_text(text + "\n")
        return text
    effA = [r["gated_is_A"]["efficiency"] for r in dec]
    v2 = sum(e >= 0.01 for e in effA)
    both = [r for r in dec if "gated_is_B" in r and r["gated_is_A"]["n_eff"] >= 100 and r["gated_is_B"]["n_eff"] >= 100]
    v3 = sum(r["js_A_vs_B"].get("luminosity_distance", 9) <= 0.05 for r in both)
    v4 = sum(bool(r["gated_is_A"]["recovers"]) for r in dec)
    v1 = bool(checks.get("B_pass"))
    shifts = []
    for r in dec:
        if r["gated_is_A"]["n_eff"] >= 100:
            m = [v["d_L_med"] for v in r["eps_sensitivity_A"].values()]
            shifts.append(max(m) - min(m))
    base_net = sum(bool(r["network_only"]["recovers"]) for r in dec)
    base_naive = sum(bool(r["naive_is"]["recovers"]) for r in dec)
    need_v2 = int(np.ceil(0.7 * n))
    need_v4 = int(np.ceil(0.8 * n))
    ok2, ok4 = v2 >= need_v2, v4 >= need_v4
    ok3 = len(both) > 0 and v3 >= int(np.ceil(0.8 * len(both)))
    lines += [
        "",
        f"Decisive set: {n} clean cells with +-0.4 s H1 gates.",
        f"- V1 likelihood checks: {'PASS' if v1 else 'FAIL'}",
        f"- V2 route-A efficiency >= 1%: {v2}/{n} (need {need_v2}) -> {'PASS' if ok2 else 'FAIL'}",
        f"- V3 d_L JS(A,B) <= 0.05 among {len(both)} cells with n_eff >= 100 in both routes: {v3} -> {'PASS' if ok3 else 'FAIL'}",
        f"- V4 gated-IS d_L recovers: {v4}/{n} (need {need_v4}) -> {'PASS' if ok4 else 'FAIL'}",
        f"  (same cells, network only: {base_net}/{n}; naive IS: {base_naive}/{n})",
        f"- V5 (reported) eps sensitivity of d_L median, max over cells: {max(shifts) if shifts else 'n/a'} Mpc",
        f"- route-A efficiency: median {np.median(effA):.4f}, min {np.min(effA):.4g}, max {np.max(effA):.4f}",
    ]
    if v1 and ok2 and ok3 and ok4:
        decision = "GO"
    elif v1 and (ok2 or ok4):
        decision = "PARTIAL"
    else:
        decision = "NO-GO"
    lines += ["", f"## Decision: {decision}"]
    other = [r for r in rows if "error" not in r and not (r.get("kind") == "clean" and abs(r.get("half", 0) - 0.4) < 1e-9)]
    if other:
        lines += ["", "## Other cells", "", "| cell | network recovers | naive IS recovers | gated IS recovers | eff A | n_eff A | d_L med A | JS(A,B) d_L | JS(A,clean) d_L |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in other:
            lines.append(
                f"| {r['id']} | {r['network_only']['recovers']} | {r['naive_is']['recovers']} | {r['gated_is_A']['recovers']} | "
                f"{r['gated_is_A']['efficiency']:.4f} | {r['gated_is_A']['n_eff']:.0f} | {r['gated_is_A']['d_L']['med']:.1f} | "
                f"{r.get('js_A_vs_B', {}).get('luminosity_distance', float('nan')):.3f} | {r['js_A_vs_clean'].get('luminosity_distance', float('nan')):.3f} |"
            )
    errs = [r for r in rows if "error" in r]
    if errs:
        lines += ["", f"Cells with errors: {[r['id'] for r in errs]}"]
    text = "\n".join(lines)
    (outdir / "STAGE2_VERDICT.md").write_text(text + "\n")
    return text


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=resolve_under_repo, default="results/gated_is_stage2")
    p.add_argument("--widths", type=float, nargs="+", default=[0.4, 0.2, 0.8])
    p.add_argument("--n-cells", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=5_000)
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--eps-sensitivity", type=float, nargs="*", default=[1e-2, 1e-4])
    p.add_argument("--raw-l1-hdf5", type=str, default=None)
    p.add_argument("--raw-l1-halfwidths", type=float, nargs="+", default=[0.1, 0.4])
    p.add_argument("--max-hours", type=float, default=10.0, help="stop starting new cells after this")
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="300 samples, 1 cell, 0.4 s only: crash test")
    args = p.parse_args(argv)
    if args.smoke:
        args.num_samples, args.batch_size, args.n_cells = 300, 300, 1
        args.widths, args.raw_l1_halfwidths = [0.4], [0.1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cells").mkdir(exist_ok=True)
    t_start = time.time()
    _write(outdir / "config.json", vars(args))

    c = setup(args)
    from adapt.gated_likelihood import build_ufd_likelihood
    from stress_gw170817 import _gated_recovers  # noqa: F401

    # 1-2. clean reference (also proposal route B) and likelihood checks
    logger.info("Clean reference draw N=%d", args.num_samples)
    clean_res, clean_raw = draw(c, c.event.data, args.num_samples)
    wC = clean_res.samples["weights"].to_numpy(float)
    clean_ci = _ci(clean_res.samples["luminosity_distance"].to_numpy(float), wC)
    _write(outdir / "clean_reference.json", {
        "d_L": clean_ci, "efficiency": clean_res.sample_efficiency, "log_evidence": clean_res.log_evidence,
        "network_only_d_L": _ci(clean_raw["luminosity_distance"].to_numpy(float)),
    })
    baseC = build_ufd_likelihood(clean_res)
    try:
        checks = likelihood_checks(c, clean_res, baseC)
    except AssertionError as exc:
        checks = {"error": str(exc), "B_pass": False}
        _write(outdir / "likelihood_checks.json", checks)
        (outdir / "STAGE2_VERDICT.md").write_text(
            "# Stage 2 verdict\n\n## Decision: STOP\n\nThe gated likelihood failed its checks on real data:\n\n"
            f"```\n{exc}\n```\n\nDo not run anything else. Send this file back.\n"
        )
        logger.error("Likelihood checks failed: %s", exc)
        return
    _write(outdir / "likelihood_checks.json", checks)
    logger.info("Likelihood checks passed: %s", checks)
    if args.checks_only:
        verdict(outdir, [], checks)
        return

    logger.info("Building route-B cache")
    cacheB = build_cacheB(c, clean_res, baseC)

    # 3. cells
    rng = np.random.default_rng(int(args.seed))
    t_rels = [float(rng.uniform(*T_REL_RANGE)) for _ in range(max(args.n_cells, 20))]
    plan = []
    for i, half in enumerate(args.widths):
        cells = [{"id": f"clean_h{half:g}_{k:02d}", "kind": "clean", "half": half, "t_rel": t_rels[k]}
                 for k in range(args.n_cells)]
        plan.extend(cells)
        if i == 0 and args.raw_l1_hdf5:
            plan.append({"id": "__raw_l1__"})
    rows = []
    for f in sorted((outdir / "cells").glob("*.json")):
        rows.append(json.loads(f.read_text()))
    done = {r["id"] for r in rows}
    for cell in plan:
        if (time.time() - t_start) / 3600.0 > args.max_hours:
            logger.warning("Time budget reached; stopping before %s", cell["id"])
            break
        batch = [cell]
        if cell["id"] == "__raw_l1__":
            try:
                batch = raw_l1_cells(c)
            except Exception as exc:
                rows.append({"id": "rawL1", "kind": "raw_l1", "error": f"{type(exc).__name__}: {exc}"})
                logger.exception("raw L1 setup failed")
                continue
        for cl in batch:
            if cl["id"] in done:
                continue
            logger.info("=== cell %s ===", cl["id"])
            try:
                rec = run_cell(c, cl, clean_ci, clean_res, cacheB, outdir)
                for k in ("data", "td", "windows"):
                    rec.pop(k, None)
            except Exception as exc:
                rec = {"id": cl["id"], "kind": cl.get("kind"), "half": cl.get("half"),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                logger.exception("cell %s failed", cl["id"])
            _write(outdir / "cells" / f"{cl['id']}.json", rec)
            rows.append(rec)
            done.add(cl["id"])
            text = verdict(outdir, rows, checks)
            logger.info("verdict so far:\n%s", text)
    pd.json_normalize([{k: v for k, v in r.items() if k not in ("traceback",)} for r in rows]).to_csv(
        outdir / "cells.csv", index=False
    )
    verdict(outdir, rows, checks)
    logger.info("done in %.2f h", (time.time() - t_start) / 3600.0)


if __name__ == "__main__":
    main()
