#!/usr/bin/env python3
"""R7, optional: two synthetic events in Gaussian noise, SNR about 32 and about 16.

The injection is the maximum-weight R1 sample. If the ungated DINGO-IS known-answer
test fails, the reason is written to results/revision/BLOCKERS.md and this script stops.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

from _repo import REPO_ROOT, resolve_under_repo

logger = logging.getLogger("synthetic_panel")


def r7_passes(ci: dict, clean: dict) -> bool:
    """d_L median within 0.25 of the clean 90% width, and the intervals overlap."""
    width = float(clean["hi"]) - float(clean["lo"])
    overlap = not (ci["hi"] < clean["lo"] or clean["hi"] < ci["lo"])
    return bool(overlap and abs(float(ci["med"]) - float(clean["med"])) <= 0.25 * width)


def scale_distance(d_l: float, snr_actual: float, snr_target: float) -> float:
    return float(d_l) * float(snr_actual) / float(snr_target)


def build_parser() -> argparse.ArgumentParser:
    from revision_eval import add_runtime_args

    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(p, default_outdir="results/revision/synthetic", default_samples=10_000)
    p.add_argument("--ref", type=resolve_under_repo, default="results/revision/clean_ref_v2")
    return p


def _strain_td(mu_w, asd, noise_std: float, fs: float, tb: float, n: int) -> np.ndarray:
    """Invert the package convention: mu_w * (asd * noise_std), undo +2 s, irfft * fs."""
    m = np.asarray(mu_w)
    a = np.asarray(asd, dtype=np.float64).ravel()
    n_freq = m.size
    strain_fd = m * (a[:n_freq] * float(noise_std))
    df = fs / float(n)
    freqs = np.arange(n_freq) * df
    full = np.zeros(n // 2 + 1, dtype=np.complex128)
    full[:n_freq] = strain_fd * np.exp(+2j * np.pi * freqs * float(tb))
    return np.fft.irfft(full * float(fs), n=n)


def _snr(mu: dict) -> float:
    return float(np.sqrt(sum(np.sum(np.abs(np.asarray(v)) ** 2) for v in mu.values())))


def _event_from_td(c, td_by_det: dict):
    import copy

    from adapt.packaging import package_fd

    data = copy.deepcopy(c.event.data)
    n_freq = len(np.asarray(next(iter(data["waveform"].values()))))
    for ifo, td in td_by_det.items():
        data["waveform"][ifo] = package_fd(
            td, c.fs, roll_off=c.roll_off, f_max=c.f_max, n_freq=n_freq, time_buffer=c.tb,
        )
        data["asds"][ifo] = np.asarray(c.asds[ifo]).copy()
    return data


def _block(reason: str) -> None:
    path = REPO_ROOT / "results" / "revision" / "BLOCKERS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = path.read_text() if path.is_file() else ""
    path.write_text(prev + f"\n## R7\n\n{reason}\n")
    logger.error("R7 stopped: %s", reason)


def _known_answer(c, data, injected_d_l: float) -> tuple[dict, object]:
    import gated_is_stage2 as S2

    res, _raw = S2.draw(c, data, int(c.args.num_samples))
    w = res.samples["weights"].to_numpy(float)
    ci = S2._ci(res.samples["luminosity_distance"].to_numpy(float), w)
    eff = float(res.sample_efficiency)
    contains = bool(ci["lo"] <= injected_d_l <= ci["hi"])
    if eff < 0.01 or not contains:
        raise RuntimeError(
            f"known-answer failed: efficiency {eff:.6g} (need >= 0.01), "
            f"d_L 90% [{ci['lo']:.3f}, {ci['hi']:.3f}] contains injected {injected_d_l:.3f}: {contains}"
        )
    return {"efficiency": eff, "n_eff": eff * int(c.args.num_samples), "d_L": ci, "injected_d_L": injected_d_l}, res


def _gated_cell(c, data, td, t_rel: float, clean_ci: dict) -> dict:
    import gated_is_stage2 as S2
    from adapt.gated_likelihood import build_ufd_likelihood, theta_frame
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td
    from revision_eval import chunked_gated

    half = 0.2
    t_peak = float(c.t_trig) + float(t_rel)
    wins = [GateWindow("H1", t_peak - half, t_peak + half)]
    gated = rebuild_event_from_gated_td(
        data, td_by_det=td, gates=wins, sample_rate=c.fs, roll_off=c.roll_off, f_max=c.f_max,
        original_asds=c.asds, fill="zero", gate_band=(20.0, 23.0),
    )
    res, raw = S2.draw(c, gated.data, int(c.args.num_samples))
    net = S2._ci(raw["luminosity_distance"].to_numpy(float))
    w = res.samples["weights"].to_numpy(float)
    ois = S2._ci(res.samples["luminosity_distance"].to_numpy(float), w)
    base = build_ufd_likelihood(res)
    L = chunked_gated(c, base, wins, theta_frame(res))
    G = S2.summarise_route(res.samples, L, "gated")
    eff = float(res.sample_efficiency)
    n = int(c.args.num_samples)
    return {
        "t_rel": float(t_rel), "half": half,
        "network_only": {**net, "passes": r7_passes(net, clean_ci)},
        "ois": {"efficiency": eff, "n_eff": eff * n, "d_L": ois, "trusted": bool(eff * n >= 100),
                "passes": r7_passes(ois, clean_ci)},
        "gis": {"efficiency": G["efficiency"], "n_eff": G["n_eff"], "d_L": G["d_L"],
                "trusted": bool(G["n_eff"] >= 100), "passes": r7_passes(G["d_L"], clean_ci)},
    }


def _one_snr(c, base, theta, target: float, seed: int, outdir: Path) -> None:
    from adapt.packaging import coloured_noise_td
    from leakage_check import phase_b_trels

    d_l = float(theta["luminosity_distance"])
    mu = base.signal(theta)["waveform"]
    snr = _snr(mu)
    theta = dict(theta)
    theta["luminosity_distance"] = scale_distance(d_l, snr, target)
    mu = base.signal(theta)["waveform"]
    snr_now = _snr(mu)
    ns = float(base.data_domain.noise_std)
    td = {}
    for k, ifo in enumerate(("H1", "L1", "V1")):
        sig = _strain_td(mu[ifo], base.asd[ifo], ns, c.fs, c.tb, c.N)
        noise = coloured_noise_td(np.asarray(c.asds[ifo]), c.N, c.fs, seed=int(seed) + k)
        td[ifo] = sig + noise
    data = _event_from_td(c, td)
    try:
        clean, _res = _known_answer(c, data, float(theta["luminosity_distance"]))
    except RuntimeError as exc:
        _block(f"SNR target {target:g}, seed {seed}: {exc}")
        raise SystemExit(2)
    (outdir / "clean_reference.json").write_text(json.dumps(clean, indent=2))
    cells = outdir / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in cells.glob("*.json")}
    t0 = time.time()
    lines = [f"SNR target {target:g}, achieved {snr_now:.2f}, injected d_L {theta['luminosity_distance']:.3f} Mpc",
             f"ungated DINGO-IS efficiency {clean['efficiency']:.4f}, d_L median {clean['d_L']['med']:.2f}", ""]
    for i, t_rel in enumerate(phase_b_trels(10, 0)):
        cid = f"gate_{i:02d}"
        if cid in done:
            continue
        if (time.time() - t0) / 3600.0 > float(c.args.max_hours):
            logger.warning("time budget reached before %s", cid)
            lines.append(f"- stopped before {cid} (max-hours)")
            break
        try:
            rec = _gated_cell(c, data, td, t_rel, clean["d_L"])
        except Exception as exc:
            rec = {"t_rel": t_rel, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            logger.exception("synthetic cell %s failed", cid)
        rec["id"] = cid
        (cells / f"{cid}.json").write_text(json.dumps(rec, indent=2, default=float))
        if "gis" in rec:
            lines.append(f"- {cid}: GIS passes {rec['gis']['passes']} (n_eff {rec['gis']['n_eff']:.1f}), "
                         f"OIS passes {rec['ois']['passes']}, NET passes {rec['network_only']['passes']}")
    (outdir / "SYNTHETIC_REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    import gated_is_stage2 as S2
    from adapt.gated_likelihood import build_ufd_likelihood, theta_frame
    from dingo.gw.result import Result
    from revision_eval import apply_smoke

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    apply_smoke(args, cells=2)
    if args.smoke:
        logger.info("smoke: R7 known-answer is not part of the Phase 1 smoke list; exiting")
        return
    res = Result(file_name=str(Path(args.ref) / "result.hdf5"))
    w = res.samples["weights"].to_numpy(float)
    theta = theta_frame(res).iloc[int(np.argmax(w))].to_dict()
    c = S2.setup(args)
    base = build_ufd_likelihood(res)
    root = Path(args.outdir)
    for target, seed, name in ((32.0, 20261125, "snr32"), (16.0, 20261132, "snr16")):
        _one_snr(c, base, theta, target, seed, root / name)


if __name__ == "__main__":
    main()
