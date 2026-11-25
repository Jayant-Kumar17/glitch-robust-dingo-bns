#!/usr/bin/env python3
"""Clean false-positive audit and component-isolated front-end timings.

False-positive audit (no parameter estimation):
  * 1000 stationary Gaussian draws coloured by the GW170817 H1/L1/V1 analysis ASDs
    (the first 500 use seed 0, the same sequence as the original audit; the
    second 500 use seed 1)
  * 500 off-source 4 s crops from the local GWOSC frames (the same open-data
    files the demonstration package was built from), excluding the 128 s
    on-source analysis window. These are reported separately: the frames are
    not a vetted glitch-free sample
  * STFT detector at the validation threshold 0.46 and the event threshold 0.73

Latency (CPU, and MPS for the front-end when available):
  * 10 warm-up runs, then 200 timed iterations of STFT, CNN+dilation, and
    matched-delta rebuild
  * 200 timed GWSamplerGNPE draws at num_samples=512
  * a shorter timed set at num_samples=20000 (default 10). Two hundred full
    20k draws are several hours on a single CPU; the script records the
    measured iteration time and the count actually run.

Outputs: results/latency_fp_audit_v1/summary.json and timings.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from _repo import REPO_ROOT, resolve_under_repo

THRESHOLDS = (0.46, 0.73)
N_GAUSS = 1000
N_OFF = 500
N_WARM = 10
N_ITER = 200
SAMPLE_RATE = 4096.0
CROP_S = 4.0
DETECTORS = ("H1", "L1", "V1")


def _rates(k: int, n: int) -> dict:
    from scipy.stats import beta, norm

    k = int(k)
    n = int(n)
    if n <= 0:
        return {"k": k, "n": n, "rate": float("nan")}
    # Clopper-Pearson exact interval.
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    z = float(norm.ppf(0.975))
    phat = k / n
    den = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / den
    half = z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / den
    return {
        "k": k,
        "n": n,
        "rate": phat,
        "clopper_pearson_95": [lo, hi],
        "wilson_95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
    }


def _summary(samples: Sequence[float]) -> dict:
    x = np.asarray(list(samples), dtype=np.float64)
    return {
        "n": int(x.size),
        "median_s": float(np.median(x)),
        "mean_s": float(np.mean(x)),
        "std_s": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p95_s": float(np.quantile(x, 0.95)),
    }


def _load_packaged_asds() -> tuple[dict, float]:
    import h5py
    from adapt.dingo_bns_demo import discover_assets

    assets = discover_assets()
    with h5py.File(assets["event_hdf5"], "r") as handle:
        asds = {
            det: np.asarray(handle["data"]["asds"][det], dtype=np.float64)
            for det in DETECTORS
        }
        # Packaged grid starts at f_min; df is 1/T.
        settings = assets
    # Domain df of the 128 s package.
    delta_f = 1.0 / 128.0
    return asds, delta_f, assets


def _analysis_noise_std() -> float:
    """DINGO base-domain noise standard deviation for the GW170817 window.

    ``noise_std = sqrt(window_factor) / sqrt(4 df)`` with the 128 s Tukey
    window (roll-off 0.4 s). The event whitener divides by ``ASD * noise_std``.
    """
    from dingo.gw.gwutils import get_window_factor

    window = {"type": "tukey", "roll_off": 0.4, "T": 128.0, "f_s": SAMPLE_RATE}
    factor = float(get_window_factor(window))
    return float(np.sqrt(factor) / np.sqrt(4.0 * (1.0 / 128.0)))


def _color_gaussian(
    asd: np.ndarray, n: int, rng: np.random.Generator, noise_std: float
) -> np.ndarray:
    """Stationary Gaussian strain coloured by the analysis ASD.

    Bilby convention on the 4 s crop: frequency-domain noise has standard
    deviation ``ASD / sqrt(4 df)`` and the time series is ``irfft(FD) * f_s``.
    Bins below 23 Hz are left empty. The packaged ASD equals 1 there (a
    padding value, not a spectrum); colouring it injects a seismic wall that
    leaks through the Tukey taper and is then amplified by the whitener.
    ``noise_std`` is the 128 s whitener scale and is not part of this draw.
    """
    del noise_std  # whitener applies this later; the draw itself is physical strain
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    # Packaged ASD bins run from 0 Hz at df = 1/128 Hz up to 1535.3 Hz.
    f_asd = np.arange(asd.size, dtype=np.float64) * (1.0 / 128.0)
    asd_i = np.interp(freqs, f_asd, asd, left=asd[0], right=asd[-1])
    asd_i = np.where((freqs >= 23.0) & (asd_i < 1e-6), asd_i, 0.0)
    n_freq = freqs.size
    real = rng.normal(size=n_freq)
    imag = rng.normal(size=n_freq)
    imag[0] = 0.0
    if n % 2 == 0:
        imag[-1] = 0.0
    df = SAMPLE_RATE / float(n)
    sigma = asd_i / np.sqrt(4.0 * df)
    return np.fft.irfft((real + 1j * imag) * sigma, n=n) * SAMPLE_RATE


def _read_frame(path: Path, channel: str) -> tuple[np.ndarray, float]:
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.read(str(path), channel=channel)
    if float(ts.sample_rate.value) != SAMPLE_RATE:
        ts = ts.resample(SAMPLE_RATE)
    return np.asarray(ts.value, dtype=np.float64), float(ts.t0.value)


def _offsource_crops(assets: dict, n_crops: int) -> list[dict]:
    """4 s crops from the GWOSC frames outside the 128 s on-source window."""
    trigger = float(assets["trigger_time"])
    on_start = trigger + 2.0 - 128.0
    on_end = trigger + 2.0
    frames = {}
    t0 = {}
    for det in DETECTORS:
        frames[det], t0[det] = _read_frame(assets["strain_gwf"][det], assets["channels"][det])
    n = int(round(CROP_S * SAMPLE_RATE))
    # Use H1's time base; the three frames share the GW170817 GPS start.
    start = t0["H1"]
    duration = frames["H1"].size / SAMPLE_RATE
    # Candidate starts every 2 s, dropping any crop that overlaps on-source.
    candidates = []
    t = start
    while t + CROP_S <= start + duration - 1.0 / SAMPLE_RATE:
        if t + CROP_S <= on_start or t >= on_end:
            candidates.append(t)
        t += 2.0
    if len(candidates) < n_crops:
        raise RuntimeError(f"only {len(candidates)} off-source crops available, need {n_crops}")
    pick = np.linspace(0, len(candidates) - 1, n_crops).astype(int)
    crops = []
    for idx in pick:
        t_crop = candidates[int(idx)]
        i0 = int(round((t_crop - start) * SAMPLE_RATE))
        crops.append({det: frames[det][i0 : i0 + n].copy() for det in DETECTORS})
    return crops


def _gaussian_draws(asds: dict, n_crop: int, noise_std: float, n: int) -> list[dict]:
    """ASD-coloured crops. Draws 0–499 use seed 0; draws 500+ use seed 1."""
    n = int(n)
    n_first = min(500, n)
    draws = []
    rng0 = np.random.default_rng(0)
    for _ in range(n_first):
        draws.append(
            {det: _color_gaussian(asds[det], n_crop, rng0, noise_std) for det in DETECTORS}
        )
    if n > 500:
        rng1 = np.random.default_rng(1)
        for _ in range(n - 500):
            draws.append(
                {det: _color_gaussian(asds[det], n_crop, rng1, noise_std) for det in DETECTORS}
            )
    return draws


def _score_population(model, draws, asds, stft_kwargs, norm_stats, noise_std, device) -> dict:
    specs = []
    for td in draws:
        spec, _energy = _spectrogram(td, asds, stft_kwargs, norm_stats, noise_std)
        spec_t = torch.from_numpy(np.asarray(spec, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            probs = model.predict_probs(spec_t.to(device)).detach().cpu().numpy()[0]
        specs.append(probs)
    out = {}
    for thr in THRESHOLDS:
        fires = sum(_fires(p, thr) for p in specs)
        out[str(thr)] = _rates(fires, len(specs))
    return out


def _spectrogram(td_map: dict, asds: dict, stft_kwargs: dict, norm_stats: dict, noise_std: float):
    from adapt.dingo_bns_demo import build_event_spectrogram_stack

    return build_event_spectrogram_stack(
        td_map,
        DETECTORS,
        SAMPLE_RATE,
        asds=asds,
        delta_f=1.0 / 128.0,
        noise_std=float(noise_std),
        robust=True,
        norm_stats=norm_stats,
        **stft_kwargs,
    )


def _fires(probs: np.ndarray, threshold: float) -> bool:
    return bool(np.any(probs > float(threshold)))


def _cnn_gates(model, spectrogram: np.ndarray, threshold: float, device: torch.device):
    from adapt.glitch_excision import time_bin_mask_to_windows
    from adapt.spectrogram_geometry import SPECTROGRAM_ANALYSIS_SECONDS

    spec_t = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        probs = model.predict_probs(spec_t.to(device)).detach().cpu().numpy()[0]
    n_crop = int(round(SPECTROGRAM_ANALYSIS_SECONDS * SAMPLE_RATE))
    gates = []
    for di, det in enumerate(DETECTORS):
        mask = probs[di] > float(threshold)
        if int(mask.sum()) == 0:
            continue
        gates.extend(
            time_bin_mask_to_windows(
                mask,
                n_samples=n_crop,
                sample_rate=SAMPLE_RATE,
                t0=0.0,
                detector=det,
                pad_s=0.4,
                scores=probs[di],
            )
        )
    return gates, probs


def _rebuild(event_data: dict, td_full: dict, asds: dict):
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td

    # A gate that actually exercises the matched-delta path (not the empty no-op).
    gates = [GateWindow(detector="H1", t_start=60.0, t_end=60.8, score=1.0)]
    return rebuild_event_from_gated_td(
        event_data,
        td_by_det=td_full,
        gates=gates,
        sample_rate=SAMPLE_RATE,
        roll_off=0.4,
        f_max=1535.3046875,
        t0=0.0,
        original_asds=asds,
        mode="matched_delta",
    )


def _time_loop(fn, n_warm: int, n_iter: int) -> list[float]:
    for _ in range(n_warm):
        fn()
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _pe_times(event, fixed: dict, ckpt: Path, num_samples: int, n_warm: int, n_iter: int) -> list[float]:
    from dingo.core.models import PosteriorModel
    from dingo.core.samplers import FixedInitSampler
    from dingo.gw.inference.gw_samplers import GWSamplerGNPE

    pm = PosteriorModel(model_filename=str(ckpt), device="cpu", load_training_info=False)
    batch = 512 if num_samples <= 512 else 5000

    def once():
        sampler = GWSamplerGNPE(
            model=pm,
            init_sampler=FixedInitSampler(fixed, log_prob=0.0),
            num_iterations=1,
            fixed_context_parameters=fixed,
        )
        sampler.context = event.data
        sampler.event_metadata = event.settings
        t0 = time.perf_counter()
        sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch))
        return time.perf_counter() - t0

    for _ in range(n_warm):
        once()
    return [once() for _ in range(n_iter)]


def _driver_overhead(td_h1: np.ndarray, n_warm: int, n_iter: int, outdir: Path) -> list[float]:
    """Injection plus one CSV append, the research-driver cost outside the front-end."""
    from adapt.glitch_augmentation import sine_gaussian_glitch

    n = td_h1.size
    path = outdir / "_overhead_probe.csv"
    samples = []
    rng = np.random.default_rng(1)

    def once(record: bool) -> None:
        t0 = time.perf_counter()
        glitch = sine_gaussian_glitch(
            n, SAMPLE_RATE, t_peak=n / SAMPLE_RATE * 0.5, f0=100.0, q=5.0, amplitude=1e-21
        )
        _ = td_h1 + glitch
        if record:
            pd.DataFrame({"dt": [time.perf_counter() - t0]}).to_csv(path, mode="a", header=not path.exists(), index=False)
        samples.append(time.perf_counter() - t0)

    for _ in range(n_warm):
        once(False)
    samples.clear()
    for _ in range(n_iter):
        once(True)
    if path.exists():
        path.unlink()
    return samples


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=resolve_under_repo, default="results/latency_fp_audit_v1")
    parser.add_argument("--pe-20000-repeats", type=int, default=10)
    parser.add_argument("--skip-pe", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-offsource", action="store_true")
    parser.add_argument("--n-gauss", type=int, default=N_GAUSS)
    parser.add_argument("--n-off", type=int, default=N_OFF)
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from examples.official_control import _load_detector
    from adapt.dingo_bns_demo import load_event_dataset, select_device
    from adapt.glitch_excision import analysis_crop_bounds

    device = torch.device("cpu")
    ckpt = REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
    model, raw = _load_detector(ckpt, device)
    stft_kwargs = {
        k: v
        for k, v in dict(raw.get("stft_kwargs") or {}).items()
        if k in ("n_time", "n_freq", "n_fft", "win_length", "hop_length")
    }
    norm_stats = raw.get("norm_stats")
    asds, _delta_f, assets = _load_packaged_asds()
    noise_std = _analysis_noise_std()
    event = load_event_dataset(assets)
    n_crop = int(round(CROP_S * SAMPLE_RATE))
    rng = np.random.default_rng(0)

    print(f"noise_std={noise_std:.6f}", flush=True)
    del rng
    prior_summary = {}
    prior_path = outdir / "summary.json"
    if prior_path.is_file():
        prior_summary = json.loads(prior_path.read_text())

    gaussian = _gaussian_draws(asds, n_crop, noise_std, int(args.n_gauss))
    if int(args.n_gauss) > 500:
        seed0 = _score_population(
            model, gaussian[:500], asds, stft_kwargs, norm_stats, noise_std, device
        )
        seed1 = _score_population(
            model, gaussian[500:], asds, stft_kwargs, norm_stats, noise_std, device
        )
        gaussian_rates = {}
        for thr in THRESHOLDS:
            k = seed0[str(thr)]["k"] + seed1[str(thr)]["k"]
            gaussian_rates[str(thr)] = _rates(k, int(args.n_gauss))
    else:
        seed0 = _score_population(
            model, gaussian, asds, stft_kwargs, norm_stats, noise_std, device
        )
        seed1 = None
        gaussian_rates = seed0
    fp = {"gaussian_asd": gaussian_rates}
    if seed0 is not None:
        fp["gaussian_asd_seed0"] = seed0
    if seed1 is not None:
        fp["gaussian_asd_seed1"] = seed1
    print(
        "gaussian_asd",
        {k: v["k"] for k, v in gaussian_rates.items()},
        flush=True,
    )

    if args.skip_offsource:
        stored = (prior_summary.get("false_positive") or {}).get("populations") or {}
        if "offsource_gwosc" not in stored:
            raise RuntimeError("summary.json has no off-source block to preserve")
        fp["offsource_gwosc"] = stored["offsource_gwosc"]
        print(
            "offsource_gwosc preserved",
            {k: v["k"] for k, v in fp["offsource_gwosc"].items()},
            flush=True,
        )
    else:
        print("building off-source crops", flush=True)
        offsource = _offsource_crops(assets, int(args.n_off))
        fp["offsource_gwosc"] = _score_population(
            model, offsource, asds, stft_kwargs, norm_stats, noise_std, device
        )
        print(
            "offsource_gwosc",
            {k: v["k"] for k, v in fp["offsource_gwosc"].items()},
            flush=True,
        )

    # Representative 4 s on-source crop and 128 s segment for timings.
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    _trig, crop0, crop1 = analysis_crop_bounds(duration=128.0, time_buffer=2.0, sample_rate=SAMPLE_RATE)
    # analysis_crop_bounds returns sample indices inside the 128 s segment.
    seg0 = trigger + 2.0 - 128.0
    td_full = {}
    for det in DETECTORS:
        ts = TimeSeries.read(str(assets["strain_gwf"][det]), channel=assets["channels"][det])
        ts = ts.crop(seg0, seg0 + 128.0)
        if float(ts.sample_rate.value) != SAMPLE_RATE:
            ts = ts.resample(SAMPLE_RATE)
        td_full[det] = np.asarray(ts.value, dtype=np.float64)[: int(128 * SAMPLE_RATE)]
    td_crop = {det: td_full[det][crop0:crop1] for det in DETECTORS}
    event_data = {"waveform": {d: np.array(event.data["waveform"][d]) for d in DETECTORS},
                  "asds": {d: np.array(event.data["asds"][d]) for d in DETECTORS}}

    def stft_call():
        return _spectrogram(td_crop, asds, stft_kwargs, norm_stats, noise_std)

    spec0, _ = stft_call()

    def cnn_call():
        return _cnn_gates(model, spec0, 0.73, device)

    def rebuild_call():
        return _rebuild(event_data, td_full, asds)

    timings: dict = {}
    mps_summary = None
    pe = {}
    if args.skip_timing:
        print("timing preserved from existing summary", flush=True)
        latency_cpu = prior_summary.get("latency_cpu") or {}
        mps_summary = prior_summary.get("latency_mps")
        pe = prior_summary.get("pe_notes") or {}
    else:
        print("timing front-end", flush=True)
        timings = {
            "t_stft": _time_loop(stft_call, N_WARM, N_ITER),
            "t_cnn": _time_loop(cnn_call, N_WARM, N_ITER),
            "t_rebuild": _time_loop(rebuild_call, N_WARM, N_ITER),
        }
        total = [a + b + c for a, b, c in zip(timings["t_stft"], timings["t_cnn"], timings["t_rebuild"])]
        timings["t_frontend_total"] = total
        timings["t_driver_overhead"] = _driver_overhead(td_crop["H1"], N_WARM, N_ITER, outdir)

        if torch.backends.mps.is_available():
            mps = torch.device("mps")
            model_mps, _ = _load_detector(ckpt, mps)

            def stft_mps():
                return _spectrogram(td_crop, asds, stft_kwargs, norm_stats, noise_std)

            def cnn_mps():
                return _cnn_gates(model_mps, spec0, 0.73, mps)

            print("timing MPS front-end", flush=True)
            mps_summary = {
                "t_stft": _summary(_time_loop(stft_mps, N_WARM, N_ITER)),
                "t_cnn": _summary(_time_loop(cnn_mps, N_WARM, N_ITER)),
            }

        if not args.skip_pe:
            print("timing PE n=512 x", N_ITER, flush=True)
            timings["t_pe_512"] = _pe_times(event, assets["fixed_context"], assets["baseline_ckpt"], 512, 2, N_ITER)
            n20 = int(args.pe_20000_repeats)
            print("timing PE n=20000 x", n20, flush=True)
            timings["t_pe_20000"] = _pe_times(event, assets["fixed_context"], assets["baseline_ckpt"], 20000, 1, n20)
            pe["t_pe_20000_note"] = (
                f"Measured {n20} iterations after 1 warm-up, not 200. "
                "A single 20_000-sample GWSamplerGNPE draw is tens of seconds on this CPU; "
                "200 repeats would be several hours and were not run."
            )
        latency_cpu = {key: _summary(val) for key, val in timings.items()}

    prior_fp = prior_summary.get("false_positive") or {}
    summary = {
        "false_positive": {
            "definition": "any IFO time-bin probability above the threshold on a 4 s crop",
            "thresholds": {"validation": 0.46, "event": 0.73},
            "checkpoint_threshold": raw.get("threshold"),
            "noise_std": noise_std,
            "stft": stft_kwargs,
            "gaussian_seeds": {"original": 0, "additional": 1, "split_at": 500},
            "populations": fp,
            "clean_gaussian": fp["gaussian_asd"],
            "combined_n1000": prior_fp.get("combined_n1000"),
        },
        "latency_cpu": latency_cpu,
        "latency_mps": mps_summary,
        "archived_research_driver_inject_detect_s": 3.158047154545784,
        "pe_notes": pe,
        "device_cpu": str(device),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if timings:
        rows = []
        for key, samples in timings.items():
            for i, value in enumerate(samples):
                rows.append({"component": key, "iteration": i, "seconds": value, "device": "cpu"})
        pd.DataFrame(rows).to_csv(outdir / "timings.csv", index=False)
    print(json.dumps({"clean_gaussian": fp["gaussian_asd"], "offsource_gwosc": fp["offsource_gwosc"]}, indent=2))


if __name__ == "__main__":
    main()
