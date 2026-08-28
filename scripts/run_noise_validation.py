"""Long-duration dual-detector validation campaign for ADAPT noise profiling.

Fetches continuous 2048 s H1 + L1 strain (GWOSC or colored-noise fallback),
injects a loud IMRPhenomD BBH into *both* contaminated arrays at the stream
midpoint (t = 1024 s), then rolls a GlobalNoiseHub over 256 s windows stepped
by 32 s. Chunks are sliced from the contaminated arrays so the trackers see
the waveform pass through updates ~25–35.

Writes:
  - results/global_network_profile_<timestamp>.pdf  (stacked H1/L1 drift)
  - results/real_validation_profile_<timestamp>.pdf (H1 rich-profile panels)

Does not modify adapt.noise_analytics library contracts beyond calling them.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Dict, Tuple

import numpy as np

from adapt.noise_analytics import (
    GlobalNoiseHub,
    NoiseSegment,
    _synthesize_colored_noise,
    fetch_background_strain,
    inject_waveform_into_background,
    plot_network_diagnostics,
    plot_rich_profile,
)

# Campaign parameters.
DETECTORS = ("H1", "L1")
GPS_START = 1240559616.0  # known clean O3 segment
TOTAL_DURATION_S = 2048.0
SAMPLE_RATE = 4096.0
WINDOW_S = 256.0
STEP_S = 32.0
HISTORY_SIZE = 16
MIDPOINT_S = TOTAL_DURATION_S / 2.0  # 1024 s
BBH_M1 = 40.0
BBH_M2 = 30.0
# merger_offset is seconds before the *end* of the array where coalescence lands.
# Want coalescence at absolute midpoint => offset = 2048 - 1024 = 1024.
MERGER_OFFSET_S = TOTAL_DURATION_S - MIDPOINT_S
FETCH_TIMEOUT_S = float(os.environ.get("ADAPT_FETCH_TIMEOUT_S", "600"))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data", "gwosc")

# Official O3a 4 kHz continuous-archive files covering GPS 1240559616 (+4096 s).
LOCAL_HDF5 = {
    "H1": os.path.join(DATA_DIR, "H-H1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"),
    "L1": os.path.join(DATA_DIR, "L-L1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"),
}
GWOSC_URL = {
    "H1": (
        "https://gwosc.org/archive/data/O3a_4KHZ_R1/1240465408/"
        "H-H1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"
    ),
    "L1": (
        "https://gwosc.org/archive/data/O3a_4KHZ_R1/1240465408/"
        "L-L1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"
    ),
}


def _fallback_segment(detector: str, reason: str, seed: int) -> NoiseSegment:
    strain = _synthesize_colored_noise(
        TOTAL_DURATION_S, SAMPLE_RATE, seed=seed, detector=detector
    )
    return NoiseSegment(
        strain=strain,
        sample_rate=SAMPLE_RATE,
        detector=detector,
        start_gps=GPS_START,
        duration_seconds=TOTAL_DURATION_S,
        used_fallback=True,
        fallback_reason=reason,
    )


def _strain_from_timeseries(ts) -> np.ndarray:
    """Resample/crop a gwpy TimeSeries to the campaign length and sample rate."""
    if abs(float(ts.sample_rate.value) - float(SAMPLE_RATE)) > 1e-9:
        ts = ts.resample(SAMPLE_RATE)
    strain = np.asarray(ts.value, dtype=np.float64)
    n_expected = int(round(TOTAL_DURATION_S * SAMPLE_RATE))
    if len(strain) > n_expected:
        strain = strain[:n_expected]
    elif len(strain) < n_expected:
        padded = np.zeros(n_expected, dtype=np.float64)
        padded[: len(strain)] = strain
        strain = padded
    return strain


def load_local_gwosc_hdf5(detector: str, path: str) -> NoiseSegment:
    """Load the first TOTAL_DURATION_S of a local GWOSC HDF5 as real strain."""
    from gwpy.timeseries import TimeSeries

    print(f"  loading local GWOSC file ({detector}): {path}", flush=True)
    ts = TimeSeries.read(path, format="hdf5.gwosc")
    end_gps = GPS_START + TOTAL_DURATION_S
    ts = ts.crop(GPS_START, end_gps)
    strain = _strain_from_timeseries(ts)
    rms = float(np.sqrt(np.mean(strain**2)))
    print(
        f"  real {detector} strain loaded: n={len(strain)}, rms={rms:.3e}, "
        f"t0={float(ts.t0.value):.1f}",
        flush=True,
    )
    return NoiseSegment(
        strain=strain,
        sample_rate=SAMPLE_RATE,
        detector=detector,
        start_gps=GPS_START,
        duration_seconds=TOTAL_DURATION_S,
        used_fallback=False,
        fallback_reason=None,
    )


def download_gwosc_hdf5(detector: str, dest: str) -> str:
    """Download the continuous-archive HDF5 with curl (resumable)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = GWOSC_URL[detector]
    print(f"  downloading real {detector} archive via curl:\n    {url}", flush=True)
    cmd = f'curl -L --fail --retry 5 --retry-delay 5 -C - -o "{dest}" "{url}"'
    rc = os.system(cmd)
    if rc != 0 or not os.path.isfile(dest) or os.path.getsize(dest) < 1_000_000:
        raise RuntimeError(f"curl download failed for {detector} (rc={rc})")
    print(f"  downloaded {os.path.getsize(dest) / 1e6:.1f} MB -> {dest}", flush=True)
    return dest


def _fetch_worker(detector: str, queue: Queue, seed: int) -> None:
    try:
        segment = fetch_background_strain(
            detector,
            GPS_START,
            duration_seconds=TOTAL_DURATION_S,
            sample_rate=SAMPLE_RATE,
            seed=seed,
            allow_fallback=True,
        )
        queue.put(("ok", segment))
    except Exception as exc:  # pragma: no cover
        queue.put(("err", exc))


def fetch_detector_strain(detector: str, seed: int) -> NoiseSegment:
    """Fetch one detector: local HDF5 -> curl -> timed gwpy fetch -> sim."""
    local = LOCAL_HDF5[detector]
    if os.path.isfile(local) and os.path.getsize(local) > 1_000_000:
        try:
            return load_local_gwosc_hdf5(detector, local)
        except Exception as exc:
            print(f"  local {detector} HDF5 load failed: {exc}", flush=True)

    if os.environ.get("ADAPT_FORCE_CURL", "1") == "1":
        try:
            download_gwosc_hdf5(detector, local)
            return load_local_gwosc_hdf5(detector, local)
        except Exception as exc:
            print(f"  curl {detector} download/load failed: {exc}", flush=True)
            print(f"  falling back to fetch_background_strain for {detector}...", flush=True)

    queue: Queue = Queue()
    proc = Process(target=_fetch_worker, args=(detector, queue, seed))
    proc.start()
    proc.join(timeout=FETCH_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        print(
            f"  {detector} GWOSC fetch exceeded {FETCH_TIMEOUT_S:.0f}s; "
            "forcing colored-noise fallback...",
            flush=True,
        )
        return _fallback_segment(
            detector,
            f"TimeoutError: GWOSC fetch exceeded {FETCH_TIMEOUT_S:.0f}s",
            seed,
        )

    if queue.empty():
        return _fallback_segment(
            detector, "RuntimeError: GWOSC fetch worker exited without result", seed
        )

    status, payload = queue.get()
    if status == "ok":
        return payload
    return _fallback_segment(detector, f"{type(payload).__name__}: {payload}", seed)


def fetch_h1_l1_background() -> Dict[str, NoiseSegment]:
    """Acquire concurrent H1 and L1 backgrounds (real preferred, hybrid OK)."""
    network: Dict[str, NoiseSegment] = {}
    for i, det in enumerate(DETECTORS):
        print(f"\n--- Fetching {det} ---", flush=True)
        network[det] = fetch_detector_strain(det, seed=42 + i)
        seg = network[det]
        print(
            f"  {det}: used_fallback={seg.used_fallback}, "
            f"length={len(seg.strain)}, sample_rate={seg.sample_rate}",
            flush=True,
        )
        if seg.used_fallback:
            print(f"  {det} fallback_reason={seg.fallback_reason}", flush=True)
    return network


def inject_bbh_at_midpoint(
    background: np.ndarray, sample_rate: float, detector: str
) -> Tuple[np.ndarray, dict]:
    """Inject IMRPhenomD so coalescence sits at absolute stream midpoint."""
    mid_idx = int(round(MIDPOINT_S * sample_rate))
    injected, meta = inject_waveform_into_background(
        background,
        sample_rate=sample_rate,
        m1=BBH_M1,
        m2=BBH_M2,
        merger_offset=MERGER_OFFSET_S,
    )
    landed = int(meta["merger_index_in_background"])
    print(
        f"  [{detector}] coalescence target={mid_idx} (t={MIDPOINT_S}s), "
        f"landed={landed}, placement=[{meta['start_idx']}, {meta['end_idx']}), "
        f"n={meta['n_placed_samples']}, peak|h+|={meta['peak_abs_signal']:.3e}",
        flush=True,
    )
    if landed != mid_idx:
        raise RuntimeError(
            f"{detector}: coalescence at {landed}, expected midpoint index {mid_idx}"
        )
    # Sanity: contaminated array must differ from background near merger.
    if not np.any(injected != background):
        raise RuntimeError(f"{detector}: injection left the background unchanged")
    return injected, meta


def run_noise_validation_campaign():
    print("=" * 70)
    print(" ADAPT DUAL-DETECTOR NOISE VALIDATION (H1 + L1 + BBH)")
    print("=" * 70)
    print(f"Detectors={list(DETECTORS)}  GPS={GPS_START}  duration={TOTAL_DURATION_S}s")
    print(f"Window={WINDOW_S}s  step={STEP_S}s  history={HISTORY_SIZE}")
    print(
        f"Midpoint BBH injection into BOTH arrays: m1={BBH_M1}, m2={BBH_M2} "
        f"(merger_offset={MERGER_OFFSET_S}s -> t={MIDPOINT_S}s)"
    )
    print()

    # ------------------------------------------------------------------
    # 1. Fetch H1 + L1 backgrounds
    # ------------------------------------------------------------------
    print("--- Step 1: Fetching continuous H1 + L1 background strain ---", flush=True)
    t0 = time.time()
    segments = fetch_h1_l1_background()
    print(f"  fetch elapsed={time.time() - t0:.1f}s", flush=True)

    n_expected = int(round(TOTAL_DURATION_S * SAMPLE_RATE))
    for det in DETECTORS:
        if len(segments[det].strain) != n_expected:
            raise RuntimeError(
                f"{det}: unexpected length {len(segments[det].strain)} != {n_expected}"
            )
        if abs(segments[det].sample_rate - SAMPLE_RATE) > 1e-9:
            raise RuntimeError(f"{det}: sample_rate mismatch")

    fs = SAMPLE_RATE
    h1_raw = np.asarray(segments["H1"].strain, dtype=np.float64)
    l1_raw = np.asarray(segments["L1"].strain, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Inject BBH into BOTH contaminated arrays at absolute midpoint
    # ------------------------------------------------------------------
    print("\n--- Step 2: Injecting loud BBH into H1 and L1 at midpoint ---", flush=True)
    h1_strain, meta_h1 = inject_bbh_at_midpoint(h1_raw, fs, "H1")
    l1_strain, meta_l1 = inject_bbh_at_midpoint(l1_raw, fs, "L1")
    mid_idx = int(round(MIDPOINT_S * fs))
    # Contaminated dict used for all subsequent streaming slices.
    contaminated = {"H1": h1_strain, "L1": l1_strain}
    print(
        f"  verification: |h1-h1_raw|@{mid_idx}="
        f"{abs(h1_strain[mid_idx] - h1_raw[mid_idx]):.3e}, "
        f"|l1-l1_raw|@{mid_idx}="
        f"{abs(l1_strain[mid_idx] - l1_raw[mid_idx]):.3e}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Rolling GlobalNoiseHub over contaminated chunks
    # ------------------------------------------------------------------
    print("\n--- Step 3: Rolling GlobalNoiseHub over contaminated stream ---", flush=True)
    hub = GlobalNoiseHub(
        list(DETECTORS),
        sample_rate=SAMPLE_RATE,
        expected_duration_seconds=WINDOW_S,
        history_size=HISTORY_SIZE,
        psd_bins=128,
        window_size_seconds=4.0,
    )

    window_samples = int(round(WINDOW_S * fs))
    step_samples = int(round(STEP_S * fs))
    max_start = n_expected - window_samples
    starts = list(range(0, max_start + 1, step_samples))
    n_steps = len(starts)
    print(f"  n_steps={n_steps} (expected {(TOTAL_DURATION_S - WINDOW_S) / STEP_S + 1:.0f})")

    drifts_h1 = []
    drifts_l1 = []
    t_loop = time.time()
    for i, start in enumerate(starts):
        end = start + window_samples
        # Critical: slice the *contaminated* arrays so the waveform enters
        # the trackers when the window overlaps the midpoint.
        data_dict = {
            "H1": contaminated["H1"][start:end],
            "L1": contaminated["L1"][start:end],
        }
        drifts = hub.update_network(data_dict)
        drifts_h1.append(drifts["H1"])
        drifts_l1.append(drifts["L1"])

        start_s = start / fs
        end_s = end / fs
        contains_mid = start_s <= MIDPOINT_S < end_s
        flag = "  << MIDPOINT WINDOW" if contains_mid else ""
        if i % 5 == 0 or contains_mid or i == n_steps - 1:
            print(
                f"  [{i + 1:3d}/{n_steps}] t=[{start_s:7.1f}, {end_s:7.1f}]s  "
                f"H1_drift={drifts['H1']:.4f}  L1_drift={drifts['L1']:.4f}{flag}",
                flush=True,
            )

    print(f"  loop elapsed={time.time() - t_loop:.1f}s")

    mid_steps = [
        i
        for i, start in enumerate(starts)
        if (start / fs) <= MIDPOINT_S < (start / fs) + WINDOW_S
    ]
    h1_arr = np.asarray(drifts_h1, dtype=np.float64)
    l1_arr = np.asarray(drifts_l1, dtype=np.float64)
    early_n = max(1, n_steps // 5)
    h1_early = float(np.mean(h1_arr[:early_n]))
    l1_early = float(np.mean(l1_arr[:early_n]))
    h1_mid = float(np.max(h1_arr[mid_steps])) if mid_steps else 0.0
    l1_mid = float(np.max(l1_arr[mid_steps])) if mid_steps else 0.0
    h1_max = float(np.max(h1_arr))
    l1_max = float(np.max(l1_arr))
    h1_spiked = h1_mid > h1_early + 1e-6 and h1_mid >= 0.5 * h1_max
    l1_spiked = l1_mid > l1_early + 1e-6 and l1_mid >= 0.5 * l1_max

    print("\n--- Drift summary ---")
    print(f"  n_steps={n_steps}, midpoint steps={mid_steps}")
    print(f"  H1 early_mean={h1_early:.4f}  mid_max={h1_mid:.4f}  global_max={h1_max:.4f}  spike={h1_spiked}")
    print(f"  L1 early_mean={l1_early:.4f}  mid_max={l1_mid:.4f}  global_max={l1_max:.4f}  spike={l1_spiked}")
    if not (h1_spiked and l1_spiked):
        raise RuntimeError(
            "BBH midpoint spike not detected in both detectors — "
            "injection may not be reaching the hub streaming loop"
        )

    # ------------------------------------------------------------------
    # 4. Diagnostics PDFs
    # ------------------------------------------------------------------
    print("\n--- Step 4: Writing publication diagnostics PDFs ---", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    network_pdf = plot_network_diagnostics(hub.state, output_dir=RESULTS_DIR)
    final_network = os.path.join(RESULTS_DIR, f"global_network_profile_{timestamp}.pdf")
    # plot_network_diagnostics already timestamped; rename for a paired stamp.
    if network_pdf != final_network:
        shutil.move(network_pdf, final_network)
    print(f"  network PDF: {final_network}")

    # Keep single-detector rich profile for H1 as the legacy artifact name.
    h1_pdf_tmp = plot_rich_profile(hub.state.tracker_states["H1"], output_dir=RESULTS_DIR)
    final_h1 = os.path.join(RESULTS_DIR, f"real_validation_profile_{timestamp}.pdf")
    shutil.move(h1_pdf_tmp, final_h1)
    print(f"  H1 rich PDF: {final_h1}")

    print("\n" + "=" * 70)
    print(
        f"CAMPAIGN COMPLETE: {n_steps} updates | "
        f"H1_spike={h1_spiked} L1_spike={l1_spiked} | "
        f"network_pdf={os.path.basename(final_network)}"
    )
    print("=" * 70)
    return {
        "n_steps": n_steps,
        "h1_max_drift": h1_max,
        "l1_max_drift": l1_max,
        "h1_midpoint_spike": h1_spiked,
        "l1_midpoint_spike": l1_spiked,
        "network_pdf": final_network,
        "h1_pdf": final_h1,
        "meta_h1": meta_h1,
        "meta_l1": meta_l1,
        "used_fallback": {d: segments[d].used_fallback for d in DETECTORS},
    }


if __name__ == "__main__":
    run_noise_validation_campaign()
