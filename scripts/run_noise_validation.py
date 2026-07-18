"""Long-duration real-data validation campaign for ADAPT noise profiling.

Fetches a continuous 2048 s H1 strain block from GWOSC (or colored-noise
fallback), injects a loud BBH at the stream midpoint, then rolls a
LocalNoiseTracker over 256 s windows stepped by 32 s. Writes a
publication-quality PDF to results/real_validation_profile_<timestamp>.pdf.

Does not modify adapt.noise_analytics or any routing code.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np

from adapt.noise_analytics import (
    AdvancedNoiseEncoder,
    LocalNoiseTracker,
    NoiseSegment,
    _synthesize_colored_noise,
    fetch_background_strain,
    inject_waveform_into_background,
    plot_rich_profile,
)

# Campaign parameters (see plan).
DETECTOR = "H1"
GPS_START = 1240559616.0  # known clean O3 segment
TOTAL_DURATION_S = 2048.0
SAMPLE_RATE = 4096.0
WINDOW_S = 256.0
STEP_S = 32.0
HISTORY_SIZE = 16
MIDPOINT_S = TOTAL_DURATION_S / 2.0  # 1024 s
BBH_M1 = 40.0
BBH_M2 = 30.0
# Continuous-archive downloads can hang on a throttled link; fall back if so.
# Override with ADAPT_FETCH_TIMEOUT_S (seconds) if needed.
FETCH_TIMEOUT_S = float(os.environ.get("ADAPT_FETCH_TIMEOUT_S", "600"))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data", "gwosc")
# Official O3a 4 kHz continuous-archive file covering GPS 1240559616 (+4096 s).
LOCAL_HDF5 = os.path.join(
    DATA_DIR, "H-H1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"
)
GWOSC_URL = (
    "https://gwosc.org/archive/data/O3a_4KHZ_R1/1240465408/"
    "H-H1_GWOSC_O3a_4KHZ_R1-1240559616-4096.hdf5"
)


def _fetch_worker(queue: Queue) -> None:
    """Child process target: put a NoiseSegment (or exception) on the queue."""
    try:
        segment = fetch_background_strain(
            DETECTOR,
            GPS_START,
            duration_seconds=TOTAL_DURATION_S,
            sample_rate=SAMPLE_RATE,
            seed=42,
            allow_fallback=True,
        )
        queue.put(("ok", segment))
    except Exception as exc:  # pragma: no cover - defensive
        queue.put(("err", exc))


def _fallback_segment(reason: str) -> NoiseSegment:
    strain = _synthesize_colored_noise(TOTAL_DURATION_S, SAMPLE_RATE, seed=42)
    return NoiseSegment(
        strain=strain,
        sample_rate=SAMPLE_RATE,
        detector=DETECTOR,
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


def load_local_gwosc_hdf5(path: str) -> NoiseSegment:
    """Load the first TOTAL_DURATION_S of a local GWOSC HDF5 as real strain."""
    from gwpy.timeseries import TimeSeries

    print(f"  loading local GWOSC file: {path}", flush=True)
    ts = TimeSeries.read(path, format="hdf5.gwosc")
    # Archive files are typically longer than the campaign (e.g. 4096 s).
    end_gps = GPS_START + TOTAL_DURATION_S
    ts = ts.crop(GPS_START, end_gps)
    strain = _strain_from_timeseries(ts)
    rms = float(np.sqrt(np.mean(strain**2)))
    print(
        f"  real H1 strain loaded: n={len(strain)}, rms={rms:.3e}, "
        f"t0={float(ts.t0.value):.1f}",
        flush=True,
    )
    return NoiseSegment(
        strain=strain,
        sample_rate=SAMPLE_RATE,
        detector=DETECTOR,
        start_gps=GPS_START,
        duration_seconds=TOTAL_DURATION_S,
        used_fallback=False,
        fallback_reason=None,
    )


def download_gwosc_hdf5(dest: str = LOCAL_HDF5) -> str:
    """Download the continuous-archive HDF5 with curl (resumable, progress bar)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading real H1 archive via curl:\n    {GWOSC_URL}", flush=True)
    cmd = (
        f'curl -L --fail --retry 5 --retry-delay 5 -C - '
        f'-o "{dest}" "{GWOSC_URL}"'
    )
    rc = os.system(cmd)
    if rc != 0 or not os.path.isfile(dest) or os.path.getsize(dest) < 1_000_000:
        raise RuntimeError(f"curl download failed (rc={rc})")
    print(f"  downloaded {os.path.getsize(dest) / 1e6:.1f} MB -> {dest}", flush=True)
    return dest


def fetch_with_timeout():
    """Fetch GWOSC strain, or force colored-noise fallback if the download hangs.

    Preference order:
      1. Local HDF5 under data/gwosc/ (fast, real data)
      2. curl download of the official continuous-archive file (real data)
      3. fetch_background_strain in a subprocess (real data or library fallback)
      4. Local colored-noise synthesis if everything else fails/times out
    """
    # 1–2. Prefer an explicit curl/local path — gwpy's fetch_open_data often
    # stalls on multi-hundred-MB continuous archives.
    if os.path.isfile(LOCAL_HDF5) and os.path.getsize(LOCAL_HDF5) > 1_000_000:
        try:
            return load_local_gwosc_hdf5(LOCAL_HDF5)
        except Exception as exc:
            print(f"  local HDF5 load failed: {exc}", flush=True)

    force_curl = os.environ.get("ADAPT_FORCE_CURL", "1") == "1"
    if force_curl:
        try:
            download_gwosc_hdf5(LOCAL_HDF5)
            return load_local_gwosc_hdf5(LOCAL_HDF5)
        except Exception as exc:
            print(f"  curl download/load failed: {exc}", flush=True)
            print("  falling back to fetch_background_strain...", flush=True)

    queue: Queue = Queue()
    proc = Process(target=_fetch_worker, args=(queue,))
    proc.start()
    proc.join(timeout=FETCH_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        print(
            f"  GWOSC fetch exceeded {FETCH_TIMEOUT_S:.0f}s; "
            "forcing colored-noise fallback path...",
            flush=True,
        )
        return _fallback_segment(
            f"TimeoutError: GWOSC fetch exceeded {FETCH_TIMEOUT_S:.0f}s"
        )

    if queue.empty():
        return _fallback_segment("RuntimeError: GWOSC fetch worker exited without result")

    status, payload = queue.get()
    if status == "ok":
        return payload
    return _fallback_segment(f"{type(payload).__name__}: {payload}")


def run_noise_validation_campaign():
    print("=" * 70)
    print(" ADAPT LONG-DURATION REAL NOISE VALIDATION CAMPAIGN")
    print("=" * 70)
    print(f"Detector={DETECTOR}  GPS={GPS_START}  duration={TOTAL_DURATION_S}s")
    print(f"Window={WINDOW_S}s  step={STEP_S}s  history={HISTORY_SIZE}")
    print(f"Midpoint BBH injection: m1={BBH_M1}, m2={BBH_M2} (total mass 70 Msun)")
    print()

    # ------------------------------------------------------------------
    # 1. Massive data retrieval
    # ------------------------------------------------------------------
    print("--- Step 1: Fetching continuous background strain ---", flush=True)
    t0 = time.time()
    segment = fetch_with_timeout()
    elapsed = time.time() - t0
    strain = segment.strain
    fs = segment.sample_rate
    n_expected = int(round(TOTAL_DURATION_S * SAMPLE_RATE))
    print(f"  used_fallback={segment.used_fallback}")
    if segment.used_fallback:
        print(f"  fallback_reason={segment.fallback_reason}")
    print(f"  length={len(strain)} samples (expected {n_expected}), sample_rate={fs} Hz")
    print(f"  fetch elapsed={elapsed:.1f}s")
    if len(strain) != n_expected:
        raise RuntimeError(f"unexpected strain length {len(strain)} != {n_expected}")

    # ------------------------------------------------------------------
    # 2. Mid-stream BBH injection at absolute midpoint
    # ------------------------------------------------------------------
    print("\n--- Step 2: Injecting loud BBH at stream midpoint ---", flush=True)
    # inject_waveform places coalescence at len - merger_offset*fs.
    # Want coalescence at mid = 1024 s => merger_offset = 2048 - 1024 = 1024.
    merger_offset = TOTAL_DURATION_S - MIDPOINT_S
    injected, meta = inject_waveform_into_background(
        strain,
        sample_rate=fs,
        m1=BBH_M1,
        m2=BBH_M2,
        merger_offset=merger_offset,
    )
    mid_idx = int(round(MIDPOINT_S * fs))
    print(f"  coalescence target index={mid_idx} (t={MIDPOINT_S}s)")
    print(f"  merger_index_in_background={meta['merger_index_in_background']}")
    print(f"  placement=[{meta['start_idx']}, {meta['end_idx']}), n={meta['n_placed_samples']}")
    print(f"  peak |h+| = {meta['peak_abs_signal']:.3e}")
    if meta["merger_index_in_background"] != mid_idx:
        raise RuntimeError(
            f"coalescence landed at {meta['merger_index_in_background']}, expected {mid_idx}"
        )

    # ------------------------------------------------------------------
    # 3. Sequential streaming simulation
    # ------------------------------------------------------------------
    print("\n--- Step 3: Rolling LocalNoiseTracker over the stream ---", flush=True)
    encoder = AdvancedNoiseEncoder(
        sample_rate=SAMPLE_RATE,
        expected_duration_seconds=WINDOW_S,
        psd_bins=128,
        window_size_seconds=4.0,
    )
    tracker = LocalNoiseTracker(encoder, history_size=HISTORY_SIZE)

    window_samples = int(round(WINDOW_S * fs))
    step_samples = int(round(STEP_S * fs))
    max_start = len(injected) - window_samples
    starts = list(range(0, max_start + 1, step_samples))
    n_steps = len(starts)
    print(f"  n_steps={n_steps} (expected {(TOTAL_DURATION_S - WINDOW_S) / STEP_S + 1:.0f})")

    drifts = []
    t_loop = time.time()
    for i, start in enumerate(starts):
        chunk = injected[start : start + window_samples]
        drift = tracker.update_profile(chunk)
        drifts.append(drift)
        start_s = start / fs
        end_s = (start + window_samples) / fs
        gps0 = GPS_START + start_s
        gps1 = GPS_START + end_s
        contains_mid = start_s <= MIDPOINT_S < end_s
        flag = "  << MIDPOINT WINDOW" if contains_mid else ""
        if i % 5 == 0 or contains_mid or i == n_steps - 1:
            print(
                f"  [{i + 1:3d}/{n_steps}] t=[{start_s:7.1f}, {end_s:7.1f}]s  "
                f"GPS=[{gps0:.1f}, {gps1:.1f}]  drift={drift:.4f}{flag}",
                flush=True,
            )

    print(f"  loop elapsed={time.time() - t_loop:.1f}s")

    drifts_arr = np.asarray(drifts, dtype=np.float64)
    # Windows that contain the midpoint: start_s <= 1024 < start_s + 256
    mid_steps = [
        i
        for i, start in enumerate(starts)
        if (start / fs) <= MIDPOINT_S < (start / fs) + WINDOW_S
    ]
    early = drifts_arr[: max(1, n_steps // 5)]
    mid_drifts = drifts_arr[mid_steps] if mid_steps else np.array([0.0])
    max_drift = float(np.max(drifts_arr))
    early_mean = float(np.mean(early))
    mid_max = float(np.max(mid_drifts))
    spiked = mid_max > early_mean + 1e-6 and mid_max >= 0.5 * max_drift

    print("\n--- Drift summary ---")
    print(f"  n_steps={n_steps}")
    print(f"  early-window mean drift={early_mean:.4f}")
    print(f"  midpoint-window max drift={mid_max:.4f} (steps {mid_steps})")
    print(f"  global max drift={max_drift:.4f}")
    print(f"  midpoint spike detected={spiked}")

    # ------------------------------------------------------------------
    # 4. Publication-grade diagnostics (rename to required filename)
    # ------------------------------------------------------------------
    print("\n--- Step 4: Writing publication diagnostics PDF ---", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_pdf = plot_rich_profile(tracker.state, output_dir=RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_pdf = os.path.join(RESULTS_DIR, f"real_validation_profile_{timestamp}.pdf")
    shutil.move(tmp_pdf, final_pdf)
    print(f"  saved: {final_pdf}")

    print("\n" + "=" * 70)
    print(
        f"CAMPAIGN COMPLETE: {n_steps} updates | max_drift={max_drift:.4f} | "
        f"midpoint_spike={spiked} | pdf={os.path.basename(final_pdf)}"
    )
    print("=" * 70)
    return {
        "n_steps": n_steps,
        "max_drift": max_drift,
        "midpoint_spike": spiked,
        "pdf_path": final_pdf,
        "used_fallback": segment.used_fallback,
        "drifts": drifts,
    }


if __name__ == "__main__":
    run_noise_validation_campaign()
