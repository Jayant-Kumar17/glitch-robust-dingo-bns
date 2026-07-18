"""Unit tests for the isolated Advanced Noise Profiling module.

Offline-first: dimension, glitch, tracker, and injection tests use synthetic
arrays. Fallback grace intentionally hits an invalid GPS range so the
colored-noise path is exercised without a live GWOSC download.
"""

import os

import numpy as np

from adapt.noise_analytics import (
    AdvancedNoiseEncoder,
    LocalNoiseTracker,
    fetch_background_strain,
    inject_waveform_into_background,
    plot_rich_profile,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_noise_analytics_tests():
    print("--- Starting ADAPT Noise Analytics Verification Tests ---")

    # Use a short expected duration for fast offline tests while preserving
    # the same feature architecture (PSD bins + 4 s windows).
    sample_rate = 4096.0
    duration = 16.0  # 16 s -> 4 windows of 4 s; vector length = 128 + 12 = 140
    encoder = AdvancedNoiseEncoder(
        sample_rate=sample_rate,
        expected_duration_seconds=duration,
        psd_bins=128,
        window_size_seconds=4.0,
    )
    n_samples = encoder.expected_samples
    expected_len = encoder.vector_length
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Test 1: Dimension integrity
    # ------------------------------------------------------------------
    print("\n[Test 1] Dimension integrity of construct_rich_profile...")
    strain_a = rng.normal(0.0, 1e-22, size=n_samples)
    strain_b = rng.normal(0.0, 1e-22, size=n_samples)
    profile_a = encoder.construct_rich_profile(strain_a)
    profile_b = encoder.construct_rich_profile(strain_b)
    print(f"  vector length: {profile_a.vector.shape[0]} (expected {expected_len})")
    assert profile_a.vector.shape == (expected_len,), "Error: unexpected rich-profile length!"
    assert profile_b.vector.shape == profile_a.vector.shape, "Error: profile length not stable!"
    assert np.all(np.isfinite(profile_a.vector)), "Error: NaN/Inf in rich profile!"
    assert np.all(np.isfinite(profile_b.vector)), "Error: NaN/Inf in second rich profile!"
    assert profile_a.freqs[0] >= 20.0 - 1e-9, "Error: PSD grid must start at >= 20 Hz!"
    print("  => PASS")

    # ------------------------------------------------------------------
    # Test 2: Glitch / anomaly sensitivity
    # ------------------------------------------------------------------
    print("\n[Test 2] Glitch/anomaly sensitivity (kurtosis/skewness jump)...")
    quiet = rng.normal(0.0, 1e-22, size=n_samples)
    glitchy = quiet.copy()
    # Large impulsive spike in the second 4-second window.
    spike_idx = encoder.window_samples + encoder.window_samples // 2
    glitchy[spike_idx] += 1e-18

    quiet_moments = encoder.extract_non_gaussian_features(quiet)
    glitch_moments = encoder.extract_non_gaussian_features(glitchy)
    quiet_kurt = float(np.max(np.abs(quiet_moments["kurtosis"])))
    glitch_kurt = float(np.max(np.abs(glitch_moments["kurtosis"])))
    quiet_skew = float(np.max(np.abs(quiet_moments["skewness"])))
    glitch_skew = float(np.max(np.abs(glitch_moments["skewness"])))
    print(f"  quiet max |kurtosis|={quiet_kurt:.3f}, glitchy={glitch_kurt:.3f}")
    print(f"  quiet max |skewness|={quiet_skew:.3f}, glitchy={glitch_skew:.3f}")
    assert glitch_kurt > 5.0 * max(quiet_kurt, 1e-6), "Error: kurtosis did not jump on glitch!"
    assert glitch_skew > 5.0 * max(quiet_skew, 1e-6) or glitch_kurt > 20.0, (
        "Error: skewness/kurtosis did not show clear anomaly sensitivity!"
    )
    print("  => PASS")

    # ------------------------------------------------------------------
    # Test 3: Fallback grace
    # ------------------------------------------------------------------
    print("\n[Test 3] Fallback grace on invalid GPS range...")
    # GPS around 0 is not a valid open-data segment for modern runs.
    segment = fetch_background_strain(
        "H1",
        start_gps=0.0,
        duration_seconds=4.0,
        sample_rate=sample_rate,
        seed=123,
        allow_fallback=True,
    )
    print(f"  used_fallback={segment.used_fallback}, reason={segment.fallback_reason}")
    assert segment.used_fallback is True, "Error: invalid GPS should trigger fallback!"
    assert len(segment.strain) == int(4.0 * sample_rate), "Error: fallback length mismatch!"
    assert segment.sample_rate == sample_rate, "Error: fallback sample_rate mismatch!"
    assert np.all(np.isfinite(segment.strain)), "Error: non-finite fallback strain!"
    assert float(np.std(segment.strain)) > 0.0, "Error: fallback strain has zero variance!"
    print("  => PASS")

    # ------------------------------------------------------------------
    # Test 4: Injection sandbox (PyCBC epoch alignment)
    # ------------------------------------------------------------------
    print("\n[Test 4] Waveform injection into background (epoch-aware)...")
    bg = rng.normal(0.0, 1e-22, size=int(8.0 * sample_rate))
    injected, meta = inject_waveform_into_background(
        bg,
        sample_rate=sample_rate,
        m1=30.0,
        m2=25.0,
        merger_offset=2.0,
    )
    print(
        f"  placed samples={meta['n_placed_samples']}, "
        f"merger_index={meta['merger_index_in_background']}, "
        f"peak|h+|={meta['peak_abs_signal']:.3e}"
    )
    assert meta["n_placed_samples"] > 0, "Error: no waveform samples placed!"
    assert meta["start_idx"] < meta["end_idx"], "Error: invalid placement indices!"
    assert 0 <= meta["merger_index_in_background"] < len(bg), "Error: merger index out of range!"
    assert np.any(injected != bg), "Error: injection did not change the background!"
    assert meta["peak_abs_signal"] > 0.0, "Error: placed waveform has zero amplitude!"
    # Coalescence should land at the requested merger index in the background.
    assert meta["merger_index_in_background"] == len(bg) - int(round(2.0 * sample_rate))
    print("  => PASS")

    # ------------------------------------------------------------------
    # Test 5: Tracker + vector diagnostics
    # ------------------------------------------------------------------
    print("\n[Test 5] LocalNoiseTracker history, drift, and PDF diagnostics...")
    tracker = LocalNoiseTracker(encoder, history_size=3)
    drifts = []
    for i in range(4):
        seg = rng.normal(0.0, 1e-22 * (1.0 + 0.2 * i), size=n_samples)
        drifts.append(tracker.update_profile(seg))
    state = tracker.state
    print(f"  drifts={drifts}")
    print(f"  history length={len(state.history)} (cap=3)")
    assert drifts[0] == 0.0, "Error: first drift should be exactly 0!"
    assert all(d >= 0.0 and np.isfinite(d) for d in drifts), "Error: drift must be finite and >= 0!"
    assert len(state.history) == 3, "Error: history was not capped at history_size!"
    assert len(state.drift_deltas) == 3, "Error: drift history not capped!"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pdf_path = plot_rich_profile(state, output_dir=RESULTS_DIR)
    print(f"  diagnostic PDF: {pdf_path}")
    assert pdf_path.endswith(".pdf"), "Error: diagnostic must be a PDF!"
    assert os.path.isfile(pdf_path), "Error: diagnostic PDF was not written!"
    assert os.path.getsize(pdf_path) > 0, "Error: diagnostic PDF is empty!"
    print("  => PASS")

    print("\nALL NOISE ANALYTICS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_noise_analytics_tests()
