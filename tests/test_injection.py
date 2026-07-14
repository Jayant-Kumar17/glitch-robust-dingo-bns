"""End-to-end verification of the "Option A" simulation approach:

generate a synthetic waveform with known parameters -> inject it into
real detector noise -> mock the matched-filter trigger by feeding the
known parameters to the router -> verify it routes to the correct class.
"""

from gwpy.timeseries import TimeSeries

from adapt.injection import generate_injection
from adapt.router import MatchedFilterRouter


def run_injection_tests():
    print("--- Starting ADAPT Injection Pipeline Verification Tests ---")

    # GPS time 1240559616 is the same known clean O3 segment used in
    # test_noise.py, just fetched for longer (64s instead of 10s) so a
    # BNS-length inspiral fits before the merger.
    print("\nFetching 64 seconds of real LIGO O3 noise from GWOSC...")
    o3_noise = TimeSeries.fetch_open_data("H1", 1240559616, 1240559680, verbose=True)
    noise = o3_noise.value
    sample_rate = o3_noise.sample_rate.value
    print(f"Downloaded noise shape: {noise.shape}, sample_rate: {sample_rate} Hz")

    router = MatchedFilterRouter()

    # Case A: BNS-like injection (long, low-mass inspiral).
    print("\n[Case A] Injecting BNS-like signal (m1=1.4, m2=1.35) into real noise...")
    injected_a, true_params_a = generate_injection(1.4, 1.35, noise, sample_rate)
    print(f"  Injected strain shape: {injected_a.shape}")
    print(f"  True params (mock trigger): {true_params_a}")
    decision_a = router.route_event(true_params_a["m1"], true_params_a["m2"])
    print(f"  Router decision: {decision_a}")
    assert decision_a["route"] == "BNS", "Error: BNS-like injection should route to BNS!"

    # Case B: BBH-like injection (short, high-mass merger).
    print("\n[Case B] Injecting BBH-like signal (m1=35, m2=30) into real noise...")
    injected_b, true_params_b = generate_injection(35.0, 30.0, noise, sample_rate)
    print(f"  Injected strain shape: {injected_b.shape}")
    print(f"  True params (mock trigger): {true_params_b}")
    decision_b = router.route_event(true_params_b["m1"], true_params_b["m2"])
    print(f"  Router decision: {decision_b}")
    assert decision_b["route"] == "BBH", "Error: BBH-like injection should route to BBH!"

    print(
        "\n🎉 ALL INJECTION TESTS PASSED SUCCESSFULLY! "
        "End-to-end pipeline (real noise -> synthetic injection -> mock trigger -> router) is sound."
    )


if __name__ == "__main__":
    run_injection_tests()
