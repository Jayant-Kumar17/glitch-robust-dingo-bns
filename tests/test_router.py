from adapt.router import MatchedFilterRouter


def run_router_tests():
    print("--- Starting ADAPT Hierarchical Router Verification Tests ---")

    router = MatchedFilterRouter()

    # Test 1: GW170817-like (BNS, soft band + mass-structure check passes)
    print("\n[Test 1] Simulating GW170817-like BNS trigger (m1=1.46, m2=1.27)...")
    result_1 = router.route_event(1.46, 1.27)
    print(f"  -> {result_1}")
    assert result_1["route"] == "BNS", "Error: GW170817-like trigger should route to BNS!"
    assert result_1["confidence"] == "high", "Error: expected 'high' confidence for GW170817-like trigger!"

    # Test 2: GW150914-like (BBH, soft band + mass-structure check passes)
    print("\n[Test 2] Simulating GW150914-like BBH trigger (m1=35.6, m2=30.6)...")
    result_2 = router.route_event(35.6, 30.6)
    print(f"  -> {result_2}")
    assert result_2["route"] == "BBH", "Error: GW150914-like trigger should route to BBH!"
    assert result_2["confidence"] == "high", "Error: expected 'high' confidence for GW150914-like trigger!"

    # Test 3: Deep hard-gated BNS (chirp mass well below 0.87 Msun)
    print("\n[Test 3] Simulating deep low-mass trigger (m1=0.4, m2=0.4)...")
    result_3 = router.route_event(0.4, 0.4)
    print(f"  -> {result_3}")
    assert result_3["route"] == "BNS", "Error: deep low-mass trigger should route to BNS!"
    assert result_3["confidence"] == "very_high", "Error: expected 'very_high' confidence for deep low-mass trigger!"

    # Test 4: Deep hard-gated BBH (chirp mass well above 39.17 Msun)
    print("\n[Test 4] Simulating deep high-mass trigger (m1=50, m2=50)...")
    result_4 = router.route_event(50.0, 50.0)
    print(f"  -> {result_4}")
    assert result_4["route"] == "BBH", "Error: deep high-mass trigger should route to BBH!"
    assert result_4["confidence"] == "very_high", "Error: expected 'very_high' confidence for deep high-mass trigger!"

    # Test 5: Pure middle-band ambiguous case (equal masses, chirp mass between soft bands)
    print("\n[Test 5] Simulating middle-band ambiguous trigger (m1=2.0, m2=2.0)...")
    result_5 = router.route_event(2.0, 2.0)
    print(f"  -> {result_5}")
    assert result_5["route"] == "AMBIGUOUS", "Error: middle-band trigger should be AMBIGUOUS!"
    assert result_5["confidence"] == "low", "Error: expected 'low' confidence for middle-band trigger!"

    # Test 6: Soft-BNS-band chirp mass, but asymmetric/high-total-mass -> fails mass-structure check
    print("\n[Test 6] Simulating asymmetric trigger that fails the BNS mass-structure check (m1=3.5, m2=1.0)...")
    result_6 = router.route_event(3.5, 1.0)
    print(f"  -> {result_6}")
    assert result_6["route"] == "AMBIGUOUS", "Error: asymmetric trigger should be downgraded to AMBIGUOUS!"
    assert result_6["confidence"] == "medium", "Error: expected 'medium' confidence for downgraded trigger!"

    # Test 7: Spin confidence modifier on a BBH-like trigger
    print("\n[Test 7] Simulating GW150914-like trigger with high aligned spin (chi_eff=0.5)...")
    result_7 = router.route_event(35.6, 30.6, chi_eff=0.5)
    print(f"  -> {result_7}")
    assert result_7["route"] == "BBH", "Error: high-spin BBH-like trigger should still route to BBH!"
    assert result_7["confidence"] == "higher", "Error: expected spin to bump confidence to 'higher'!"

    # Test 8: Spin confidence modifier on a BNS-like trigger
    print("\n[Test 8] Simulating GW170817-like trigger with low aligned spin (chi_eff=0.05)...")
    result_8 = router.route_event(1.46, 1.27, chi_eff=0.05)
    print(f"  -> {result_8}")
    assert result_8["route"] == "BNS", "Error: low-spin BNS-like trigger should still route to BNS!"
    assert result_8["confidence"] == "higher", "Error: expected low spin to bump confidence to 'higher'!"

    print("\n🎉 ALL ROUTER TESTS PASSED SUCCESSFULLY! Hierarchical routing logic is sound.")


if __name__ == "__main__":
    run_router_tests()
