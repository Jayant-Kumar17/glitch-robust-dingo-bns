"""Verification tests for the component-mass boundary MatchedFilterRouter.

The router classifies strictly on component masses:
  - BNS  : both masses <= ns_max (2.2 Msun)
  - BBH  : both masses >= bh_min (5.0 Msun)
  - AMBIGUOUS : everything else (NSBH, lower mass gap) -> offline analysis
"""

from adapt.router import MatchedFilterRouter


def run_router_tests():
    print("--- Starting ADAPT Boundary Router Verification Tests ---")

    router = MatchedFilterRouter()

    # Test 1: GW170817-like clean BNS (both masses below the NS ceiling).
    print("\n[Test 1] Clean BNS trigger (m1=1.46, m2=1.27)...")
    result_1 = router.route_event(1.46, 1.27)
    print(f"  -> {result_1}")
    assert result_1["route"] == "BNS", "Error: light near-equal masses should route to BNS!"
    assert result_1["confidence"] == 1.0, "Error: clean BNS should have confidence 1.0!"

    # Test 2: GW150914-like clean BBH (both masses above the BH floor).
    print("\n[Test 2] Clean BBH trigger (m1=35.6, m2=30.6)...")
    result_2 = router.route_event(35.6, 30.6)
    print(f"  -> {result_2}")
    assert result_2["route"] == "BBH", "Error: heavy masses should route to BBH!"
    assert result_2["confidence"] == 1.0, "Error: clean BBH should have confidence 1.0!"

    # Test 3: Very light system, still BNS.
    print("\n[Test 3] Very light trigger (m1=0.4, m2=0.4)...")
    result_3 = router.route_event(0.4, 0.4)
    print(f"  -> {result_3}")
    assert result_3["route"] == "BNS", "Error: very light system should route to BNS!"
    assert result_3["confidence"] == 1.0, "Error: clean BNS should have confidence 1.0!"

    # Test 4: Very heavy system, still BBH.
    print("\n[Test 4] Very heavy trigger (m1=50, m2=50)...")
    result_4 = router.route_event(50.0, 50.0)
    print(f"  -> {result_4}")
    assert result_4["route"] == "BBH", "Error: very heavy system should route to BBH!"
    assert result_4["confidence"] == 1.0, "Error: clean BBH should have confidence 1.0!"

    # Test 5: Both masses exactly at the NS ceiling -> still BNS.
    print("\n[Test 5] Both masses at the NS ceiling (m1=2.0, m2=2.0)...")
    result_5 = router.route_event(2.0, 2.0)
    print(f"  -> {result_5}")
    assert result_5["route"] == "BNS", "Error: masses within the NS ceiling should route to BNS!"
    assert result_5["confidence"] == 1.0, "Error: clean BNS should have confidence 1.0!"

    # Test 6: Primary in the lower mass gap (2.2-5.0) -> AMBIGUOUS.
    print("\n[Test 6] Primary in the lower mass gap (m1=3.5, m2=1.0)...")
    result_6 = router.route_event(3.5, 1.0)
    print(f"  -> {result_6}")
    assert result_6["route"] == "AMBIGUOUS", "Error: mass-gap primary should route to AMBIGUOUS!"
    assert result_6["confidence"] == 0.5, "Error: AMBIGUOUS should have confidence 0.5!"

    # Test 7: Asymmetric NSBH-like system (heavy primary, NS secondary) -> AMBIGUOUS.
    print("\n[Test 7] NSBH-like trigger (m1=23.3, m2=2.6)...")
    result_7 = router.route_event(23.3, 2.6)
    print(f"  -> {result_7}")
    assert result_7["route"] == "AMBIGUOUS", "Error: NSBH-like system should route to AMBIGUOUS!"
    assert result_7["confidence"] == 0.5, "Error: AMBIGUOUS should have confidence 0.5!"

    # Test 8: Another NSBH-like system (BH primary, light NS secondary) -> AMBIGUOUS.
    print("\n[Test 8] NSBH-like trigger (m1=8.0, m2=1.5)...")
    result_8 = router.route_event(8.0, 1.5)
    print(f"  -> {result_8}")
    assert result_8["route"] == "AMBIGUOUS", "Error: NSBH-like system should route to AMBIGUOUS!"
    assert result_8["confidence"] == 0.5, "Error: AMBIGUOUS should have confidence 0.5!"

    # Test 9: Mass order should not matter (swapped inputs give the same result).
    print("\n[Test 9] Mass-order invariance (m1=2.6, m2=23.3 == Test 7 swapped)...")
    result_9 = router.route_event(2.6, 23.3)
    print(f"  -> {result_9}")
    assert result_9 == result_7, "Error: routing should be independent of m1/m2 order!"

    # Test 10: chi_eff is accepted but does not change the boundary decision.
    print("\n[Test 10] chi_eff accepted but ignored (m1=35.6, m2=30.6, chi_eff=0.5)...")
    result_10 = router.route_event(35.6, 30.6, chi_eff=0.5)
    print(f"  -> {result_10}")
    assert result_10 == result_2, "Error: chi_eff should not affect the boundary decision!"

    print("\nALL ROUTER TESTS PASSED SUCCESSFULLY! Boundary routing logic is sound.")


if __name__ == "__main__":
    run_router_tests()
