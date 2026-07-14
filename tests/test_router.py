from adapt.router import MatchedFilterRouter

def run_router_tests():
    print("--- Starting ADAPT Router Verification Tests ---")
    
    # Initialize our router with the standard 2.0 Solar Mass threshold
    router = MatchedFilterRouter(threshold_msun=2.0)
    
    # Event 1: GW170817 (BNS-like chirp mass ~ 1.188 M_sun)
    print("\n[Test 1] Simulating GW170817 (BNS Trigger)...")
    route_1 = router.route_event(1.188)
    assert route_1 == "light", "Error: GW170817 should be routed to 'light'!"
    
    # Event 2: GW150914 (BBH-like chirp mass ~ 28.6 M_sun)
    print("\n[Test 2] Simulating GW150914 (BBH Trigger)...")
    route_2 = router.route_event(28.6)
    assert route_2 == "heavy", "Error: GW150914 should be routed to 'heavy'!"
    
    # Event 3: Boundary edge case (Chirp mass ~ 1.95 M_sun)
    print("\n[Test 3] Simulating Low-Mass Candidate Trigger...")
    route_3 = router.route_event(1.95)
    assert route_3 == "light", "Error: 1.95 M_sun should be routed to 'light'!"
    
    print("\n🎉 ALL ROUTER TESTS PASSED SUCCESSFULLY! Framework routing logic is sound.")

if __name__ == "__main__":
    run_router_tests()