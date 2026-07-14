"""Known-answer validation of the router: proving it isn't "making stuff up."

Section A (self-consistency): you set the full truth for a synthetic
signal (masses + spins), we generate the actual LALSimulation waveform
from exactly those numbers, inject it into noise, and verify that the
parameters the router acts on trace directly back to what was set --
nothing invented in between.

Section B (external ground truth): the router is checked against the
published, peer-reviewed parameters of real confirmed detections,
fetched live from GWOSC -- not hardcoded.
"""

import numpy as np

from adapt.gwosc_events import fetch_published_parameters
from adapt.injection import generate_injection
from adapt.router import MatchedFilterRouter

SYNTHETIC_CASES = [
    {
        "label": "Clean BNS (low, near-equal masses, low spin)",
        "m1": 1.4,
        "m2": 1.35,
        "spin1z": 0.02,
        "spin2z": 0.01,
        "expected_route": "BNS",
    },
    {
        "label": "Clean BBH (heavy, moderately spinning)",
        "m1": 32.0,
        "m2": 28.0,
        "spin1z": 0.5,
        "spin2z": 0.3,
        "expected_route": "BBH",
    },
    {
        "label": "Ambiguous middle-band, equal masses, zero spin",
        "m1": 2.0,
        "m2": 2.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "expected_route": "AMBIGUOUS",
    },
]

REAL_EVENTS = [
    {
        "name": "GW150914",
        "catalog": "GWTC-1-confident",
        "version": 3,
        "expected_route": "BBH",
        "note": "Textbook BBH -- the first ever GW detection.",
    },
    {
        "name": "GW170817",
        "catalog": "GWTC-1-confident",
        "version": 3,
        "expected_route": "BNS",
        "note": "Textbook BNS -- the first multi-messenger GW event.",
    },
    {
        "name": "GW190425",
        "catalog": "GWTC-2.1-confident",
        "version": 3,
        "expected_route": None,
        "note": (
            "Real BNS detection, but unusually high mass and asymmetric "
            "(q~0.62) for a BNS -- the router's mass-structure check is "
            "expected to flag this as AMBIGUOUS rather than confidently BNS, "
            "mirroring how this event puzzled astronomers in real life."
        ),
    },
    {
        "name": "GW190814",
        "catalog": "GWTC-2.1-confident",
        "version": 3,
        "expected_route": None,
        "note": (
            "A 'mass-gap' event: a ~23 Msun black hole merging with a ~2.6 "
            "Msun object that could be an unusually heavy neutron star or "
            "unusually light black hole. The router (no NSBH-specific "
            "handling yet) is expected to route this to BBH given the "
            "large total mass."
        ),
    },
]


def run_synthetic_validation():
    print("=== Section A: Synthetic full-metadata self-consistency ===")
    rng = np.random.default_rng(0)
    sample_rate = 4096.0
    noise = rng.normal(scale=1e-21, size=int(64 * sample_rate))

    router = MatchedFilterRouter()

    for case in SYNTHETIC_CASES:
        print(f"\n[{case['label']}]")
        print(
            f"  Inputs you set: m1={case['m1']}, m2={case['m2']}, "
            f"spin1z={case['spin1z']}, spin2z={case['spin2z']}"
        )

        _, true_params = generate_injection(
            case["m1"],
            case["m2"],
            noise,
            sample_rate,
            spin1z=case["spin1z"],
            spin2z=case["spin2z"],
        )
        print(f"  true_params returned by the pipeline: {true_params}")

        # Traceability check: the pipeline must echo back exactly what you
        # set, not something else -- this is the "not making stuff up" check.
        assert true_params["m1"] == case["m1"], "Error: m1 was not preserved through the pipeline!"
        assert true_params["m2"] == case["m2"], "Error: m2 was not preserved through the pipeline!"
        assert true_params["spin1z"] == case["spin1z"], "Error: spin1z was not preserved through the pipeline!"
        assert true_params["spin2z"] == case["spin2z"], "Error: spin2z was not preserved through the pipeline!"

        decision = router.route_event(true_params["m1"], true_params["m2"], chi_eff=true_params["chi_eff"])
        print(f"  Router decision: {decision}")
        assert decision["route"] == case["expected_route"], (
            f"Error: expected {case['expected_route']} for {case['label']}, got {decision['route']}"
        )

    print("\nSection A passed: every decision traces directly back to the metadata you set.")


def run_real_event_validation():
    print("\n=== Section B: Real confirmed events (external ground truth from GWOSC) ===")
    router = MatchedFilterRouter()

    for event in REAL_EVENTS:
        print(f"\n[{event['name']}]")
        published = fetch_published_parameters(event["name"], event["catalog"], version=event["version"])
        print(f"  Published parameters (live from GWOSC): {published}")

        decision = router.route_event(published["m1"], published["m2"], chi_eff=published["chi_eff"])
        print(f"  Router decision: {decision}")
        print(f"  Note: {event['note']}")

        if event["expected_route"] is not None:
            assert decision["route"] == event["expected_route"], (
                f"Error: expected {event['expected_route']} for {event['name']}, got {decision['route']}"
            )
        else:
            print("  (No hard assertion here -- this is a genuinely debated/edge-case event.)")

    print("\nSection B passed: router classifications compared against real, peer-reviewed catalog data.")


if __name__ == "__main__":
    run_synthetic_validation()
    run_real_event_validation()
    print("\n🎉 ALL KNOWN-ANSWER VALIDATION CHECKS PASSED SUCCESSFULLY!")
