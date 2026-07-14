"""Full real-event validation: real raw strain + published parameters + router.

For a confirmed event, this:
1. Downloads the actual real raw strain data GWOSC recorded around it
   (the small, pre-cut ~32s segment, not the ~500MB continuous archive).
2. Fetches the published, peer-reviewed parameter estimate for the same
   event, live from GWOSC's event API.
3. Runs those published parameters through the router.
4. Compares the router's decision against the event's known, verified
   classification.

Every step below prints its progress as it happens -- nothing runs
silently in the background.
"""

from gwpy.timeseries import TimeSeries

from adapt.gwosc_events import fetch_event_strain, fetch_published_parameters
from adapt.router import MatchedFilterRouter

REAL_EVENTS = [
    {
        "name": "GW150914",
        "catalog": "GWTC-1-confident",
        "version": 3,
        "detector": "H1",
        "verified_type": "BBH",
        "description": "The first-ever gravitational-wave detection (LIGO, 2015).",
    },
    {
        "name": "GW170817",
        "catalog": "GWTC-1-confident",
        "version": 3,
        "detector": "H1",
        "verified_type": "BNS",
        "description": "The first multi-messenger gravitational-wave event (LIGO/Virgo, 2017).",
    },
]


def run_real_event_full_validation():
    router = MatchedFilterRouter()

    for event in REAL_EVENTS:
        name = event["name"]
        print(f"\n{'=' * 70}")
        print(f"Validating against {name}: {event['description']}")
        print(f"{'=' * 70}")

        print("\n--- Step 1: Download the real raw strain data ---")
        strain_path = fetch_event_strain(name, event["catalog"], event["detector"], version=event["version"])
        strain = TimeSeries.read(strain_path, format="hdf5.gwosc")
        print(f"Loaded {len(strain)} real samples, duration={strain.duration}, sample_rate={strain.sample_rate}")
        print(f"GPS start: {strain.t0}")
        print(f"Strain stats: min={strain.value.min():.3e}, max={strain.value.max():.3e}, std={strain.value.std():.3e}")

        print("\n--- Step 2: Fetch the published (peer-reviewed) parameters, live from GWOSC ---")
        published = fetch_published_parameters(name, event["catalog"], version=event["version"])
        print(f"Published parameters: {published}")

        print("\n--- Step 3: Run the published parameters through the router ---")
        decision = router.route_event(published["m1"], published["m2"], chi_eff=published["chi_eff"])
        print(f"Router decision: {decision}")

        print("\n--- Step 4: Compare against the actual, verified classification ---")
        print(f"Verified type: {event['verified_type']}")
        print(f"Router says:   {decision['route']} (confidence: {decision['confidence']})")
        assert decision["route"] == event["verified_type"], (
            f"Error: {name} is verified as {event['verified_type']}, but the router said {decision['route']}!"
        )
        print("==> MATCH")

    print(f"\n{'=' * 70}")
    print("🎉 ALL REAL-EVENT FULL VALIDATION CHECKS PASSED -- real strain confirmed, real router decisions match verified science.")


if __name__ == "__main__":
    run_real_event_full_validation()
