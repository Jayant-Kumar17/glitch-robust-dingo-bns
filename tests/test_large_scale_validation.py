"""Large-scale validation: router vs. every confirmed event's OFFICIAL classification.

Pulls published parameters live from GWOSC's cumulative ``GWTC`` catalog
(the full up-to-date transient list through GWTC-5.0 / O4b, not the old
GWTC-1/2.1/3 trio that only covered ~90 events). Events without published
component masses yet (common for very recent O4 triggers) are skipped.

For each event with masses, the router is scored against LVK's own
real-time source classification (p_astro from GraceDB) where public,
otherwise a clearly-labeled mass-threshold heuristic.
"""

import csv
import os
from datetime import datetime

from adapt.gwosc_events import fetch_confident_catalog_events, fetch_published_parameters, fetch_source_classification
from adapt.router import MatchedFilterRouter

NS_MAX_MASS_MSUN = 3.0
# Cumulative GWTC list: latest parameters for all releases through GWTC-5.0.
CATALOGS = ("GWTC",)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

OUT_OF_SCOPE_LABELS = {"NSBH", "MassGap"}


def heuristic_type(m1: float, m2: float) -> str:
    if m1 <= NS_MAX_MASS_MSUN and m2 <= NS_MAX_MASS_MSUN:
        return "BNS"
    if m1 > NS_MAX_MASS_MSUN and m2 > NS_MAX_MASS_MSUN:
        return "BBH"
    return "NSBH"


def run_large_scale_validation():
    print(f"Fetching event list live from GWOSC catalog: {', '.join(CATALOGS)} (through GWTC-5.0)", flush=True)
    events = fetch_confident_catalog_events(CATALOGS)
    names = sorted(events, key=lambda n: events[n]["gps"])
    print(f"Retrieved {len(names)} events with published component masses.\n", flush=True)

    router = MatchedFilterRouter()
    rows = []

    for i, name in enumerate(names, start=1):
        bulk = events[name]
        if i % 25 == 0 or i == len(names):
            print(f"[{i}/{len(names)}] {name}: fetching GraceDB classification...", flush=True)

        # Use the event's originating release catalog (e.g. GWTC-5.0) for gracedb_id lookup.
        detail = fetch_published_parameters(name, bulk["source_catalog"])
        gracedb_id = detail["gracedb_id"]
        official = fetch_source_classification(gracedb_id)

        if official is not None:
            expected = official["label"]
            expected_source = f"official (GraceDB p_astro, {gracedb_id})"
        else:
            expected = heuristic_type(bulk["m1"], bulk["m2"])
            expected_source = "heuristic (mass threshold, no public p_astro for this event)"

        decision = router.route_event(bulk["m1"], bulk["m2"], chi_eff=bulk["chi_eff"])
        route = decision["route"]

        if expected in OUT_OF_SCOPE_LABELS:
            bucket = "out_of_scope"
        elif route == expected:
            bucket = "match"
        elif route == "AMBIGUOUS":
            bucket = "ambiguous"
        else:
            bucket = "mismatch"

        rows.append(
            {
                "name": name,
                "source_catalog": bulk["source_catalog"],
                "m1": bulk["m1"],
                "m2": bulk["m2"],
                "chi_eff": bulk["chi_eff"],
                "expected": expected,
                "expected_source": expected_source,
                "route": route,
                "confidence": decision["confidence"],
                "bucket": bucket,
            }
        )

    status_label = {
        "match": "OK",
        "ambiguous": "AMBIGUOUS (flagged, not wrong)",
        "mismatch": "MISMATCH",
    }

    print(f"\n{'=' * 130}")
    header = f"{'event':<24}{'m1':>7}{'m2':>7}{'chi_eff':>9}  {'expected':<9}  {'source':<50}  {'route':<11}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        status = status_label.get(r["bucket"], f"{r['expected']} (out of router scope)")
        print(
            f"{r['name']:<24}{r['m1']:>7.2f}{r['m2']:>7.2f}{r['chi_eff']:>9.2f}  "
            f"{r['expected']:<9}  {r['expected_source']:<50}  {r['route']:<11}  {status}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_path = os.path.join(RESULTS_DIR, f"large_scale_validation_{timestamp}.csv")
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "source_catalog",
                "m1",
                "m2",
                "chi_eff",
                "expected",
                "expected_source",
                "route",
                "confidence",
                "bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results table saved to: {results_csv_path}")

    n_total = len(rows)
    n_official = sum(1 for r in rows if "official" in r["expected_source"])
    n_out_of_scope = sum(1 for r in rows if r["bucket"] == "out_of_scope")
    n_scored = n_total - n_out_of_scope
    n_match = sum(1 for r in rows if r["bucket"] == "match")
    n_ambiguous = sum(1 for r in rows if r["bucket"] == "ambiguous")
    n_mismatch = sum(1 for r in rows if r["bucket"] == "mismatch")

    print(f"\n{'=' * 70}")
    print(f"Total events tested (with published masses): {n_total}")
    print(f"  Using LVK's official real-time classification (p_astro): {n_official}")
    print(f"  Using mass-threshold heuristic (no public p_astro): {n_total - n_official}")
    print(f"\n  Out of router scope (official label NSBH/MassGap): {n_out_of_scope}")
    print(f"  Scored (BNS/BBH expected): {n_scored}")
    print(f"    Matched exactly:   {n_match}  ({100 * n_match / n_scored:.1f}%)")
    print(f"    Flagged AMBIGUOUS: {n_ambiguous}  ({100 * n_ambiguous / n_scored:.1f}%)")
    print(f"    Hard mismatch:     {n_mismatch}  ({100 * n_mismatch / n_scored:.1f}%)")

    if n_out_of_scope:
        print("\nOut-of-scope events (NSBH / MassGap officially, per LVK's own classifier):")
        for r in rows:
            if r["bucket"] == "out_of_scope":
                print(f"  {r['name']}: m1={r['m1']:.2f}, m2={r['m2']:.2f}, official={r['expected']} -> router said {r['route']}")

    if n_mismatch:
        print("\nHard mismatches (worth investigating):")
        for r in rows:
            if r["bucket"] == "mismatch":
                print(f"  {r['name']}: m1={r['m1']:.2f}, m2={r['m2']:.2f}, chi_eff={r['chi_eff']:.2f} -> expected {r['expected']} ({r['expected_source']}), router said {r['route']}")

    assert n_mismatch == 0, f"{n_mismatch} confident BNS/BBH event(s) were hard-misclassified -- see list above."
    print(f"\n🎉 No hard mismatches across {n_scored} confident BNS/BBH events, checked against LVK's own official classification where public.")


if __name__ == "__main__":
    run_large_scale_validation()
