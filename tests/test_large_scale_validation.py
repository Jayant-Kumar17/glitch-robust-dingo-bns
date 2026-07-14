"""Large-scale validation: router vs. every confident confirmed event GWOSC has published.

This pulls published parameters (live, nothing hardcoded) for every
confident event across GWTC-1, GWTC-2.1, and GWTC-3 -- ~90+ real
confirmed gravitational-wave detections -- and runs each one through
the router, then compares the router's decision against an expected
type derived from the published component masses.

Ground-truth convention
------------------------
GWOSC's API does not publish an explicit "this was a BNS/BBH/NSBH"
label, so the expected type below is derived from a standard
astrophysical convention: a compact object is treated as a neutron
star candidate if its mass is below `NS_MAX_MASS_MSUN` (the
commonly-cited approximate maximum non-spinning neutron star mass),
and a black hole candidate otherwise:

  - both components below the line  -> BNS
  - both components above the line  -> BBH
  - one above, one below            -> NSBH

NSBH-specific routing is explicitly out of scope for this router
(documented in `router.py`), so NSBH events are EXPECTED to be
classified as BBH or AMBIGUOUS rather than counted as failures --
they're reported in their own bucket, not lumped into the BNS/BBH
accuracy figures.
"""

from adapt.gwosc_events import fetch_confident_catalog_events
from adapt.router import MatchedFilterRouter

NS_MAX_MASS_MSUN = 3.0

CATALOGS = ("GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident")


def expected_type(m1: float, m2: float) -> str:
    if m1 <= NS_MAX_MASS_MSUN and m2 <= NS_MAX_MASS_MSUN:
        return "BNS"
    if m1 > NS_MAX_MASS_MSUN and m2 > NS_MAX_MASS_MSUN:
        return "BBH"
    return "NSBH"


def run_large_scale_validation():
    print(f"Fetching published parameters live from GWOSC for catalogs: {', '.join(CATALOGS)}", flush=True)
    events = fetch_confident_catalog_events(CATALOGS)
    print(f"Retrieved {len(events)} unique confident events.\n", flush=True)

    router = MatchedFilterRouter()

    rows = []
    for name in sorted(events, key=lambda n: events[n]["gps"]):
        ev = events[name]
        exp = expected_type(ev["m1"], ev["m2"])
        decision = router.route_event(ev["m1"], ev["m2"], chi_eff=ev["chi_eff"])
        route = decision["route"]

        if exp == "NSBH":
            bucket = "nsbh_out_of_scope"
        elif route == exp:
            bucket = "match"
        elif route == "AMBIGUOUS":
            bucket = "ambiguous"
        else:
            bucket = "mismatch"

        rows.append(
            {
                "name": name,
                "catalog": ev["catalog"],
                "m1": ev["m1"],
                "m2": ev["m2"],
                "chi_eff": ev["chi_eff"],
                "expected": exp,
                "route": route,
                "confidence": decision["confidence"],
                "bucket": bucket,
            }
        )

    header = f"{'event':<24}{'catalog':<20}{'m1':>7}{'m2':>7}{'chi_eff':>9}{'expected':>10}{'route':>12}{'confidence':>12}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        status = {
            "match": "OK",
            "ambiguous": "AMBIGUOUS (flagged, not wrong)",
            "mismatch": "MISMATCH",
            "nsbh_out_of_scope": "NSBH (out of router scope)",
        }[r["bucket"]]
        print(
            f"{r['name']:<24}{r['catalog']:<20}{r['m1']:>7.2f}{r['m2']:>7.2f}{r['chi_eff']:>9.2f}"
            f"{r['expected']:>10}{r['route']:>12}{r['confidence']:>12}  {status}"
        )

    n_total = len(rows)
    n_nsbh = sum(1 for r in rows if r["bucket"] == "nsbh_out_of_scope")
    n_scored = n_total - n_nsbh
    n_match = sum(1 for r in rows if r["bucket"] == "match")
    n_ambiguous = sum(1 for r in rows if r["bucket"] == "ambiguous")
    n_mismatch = sum(1 for r in rows if r["bucket"] == "mismatch")

    print(f"\n{'=' * 70}")
    print(f"Total confident events tested: {n_total}")
    print(f"  NSBH (out of router scope, excluded from accuracy): {n_nsbh}")
    print(f"  Scored (BNS/BBH expected): {n_scored}")
    print(f"    Matched exactly:   {n_match}  ({100 * n_match / n_scored:.1f}%)")
    print(f"    Flagged AMBIGUOUS: {n_ambiguous}  ({100 * n_ambiguous / n_scored:.1f}%)")
    print(f"    Hard mismatch:     {n_mismatch}  ({100 * n_mismatch / n_scored:.1f}%)")

    if n_nsbh:
        print("\nNSBH events (excluded above, listed here since NSBH routing is a documented gap):")
        for r in rows:
            if r["bucket"] == "nsbh_out_of_scope":
                print(f"  {r['name']}: m1={r['m1']:.2f}, m2={r['m2']:.2f} -> router said {r['route']}")

    if n_mismatch:
        print("\nHard mismatches (BNS/BBH expected, router disagreed outright -- worth investigating):")
        for r in rows:
            if r["bucket"] == "mismatch":
                print(f"  {r['name']}: m1={r['m1']:.2f}, m2={r['m2']:.2f}, chi_eff={r['chi_eff']:.2f} -> expected {r['expected']}, router said {r['route']}")

    assert n_mismatch == 0, f"{n_mismatch} confident BNS/BBH event(s) were hard-misclassified -- see list above."
    print(f"\n🎉 No hard mismatches across {n_scored} confident BNS/BBH events. AMBIGUOUS flags above are conservative, not wrong.")


if __name__ == "__main__":
    run_large_scale_validation()
