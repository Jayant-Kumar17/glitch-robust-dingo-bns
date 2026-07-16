"""Distributed simulation campaign: stress-test the router on a synthetic population.

Implements the spirit of the ADAPT report's Section 4.3 simulation loop,
adapted to the current hierarchical MatchedFilterRouter (m1, m2, chi_eff ->
BNS / BBH / AMBIGUOUS), not the old single-threshold light/heavy API.

For each sample:
1. Flip a coin: BNS or BBH.
2. Draw astrophysically plausible component masses (and spins).
3. Compute true derived quantities (chirp mass, total mass, q, chi_eff).
4. Optionally generate a real LALSimulation/PyCBC waveform from those
   parameters (proves the draw is physically realizable).
5. Mock matched-filter recovery by adding small Gaussian measurement noise
   to the component masses and spins.
6. Feed the noisy recovered parameters into the router and score.

Results are printed as a performance report and written to
results/simulation_batch.csv.
"""

import csv
import os
import time

import numpy as np
from pycbc.waveform import get_td_waveform

from adapt.physics import chirp_mass, effective_spin, mass_ratio, total_mass
from adapt.router import MatchedFilterRouter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "simulation_batch.csv")

# Relative measurement uncertainty on recovered component masses / spins
# (mocks low-latency matched-filter template scatter).
MASS_NOISE_FRAC = 0.02
SPIN_NOISE_ABS = 0.02


def sample_event(rng: np.random.Generator) -> dict:
    """Draw one synthetic event with known ground-truth class and parameters."""
    true_class = rng.choice(["BNS", "BBH"])

    if true_class == "BNS":
        m1 = float(rng.uniform(1.0, 2.0))
        m2 = float(rng.uniform(1.0, m1))
        spin1z = float(rng.uniform(-0.05, 0.05))
        spin2z = float(rng.uniform(-0.05, 0.05))
    else:
        m1 = float(rng.uniform(5.0, 50.0))
        m2 = float(rng.uniform(5.0, m1))
        spin1z = float(rng.uniform(-0.8, 0.8))
        spin2z = float(rng.uniform(-0.8, 0.8))

    return {
        "true_class": true_class,
        "m1": m1,
        "m2": m2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "mc": chirp_mass(m1, m2),
        "mtot": total_mass(m1, m2),
        "q": mass_ratio(m1, m2),
        "chi_eff": effective_spin(m1, m2, spin1z, spin2z),
    }


def mock_matched_filter_recovery(truth: dict, rng: np.random.Generator) -> dict:
    """Perturb true params with Gaussian noise to mimic MF template scatter."""
    m1 = max(0.1, truth["m1"] + rng.normal(0.0, MASS_NOISE_FRAC * truth["m1"]))
    m2 = max(0.1, truth["m2"] + rng.normal(0.0, MASS_NOISE_FRAC * truth["m2"]))
    if m2 > m1:
        m1, m2 = m2, m1

    spin1z = float(np.clip(truth["spin1z"] + rng.normal(0.0, SPIN_NOISE_ABS), -0.99, 0.99))
    spin2z = float(np.clip(truth["spin2z"] + rng.normal(0.0, SPIN_NOISE_ABS), -0.99, 0.99))

    return {
        "m1": float(m1),
        "m2": float(m2),
        "spin1z": spin1z,
        "spin2z": spin2z,
        "mc": chirp_mass(m1, m2),
        "mtot": total_mass(m1, m2),
        "q": mass_ratio(m1, m2),
        "chi_eff": effective_spin(m1, m2, spin1z, spin2z),
    }


def generate_waveform(m1: float, m2: float, spin1z: float, spin2z: float, sample_rate: float = 4096.0) -> int:
    """Generate a time-domain waveform; return its length in samples."""
    # BNS signals are long; start a bit higher in frequency for speed while
    # still requiring a real LALSimulation call.
    f_lower = 30.0 if max(m1, m2) < 3.0 else 20.0
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=1.0 / sample_rate,
        f_lower=f_lower,
    )
    return len(hp)


def score_decision(true_class: str, route: str) -> str:
    if route == true_class:
        return "match"
    if route == MatchedFilterRouter.AMBIGUOUS:
        return "ambiguous"
    return "mismatch"


def run_simulation_campaign(
    num_samples: int = 1000,
    waveform_samples: int = 50,
    seed: int = 42,
):
    print("=" * 60)
    print(f" ADAPT SIMULATION CAMPAIGN ({num_samples} samples)")
    print("=" * 60)
    print(f"Mass measurement noise: {100 * MASS_NOISE_FRAC:.1f}% relative Gaussian")
    print(f"Waveform generation: first {waveform_samples} samples (LALSimulation/PyCBC)")
    print()

    rng = np.random.default_rng(seed)
    router = MatchedFilterRouter()
    rows = []

    n_match = n_ambiguous = n_mismatch = 0
    false_bns_as_bbh = false_bbh_as_bns = 0
    n_waveforms_ok = 0
    t0 = time.time()

    for i in range(num_samples):
        truth = sample_event(rng)

        waveform_len = None
        if i < waveform_samples:
            try:
                waveform_len = generate_waveform(truth["m1"], truth["m2"], truth["spin1z"], truth["spin2z"])
                n_waveforms_ok += 1
            except Exception as exc:
                print(f"  [{i + 1}] waveform failed for {truth['true_class']} "
                      f"m1={truth['m1']:.2f}, m2={truth['m2']:.2f}: {exc}", flush=True)

        recovered = mock_matched_filter_recovery(truth, rng)
        decision = router.route_event(recovered["m1"], recovered["m2"], chi_eff=recovered["chi_eff"])
        bucket = score_decision(truth["true_class"], decision["route"])

        if bucket == "match":
            n_match += 1
        elif bucket == "ambiguous":
            n_ambiguous += 1
        else:
            n_mismatch += 1
            if truth["true_class"] == "BNS" and decision["route"] == "BBH":
                false_bns_as_bbh += 1
            elif truth["true_class"] == "BBH" and decision["route"] == "BNS":
                false_bbh_as_bns += 1

        rows.append(
            {
                "sample": i,
                "true_class": truth["true_class"],
                "true_m1": truth["m1"],
                "true_m2": truth["m2"],
                "true_mc": truth["mc"],
                "true_chi_eff": truth["chi_eff"],
                "recovered_m1": recovered["m1"],
                "recovered_m2": recovered["m2"],
                "recovered_mc": recovered["mc"],
                "recovered_chi_eff": recovered["chi_eff"],
                "route": decision["route"],
                "confidence": decision["confidence"],
                "bucket": bucket,
                "waveform_samples": waveform_len if waveform_len is not None else "",
            }
        )

        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{num_samples}] elapsed={elapsed:.1f}s  "
                f"match={n_match} ambiguous={n_ambiguous} mismatch={n_mismatch}",
                flush=True,
            )

    accuracy = 100.0 * n_match / num_samples
    safe_rate = 100.0 * (n_match + n_ambiguous) / num_samples

    print("\n" + "=" * 50)
    print("      ADAPT ROUTER PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Total simulated events     : {num_samples}")
    print(f"Waveforms successfully made: {n_waveforms_ok}/{waveform_samples}")
    print(f"Exact route matches        : {n_match}  ({accuracy:.2f}%)")
    print(f"Conservative AMBIGUOUS     : {n_ambiguous}  ({100.0 * n_ambiguous / num_samples:.2f}%)")
    print(f"Hard mismatches            : {n_mismatch}  ({100.0 * n_mismatch / num_samples:.2f}%)")
    print(f"Safe rate (match+ambiguous): {safe_rate:.2f}%")
    print("-" * 50)
    print("Misclassification breakdown:")
    print(f"  False BBH  (true BNS -> BBH) : {false_bns_as_bbh}")
    print(f"  False BNS  (true BBH -> BNS) : {false_bbh_as_bns}")
    print("=" * 50)

    if n_mismatch == 0 and accuracy >= 95.0:
        print("CRITICAL VALIDATION PASSED: router is robust under mocked MF noise.")
    elif n_mismatch == 0:
        print("No hard mismatches; exact-match rate below 95% due to AMBIGUOUS flags.")
    else:
        print("WARNING: hard mismatches present -- inspect results CSV near class boundary.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results table saved to: {RESULTS_CSV_PATH}")

    assert n_mismatch == 0, f"{n_mismatch} hard misclassifications in the simulation campaign."
    return {
        "num_samples": num_samples,
        "accuracy": accuracy,
        "n_match": n_match,
        "n_ambiguous": n_ambiguous,
        "n_mismatch": n_mismatch,
        "safe_rate": safe_rate,
    }


if __name__ == "__main__":
    run_simulation_campaign(num_samples=1000, waveform_samples=50)
