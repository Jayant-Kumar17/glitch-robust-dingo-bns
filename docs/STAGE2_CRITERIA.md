# Stage 2 pre-registered criteria (gate-aware importance sampling)

Written and committed before any Stage 2 result existed. The driver
`examples/gated_is_stage2.py` applies these mechanically and writes
`results/gated_is_stage2/STAGE2_VERDICT.md`.

## Hard checks on real GW170817 data (V1)

- **A.** With an empty gate, the uniform-grid log-likelihood used here equals DINGO's own
  uniform-grid `StationaryGaussianGWLikelihood` to a relative 1e-8 for 12 posterior samples.
- **B.** A compact 150 Hz burst of whitened SNR 1000, placed inside a ±0.2 s H1 gate,
  changes the gated log-likelihood by less than 1e-3 of its effect on the ordinary
  log-likelihood (which must exceed 100) for 6 posterior samples; the same gate moved by
  the 2 s DINGO time buffer must NOT be invariant (change ≥ 0.1 of the ordinary effect).

If V1 fails, the run stops and nothing else is reported.

## Decisive set

Ten clean GW170817 cells with a forced ±0.4 s H1 gate centred at the first ten Phase B
placements (seed 0, t_rel uniform in [-2, -0.2] s). Gate recipe: band-limited strain
(20–23 Hz ramp), zero fill. In Phase B the network alone recovered 7/20 such cells.

- **V2.** Route-A (network on gated data, reweighted with the gated likelihood) sample
  efficiency ≥ 1% in at least 7 of 10 cells.
- **V3.** Among cells where both routes have n_eff ≥ 100, the luminosity-distance JS divergence
  between route A and route B (clean-data proposal, same gated likelihood) is ≤ 0.05 nat in at
  least 80% of them.
- **V4.** Route-A luminosity distance passes the paper's recovery criterion (median in
  [20, 50] Mpc, within 10 Mpc of the clean importance-sampled median, 90% intervals overlap)
  in at least 8 of 10 cells.
- **V5 (reported, not decisive).** Change of the d_L median when the out-of-band regulariser
  eps goes from 1e-2 to 1e-4.

## Decision

- **GO:** V1–V4 all pass. Gate-aware importance sampling becomes the main result.
- **PARTIAL:** V1 passes and V2 or V4 passes. Report as a limited extension; decide with the author.
- **NO-GO:** otherwise. Publish the audit alone; gate-aware IS is not pursued further now.

Widths 0.2 s and 0.8 s and the raw unsubtracted Livingston frame are reported alongside but
do not enter the decision.
