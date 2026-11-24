# Stage 3b plan: more samples for the untrusted cells (fixed before any Stage 3b result)

Stage 3 (10⁴ proposal samples) left 18 cells *untrusted* at the primary ±0.1 s pad
(8 damaged, 10 broken): gated importance sampling had n_eff < 100, so by
docs/STAGE3_PLAN.md they could not be counted as recovered.

Stage 3b reruns exactly those 18 (group, cell, pad) combinations with 10⁵ proposal
samples (5×10⁴ if 10⁵ does not fit in memory), everything else unchanged: same
injections (rebuilt and checked against Phase C), same localiser, same gated likelihood
(eps = 1e-3), same clean reference size (10⁴), same recovery criterion, same trust rule
(n_eff ≥ 100).

Reporting:
1. For each of the 18 cells, report n_eff, efficiency, d_L median and whether it is now
   trusted and recovered.
2. The paper's headline counts for ±0.1 s become: Stage 3 trusted recoveries + Stage 3b
   trusted recoveries, with the proposal size stated for each. Cells still untrusted stay
   untrusted. No other cell is rerun and none is dropped.
3. The compute cost per cell at 10⁵ samples is reported alongside.
