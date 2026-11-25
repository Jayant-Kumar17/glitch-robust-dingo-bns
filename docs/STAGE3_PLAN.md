# Stage 3 plan and reporting rules (fixed before any Stage 3 result)

Purpose: show what gate-aware importance sampling does on real Gravity Spy glitches,
where the Phase C localiser gate alone fell short.

## Cells

From `results/stress_real_glitches_v3_final` (Phase C), with the same recovery criterion
used everywhere (d_L median in [20, 50] Mpc, within 10 Mpc of the clean median, 90%
intervals overlap):

- **damaged**: the poisoned (ungated) posterior fails the criterion;
- **broken**: the poisoned posterior passes but the Phase C localiser gate made it fail.

Each cell is rebuilt from the cached excerpt at its archived placement. If the rebuilt
injected whitened SNR differs from the archived value by more than 1e-6 (relative), the
cell is skipped and counted as a mismatch; it is never analysed.

## Method per cell

Whitened-excess localiser at 8σ on H1, gate pads of 0.1 s and 0.2 s, band-limited
(20–23 Hz) zero-fill gate. Proposal: 10⁴ samples from the frozen network on the gated
package. Gated importance sampling with eps = 1e-3. The reference is DINGO importance
sampling on the clean GW170817 package (10⁴ samples).

## Reporting rules

1. A cell is **recovered by gated IS** only if its gated-IS n_eff ≥ 100 AND the d_L
   criterion passes. Cells with n_eff < 100 are reported as *untrusted*, never as recovered.
2. Every number is reported for each pad separately, next to the Phase C localiser-alone
   result, the network-only result and the naive-IS result on the same cells.
3. **Adaptive rule:** for each cell, take the pad whose gated IS has the larger n_eff. This
   uses no knowledge of the clean answer, so it is a legitimate operating rule. It is reported
   in addition to, not instead of, rule 2.
4. Cells where the localiser does not fire are counted as not recovered.
5. Results by morphology and Omicron-SNR bin are reported for the damaged cells. No subset
   is dropped after seeing the results.
