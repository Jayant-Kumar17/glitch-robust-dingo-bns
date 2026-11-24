# Referee-revision plan (fixed before any revision run)

Commit this file before any Phase 2 job starts. It fixes what will be run and how every new
number will be computed. Analyses not listed here may still be reported, but they are labelled
exploratory in the paper.

## Definitions (unchanged from docs/STAGE3_PLAN.md unless stated)

- **Pass.** The d_L median is in [20, 50] Mpc, it is within 10 Mpc of the clean median, and the 90% intervals overlap. The reference is the **pre-registered Stage 3 clean reference** (`results/gated_is_stage3/clean_reference.json`, 10^4 samples). All pass/fail counts use it. Note that the [20, 50] clause is implied by the other two for a clean median of 34.2 Mpc; it is kept so the code is unchanged.
- **Trusted.** The method's own effective sample size satisfies n_eff >= 100, at 10^4 proposal samples unless stated.
- **Methods compared, each with its own n_eff >= 100 flag.**
  - GIS: gated importance sampling, i.e. the network on the gated package, reweighted with the gated likelihood.
  - OIS: ordinary DINGO importance sampling on the same gated package. Its n_eff is its sample efficiency times 10^4.
  - UIS: ordinary DINGO importance sampling on the **ungated**, glitch-contaminated package.
  - NET: unweighted network samples on the gated package. NET has no flag.
- **Cell groups** (from `results/stress_real_glitches_v3_final/results.csv`, excluding errors, noise controls and placements with injected_rho_w <= 1):
  - damaged: the ungated network fails;
  - broken: the ungated network passes and the Phase C localiser-gated network fails;
  - fine: the ungated network passes and the Phase C localiser-gated network passes.
- **Cluster.** A glitch, identified by `gs_event_time`. Cluster-bootstrap intervals resample glitches with replacement (10^4 draws, percentile 95%).

## Phase 2 runs

| id | what | n | output |
| --- | --- | --- | --- |
| R1 | Clean reference v2: DINGO-IS on the clean GW170817 package, seed 20261125 | 10^5 | `results/revision/clean_ref_v2/` |
| R6 | Weights check: our uniform-grid reweighting with an empty gate vs DINGO's own weights, on the first 10^4 samples of R1 | 10^4 | `results/revision/weights_check.json` |
| R2 | UIS on all 195 real-glitch cells (ungated contaminated package) | 10^4 each | `results/revision/ungated_is/` |
| R3 | GIS/OIS/NET on the fine cells, pad 0.1 s, same code path as Stage 3 | 10^4 each | `results/revision/stage3_fine/` |
| R4 | Independent proposals on 8 cells (route A repeat; B = clean-data proposal from R1; C = noise-fill gate) | A,C 10^4; B 2x10^4 | `results/revision/routes/` |
| R5 | Timing on an idle machine (network, DINGO-IS, gated likelihood) | see prompt | `results/revision/timing.json` |
| R7 (optional) | Two synthetic events in Gaussian noise (SNR ~32 and ~16), 10 forced ±0.2 s gates each | 10^4 | `results/revision/synthetic/` |

**R4 cell choice (fixed now).**
1. Every Stage 3 analysis that is trusted but fails (expected: 2).
2. The 3 trusted, passing damaged analyses with the smallest n_eff.
3. The 3 untrusted damaged ±0.1 s analyses with the largest n_eff.

**R7 pass rule.** This is a new event, so the 34.2 Mpc window does not apply. A cell passes if its d_L median is within 0.25 x (clean 90% width) of the clean median of that synthetic event, and the 90% intervals overlap.

## Reported statistics (examples/revision_analysis.py)

1. For every method, group and pad: the counts trusted & pass, trusted & fail, untrusted & pass, untrusted & fail.
2. Precision, meaning trusted results that pass, with Wilson and cluster-bootstrap intervals.
3. Refusal rate (untrusted fraction), with a cluster interval. On the fine group, this is the false-refusal rate.
4. NET pass rate among untrusted analyses, **damaged and broken reported separately**. Broken cells fail under NET by construction.
5. Exact McNemar tests on the same cells, per group and pad:
   - GIS trusted & pass vs NET pass;
   - OIS pass vs NET pass;
   - GIS vs OIS.
6. Trust-threshold sensitivity at n_eff ∈ {25, 50, 100, 200, 500}. This is **exploratory**.
7. PSIS Pareto k-hat (Vehtari et al.) for every GIS, OIS and UIS analysis. Reported: the crosstab against trust and pass, and k-hat of the trusted failures.
8. One-dimensional JS divergence against the R1 reference for chirp mass, q, θ_JN, d_L, Λ1, Λ2, Λ̃ and t_c, plus the maximum over all archived parameters.
   - JS is reported separately for GIS (split by trust), OIS, NET and UIS.
   - A resampling floor is reported alongside.
9. Pass/fail recomputed against the R1 reference. This is a sensitivity check only; the headline counts keep the pre-registered reference.
10. Stage 3b cells, where they exist, are reported in a separate table. For each, n_eff growth from 10^4 to 10^5 samples is given; growth well below 10x indicates missing proposal mass.
11. R4 route agreement: n_eff, k-hat, d_L, and JS between routes.

## Deviations from earlier pre-registration (to be reported in the paper)

`results/revision/FACTS.md` (task F8) lists them.
