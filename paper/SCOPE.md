# Scope of the study

This repository is the software companion to a methods manuscript in preparation
for *Astronomy and Computing*. The present file states the scientific claim and
experimental boundaries.

## Objective

To demonstrate that parameter estimation with frozen official DINGO-BNS is
susceptible to short-duration transient glitches, and that a preprocessing
front-end—time-domain gating, matched-delta frequency-domain reconstruction,
and retention of the analysis ASD—can restore posteriors consistent with the
clean reference without retraining the neural network.

Provisional title: *Transient-glitch resilience in neural gravitational-wave
parameter estimation without network retraining*.

## Procedure

1. Introduce or encounter a transient glitch in the analysis segment.
2. Identify candidate intervals with an STFT time-bin detector (oracle gates are
   retained for diagnostics).
3. Apply Tukey gates in the time domain.
4. Reconstruct frequency-domain strain with matched-delta updates
   (`adapt.glitch_excision`), as distinct from naive full-FFT replacement.
5. Retain the original analysis ASDs rather than Welch estimates formed on the
   gated segment.
6. Draw posterior samples with the frozen official DINGO-BNS model.

In ablation tables the complete recipe is denoted `adapt_full`, signifying
gating, matched-delta reconstruction, and original ASD retention.

## Canonical experiments

| Experiment | Path | Principal result |
|---|---|---|
| Official control | `results/dingo_official_control/` | Clean, poisoned, and gated inference on frozen DINGO-BNS |
| Honest excision | `results/excision_honest/` | Matched-delta with original ASD recovers; Welch or FFT replacement fails |
| GW170817 stress | `results/stress_test_excision_v1/` | 240 cells; gated recovery ≈ 85.8%; held-out ≈ 95.6%; weakest family `scattered_light` |
| Synthetic BNS stress | `results/stress_test_synthetic_bns_v1/` | 160 cells; gated recovery ≈ 91.9%; detector trigger rate 100% |
| Method hardening | `results/journal_method_hardening_v1/` | Ablation: `adapt_full` ≈ 81% recovery; `fft_replace` 0%; oracle-gap and runtime summaries |

## Boundaries

- Target estimator: DINGO-BNS with frozen weights.
- Event panels: GW170817 packaging and a locked synthetic BNS stress set.
- Network retraining of DINGO-BNS and extension to BBH DINGO are outside the
  present manuscript.

## Repository organisation

```
src/adapt/      installable library
examples/       end-to-end experimental drivers
paper/          scope, method, and reproduction notes
results/        archived summary artefacts
tests/          unit and integration tests
```
