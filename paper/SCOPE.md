# Paper scope

Standalone companion repository for a methods manuscript (in preparation for
*Astronomy and Computing*). This file locks the scientific claim.

## Aim

Show that **frozen official DINGO-BNS** parameter estimation collapses under
short-duration transient glitches, and that a **small preprocessing front-end**
restores clean-like posteriors **without retraining** the neural network.

Working title:
*Transient-glitch resilience in neural gravitational-wave parameter estimation
without network retraining*.

## Method

1. Inject or encounter a transient glitch in the analysis segment.
2. Detect candidate intervals (STFT time-bin detector; oracle gates for diagnostics).
3. Apply Tukey gates in the time domain.
4. Rebuild frequency-domain strain with **matched-delta** reconstruction
   (`adapt.glitch_excision`), not naive full-FFT replacement.
5. **Keep the original analysis ASDs** (do not replace with Welch estimated from
   the gated segment).
6. Sample with the **frozen official DINGO-BNS** model.

In ablation tables the full recipe is labeled `adapt_full`. That label means:
**gate + matched-delta + original ASD**.

## Canonical results

| Experiment | Path | Headline |
|---|---|---|
| Official control | `results/dingo_official_control/` | Clean vs poison vs gated on frozen DINGO |
| Honest excision | `results/excision_honest/` | Matched-delta + orig ASD recovers; Welch/replace fail |
| GW170817 stress | `results/stress_test_excision_v1/` | 240 cells; gated recovery ≈ **85.8%**; held-out ≈ **95.6%**; weak family `scattered_light` |
| Synthetic BNS stress | `results/stress_test_synthetic_bns_v1/` | 160 cells; gated recovery ≈ **91.9%**; detector fire 100% |
| Method hardening | `results/journal_method_hardening_v1/` | Ablation: `adapt_full` ≈ **81%** recover; `fft_replace` **0%**; oracle gap + runtime (~2× overhead vs poison PE) |

## Boundaries of this study

- Target model: **DINGO-BNS** only (frozen weights).
- Events / panels: **GW170817** packaging + locked **synthetic BNS** stress set.
- No network retraining; no BBH DINGO port in this manuscript.

## Repository map

```
src/adapt/                 # installable library
examples/                  # end-to-end paper studies
paper/                     # SCOPE + reproduce commands
results/                   # canonical artifacts
tests/
```
