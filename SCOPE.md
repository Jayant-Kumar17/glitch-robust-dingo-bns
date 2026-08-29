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
   (`src/adapt/glitch_excision.py`), not naive full-FFT replacement.
5. **Keep the original analysis ASDs** (do not replace with Welch estimated from
   the gated segment).
6. Sample with the **frozen official DINGO-BNS** model.

In ablation tables the full recipe is labeled `adapt_full`. That label means:
**gate + matched-delta + original ASD** (the complete front-end).

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
- Package import name is `adapt` (front-end utilities); it does not imply a larger system.

## Code map

```
src/adapt/glitch_excision.py
src/adapt/glitch_augmentation.py
src/adapt/stft_context.py
src/adapt/spectrogram_geometry.py
src/adapt/models/glitch_detector.py
scripts/event_glitch_io.py
scripts/compare_official_vs_gated.py
scripts/evaluate_glitch_excision.py
scripts/stress_test_glitch_excision.py
scripts/stress_test_synthetic_bns.py
scripts/journal_method_hardening.py
tests/test_glitch_excision.py
tests/test_honest_excision_dl.py
results/{dingo_official_control,excision_honest,
         stress_test_excision_v1,stress_test_synthetic_bns_v1,
         journal_method_hardening_v1}/
```
