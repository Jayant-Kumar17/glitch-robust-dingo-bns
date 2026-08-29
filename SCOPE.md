# Paper scope (locked)

This file is the single source of truth for what this repository claims.
Anything outside this scope was part of an earlier, abandoned ADAPT design and
must not appear in the public code, docs, or results.

## Aim

Show that **frozen official DINGO-BNS** parameter estimation collapses under
short-duration transient glitches, and that a **small preprocessing front-end**
restores clean-like posteriors **without retraining** the neural network.

Working manuscript title:
*Transient-glitch resilience in neural gravitational-wave parameter estimation
without network retraining* (*Astronomy and Computing*).

## Method (what we actually do)

1. Inject / encounter a transient glitch in the analysis segment.
2. Detect candidate intervals (STFT time-bin detector; oracle gates for diagnostics).
3. Apply Tukey gates in the time domain.
4. Rebuild frequency-domain strain with **matched-delta** reconstruction
   (`src/adapt/glitch_excision.py`), not naive full-FFT replacement.
5. **Keep the original analysis ASDs** (do not replace with Welch from the gated
   segment).
6. Sample with the **frozen official DINGO-BNS** model.

In ablation tables the full recipe is labeled `adapt_full` for historical
continuity of the CSV/JSON keys. It means only:
**detector/oracle gate + matched-delta + original ASD**.

## Relevant results (canonical)

| Experiment | Path | Headline |
|---|---|---|
| Official control | `results/dingo_official_control/` | Clean vs poison vs gated on frozen DINGO |
| Honest excision | `results/excision_honest/` | Matched-delta + orig ASD recovers; Welch/replace fail |
| GW170817 stress | `results/stress_test_excision_v1/` | 240 cells; gated recovery ≈ **85.8%**; held-out ≈ **95.6%**; weak family `scattered_light` |
| Synthetic BNS stress | `results/stress_test_synthetic_bns_v1/` | 160 cells; gated recovery ≈ **91.9%**; detector fire 100% |
| Method hardening | `results/journal_method_hardening_v1/` | Ablation: `adapt_full` ≈ **81%** recover; `fft_replace` **0%**; oracle gap + runtime (~2× overhead vs poison PE) |

## Explicitly out of scope (cut / do not revive)

From the preliminary ADAPT report and earlier prototypes — **not this paper**:

- Hub-and-spoke / decentralized continuous training
- BBH ↔ BNS dual-pathway routing
- Site-specific noise hubs / global network noise profiling
- Retraining DINGO embeddings / NSF / “glitch-robust” PE heads
- Detector retrain campaigns presented as the main fix
- Extra events beyond the locked GW170817 + synthetic BNS panels (unless added later as a separate study)

## Repo map (allowed code)

```
src/adapt/glitch_excision.py
src/adapt/glitch_augmentation.py
src/adapt/stft_context.py
src/adapt/spectrogram_geometry.py
src/adapt/models/glitch_detector.py
scripts/event_glitch_io.py              # TD/FD inject helpers only
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

The Python package name remains `adapt` for import stability; it no longer
denotes the hub-and-spoke framework.
