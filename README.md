# Glitch-robust DINGO-BNS inference (no retraining)

Public companion code for a methods study on restoring [DINGO-BNS](https://github.com/dingo-gw/dingo) posteriors under transient glitches **without retraining** the neural network.

See **[SCOPE.md](SCOPE.md)** for the locked claim, method, and canonical results.

**Recipe:** detect → Tukey gate → **matched-delta** frequency-domain rebuild → **keep the original analysis ASD** → sample with frozen official DINGO-BNS.

## Headline results

| Study | Result |
|---|---|
| GW170817 stress (240 cells) | gated recovery ≈ **85.8%** |
| Synthetic BNS (160 cells) | gated recovery ≈ **91.9%** |
| Ablation (16 cells) | full recipe ≈ **81%** recover; FFT-replace **0%** |
| Runtime | front-end overhead ≈ **2×** vs poison-only PE |

Artifacts live under `results/`. Exact commands: [REPRODUCE.md](REPRODUCE.md).

## Layout

```
SCOPE.md
src/adapt/
  glitch_excision.py             # matched-delta / replace rebuild
  glitch_augmentation.py         # synthetic glitch families
  stft_context.py                # STFT helpers for the detector
  spectrogram_geometry.py        # STFT grid constants
  models/glitch_detector.py      # time-bin detector (NSF stays frozen)
scripts/
  event_glitch_io.py             # TD/FD inject helpers
  compare_official_vs_gated.py
  evaluate_glitch_excision.py
  stress_test_glitch_excision.py
  stress_test_synthetic_bns.py
  journal_method_hardening.py
results/
tests/
```

## Setup

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE
```

Official DINGO-BNS weights and the GW170817 demo packaging are obtained upstream (not redistributed here). Detector-gated runs expect `checkpoints/glitch_detector_v1/best_glitch_detector.pt`.

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).  
Working title: *Transient-glitch resilience in neural gravitational-wave parameter estimation without network retraining*.

## License

MIT (`LICENSE`). Upstream DINGO and LIGO/Virgo data products remain under their own terms.
