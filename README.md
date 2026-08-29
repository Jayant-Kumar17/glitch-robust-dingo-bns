# Glitch-robust DINGO-BNS inference (no retraining)

Companion code for the methods study defined in **[SCOPE.md](SCOPE.md)**.

**Aim:** restore frozen [DINGO-BNS](https://github.com/dingo-gw/dingo) posteriors under transient glitches **without retraining**.

**Recipe:** detect → Tukey gate → **matched-delta** FD rebuild → **keep analysis ASD** → sample frozen official DINGO-BNS.

This is **not** the earlier hub-and-spoke / dual-pathway ADAPT framework (router, BBH pathway, continuous training hub, site noise hubs). Those pieces are out of scope and removed.

## Headline results

| Study | Result |
|---|---|
| GW170817 stress (240 cells) | gated recovery ≈ **85.8%** |
| Synthetic BNS (160 cells) | gated recovery ≈ **91.9%** |
| Ablation (16 cells) | full recipe ≈ **81%** recover; FFT-replace **0%** |
| Runtime | front-end overhead ≈ **2×** vs poison-only PE |

Details and artifacts: `results/` (see also [REPRODUCE.md](REPRODUCE.md)).

## Layout

```
SCOPE.md                         # locked paper claim
src/adapt/
  glitch_excision.py             # matched-delta / replace rebuild
  glitch_augmentation.py         # synthetic glitch families
  stft_context.py                # STFT helpers for the detector
  spectrogram_geometry.py        # STFT grid constants
  models/glitch_detector.py      # time-bin detector (NSF frozen)
scripts/
  event_glitch_io.py             # TD/FD inject helpers only
  compare_official_vs_gated.py
  evaluate_glitch_excision.py
  stress_test_glitch_excision.py
  stress_test_synthetic_bns.py
  journal_method_hardening.py
results/                         # five canonical paper dirs only
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

Official DINGO-BNS weights / GW170817 demo packaging are upstream (not redistributed). Detector-gated runs expect `checkpoints/glitch_detector_v1/best_glitch_detector.pt`.

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).  
Working title: *Transient-glitch resilience in neural gravitational-wave parameter estimation without network retraining*.

## License

MIT (`LICENSE`). Upstream DINGO and LIGO/Virgo data remain under their own terms.
