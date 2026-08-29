# Glitch-robust DINGO-BNS inference (no retraining)

Public companion code for a methods study on **why transient glitches collapse frozen [DINGO-BNS](https://github.com/dingo-gw/dingo) posteriors**, and **how a small front-end restores clean-like inference without retraining the neural network**.

This repository is **not** the earlier hub-and-spoke / multi-class ADAPT framework. Scope is intentionally narrow:

> Detect → Tukey-gate → **matched-delta** frequency-domain rebuild → **keep the original analysis ASD** → sample with **frozen official DINGO-BNS**.

## What failed (and what fixed it)

| Condition | Typical outcome |
|---|---|
| Glitchy strain + Welch ASD | Posterior collapses (e.g. luminosity distance rails) |
| Gate + full FFT replace + original ASD | Still fails (known-bad control) |
| Gate + matched-delta rebuild + original ASD | Recovers clean-compatible posteriors |

Ablation, stress grids, oracle diagnostics, and runtime are under `results/`.

## Layout

```
src/adapt/
  glitch_excision.py         # Tukey gates + matched-delta / replace rebuild
  glitch_augmentation.py     # synthetic glitch families
  stft_context.py            # STFT / whitening helpers for the detector
  spectrogram_geometry.py    # shared STFT grid constants
  models/glitch_detector.py  # STFT time-bin detector (NSF stays frozen)
scripts/
  compare_official_vs_gated.py
  evaluate_glitch_excision.py
  evaluate_glitch_robustness.py   # shared inject / TD helpers
  stress_test_glitch_excision.py  # GW170817 stress grid
  stress_test_synthetic_bns.py
  journal_method_hardening.py     # ablation + oracle + runtime
results/                     # paper tables / JSON / report PDFs
tests/                       # excision unit tests
```

## Setup

```bash
conda activate adapt_env   # or any env with the pins in requirements.txt
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE
```

Official **DINGO-BNS weights** and the GW170817 demo event packaging are **not** redistributed here — follow the upstream DINGO-BNS demo / Zenodo instructions. A local glitch-detector checkpoint is expected at `checkpoints/glitch_detector_v1/best_glitch_detector.pt` when running detector-gated experiments.

## Reproduce paper runs

See **[REPRODUCE.md](REPRODUCE.md)**. Canonical artifact directories:

| Experiment | Path |
|---|---|
| Official clean / poison / gated control | `results/dingo_official_control/` |
| Honest excision diagnostics | `results/excision_honest/` |
| GW170817 stress grid | `results/stress_test_excision_v1/` |
| Synthetic BNS stress | `results/stress_test_synthetic_bns_v1/` |
| Ablation + oracle + runtime | `results/journal_method_hardening_v1/` |

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).  
Working title: *Transient-glitch resilience in neural gravitational-wave parameter estimation without network retraining*.

## License

MIT (see `LICENSE`). Upstream DINGO and LIGO/Virgo data products remain under their own terms.
