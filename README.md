# Glitch-robust neural BNS PE without retraining

Research code accompanying a methods study on **restoring [DINGO-BNS](https://github.com/dingo-gw/dingo) posteriors under transient glitches without retraining the neural network**.

**Core idea.** Short-duration glitches can collapse luminosity-distance inference in frozen DINGO-BNS. Time-domain Tukey gating alone is not enough for neural PE: a **matched-delta frequency-domain rebuild** that **preserves the original analysis ASD** recovers clean-like posteriors. Full FFT replacement is a known-bad control.

> Scope of this repository / paper: **DINGO-BNS + glitch detect → gate → matched-delta rebuild → original ASD**.  
> Not in scope: BBH DINGO ports, automatic BNS/BBH routers, or globally distributed training.

Affiliation for the author: **Karachi Grammar School** (listed on the manuscript; this GitHub account is the long-lived personal research account).

## Method (ADAPT front-end)

1. Detect candidate glitch intervals (STFT-based detector; oracle gates used for diagnostics).
2. Apply Tukey gates in the time domain.
3. Rebuild the frequency-domain strain with **`matched_delta`** (`src/adapt/glitch_excision.py`).
4. Keep the **original analysis ASDs** (do not replace with Welch from the gated segment).
5. Sample with the **frozen official DINGO-BNS** posterior model (no NSF / embedding retrain).

## Canonical results (paper)

| Experiment | Directory |
|---|---|
| Official clean / poison / gated control | `results/dingo_official_control/` (summaries + PDF; large HDF5 samples not in git) |
| Honest excision diagnostics | `results/excision_honest/` |
| GW170817 stress grid (240 cells) | `results/stress_test_excision_v1/` |
| Synthetic BNS stress | `results/stress_test_synthetic_bns_v1/` |
| Ablation + oracle gap + runtime | `results/journal_method_hardening_v1/` |

See **[REPRODUCE.md](REPRODUCE.md)** for exact commands.

## Repository layout

```
src/adapt/
  glitch_excision.py      # matched-delta / replace rebuild + ASD handling
  glitch_augmentation.py  # synthetic glitch families
  …                       # supporting utilities
scripts/
  stress_test_glitch_excision.py
  stress_test_synthetic_bns.py
  journal_method_hardening.py
  compare_official_vs_gated.py
  evaluate_glitch_excision.py
results/                  # paper tables, JSONs, report PDFs
tests/                    # unit / regression tests for excision
```

## Setup

### 1. Environment

Use a conda env with scientific GW + ML stacks (example name used in this work: `adapt_env`):

```bash
conda activate adapt_env
cd /path/to/this/repo
pip install -e .
pip install -r requirements.txt
```

`requirements.txt` pins a `dingo-gw` install from GitHub. You also need the **official DINGO-BNS GW170817 demo / Zenodo weights** and event packaging used by the scripts (not redistributed here — follow the upstream DINGO-BNS demo instructions).

### 2. Path / runtime flags

```bash
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE
```

If you keep a local DINGO source checkout for the BNS demo data products, prepend that path as well (historically `DINGO-BNS/dingo` on the development machine).

### 3. Smoke tests

```bash
python -u scripts/journal_method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).  
Working title: *Transient-glitch resilience in neural gravitational-wave parameter estimation without network retraining*.

## License

Code in this repository is released under the MIT License (see `LICENSE`).  
Upstream DINGO / LIGO data products remain under their respective licenses and terms of use.
