# Glitch mitigation for neural BNS parameter estimation with frozen DINGO-BNS

[![CITATION.cff](https://img.shields.io/badge/cite-CITATION.cff-blue)](CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the software accompanying a methods study on restoring
[DINGO-BNS](https://github.com/dingo-gw/dingo) posterior inference in the presence
of short-duration transient glitches, without retraining the neural posterior
estimator.

**Summary.** Short transient glitches overlapping the analysis segment collapse
the luminosity-distance posterior of the frozen official DINGO-BNS network in
46% of a 240-cell GW170817 injection grid. A preprocessing front-end (STFT
glitch detector, Tukey gates, matched-delta frequency-domain reconstruction,
retention of the original analysis ASD) restores clean-like posteriors in 86%
of those cells (96% for glitch families never seen by the detector) and in 92%
of a 160-cell synthetic BNS panel, at a few seconds of overhead and with the
DINGO-BNS weights untouched. Ablations show the reconstruction step is
load-bearing: full FFT replacement recovers 0% of cells, and Welch ASDs
recomputed on gated data recover 13%. A clean-gate control (spurious gate on
glitch-free data) recovers 92% with no collapse, and a panel of real Gravity Spy
O3 glitches transplanted into the GW170817 segment is provided as an
out-of-distribution check (`results/stress_real_glitches_v1/`).

When the analysis segment is contaminated by a transient, official DINGO-BNS
posteriors can collapse (most conspicuously in luminosity distance). The
procedure implemented here is a preprocessing front-end applied to the event
data package prior to sampling:

1. detection of candidate glitch intervals;
2. Tukey gating in the time domain;
3. frequency-domain reconstruction via a matched-delta update;
4. retention of the original analysis amplitude spectral densities (ASDs);
5. sampling with the frozen official DINGO-BNS model.

Scientific scope and summary metrics are given in [`paper/SCOPE.md`](paper/SCOPE.md).
The precise relationship to stock DINGO-BNS inference is specified in
[`paper/METHOD.md`](paper/METHOD.md).

## Relation to official DINGO-BNS

The official DINGO-BNS network weights and architecture are left unchanged.
Reproduction therefore consists of:

1. obtaining the official GW170817 DINGO-BNS demo model and event packaging from
   upstream sources;
2. applying the front-end defined in this repository to the event data package
   (detection, gating, matched-delta reconstruction, original ASD retention);
3. sampling with the same frozen DINGO-BNS posterior model.

The sole trained component contributed here is a compact STFT glitch detector
(`checkpoints/glitch_detector_v1/`). Algorithmic detail and the reconstruction
API are documented in [`paper/METHOD.md`](paper/METHOD.md).

## External dependencies

The repository distributes source code, the glitch-detector checkpoint, and
summary result artefacts (JSON/CSV/PDF). Official DINGO-BNS weights and LIGO
strain frames are not redistributed. Regeneration of numerical results requires:

1. The DINGO-BNS GW170817 demonstration tree (or an equivalent layout), e.g.
   ```text
   DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/
     downloads/                 # dingo-bns-model_*.pt, H/L/V *.gwf, PSD files
     inference-dingo-pipe/      # GW170817.ini and outdir/data/*_event_data.hdf5
   ```
   Installation follows the upstream DINGO-BNS demo documentation
   ([dingo-gw/dingo](https://github.com/dingo-gw/dingo)).

2. The glitch-detector checkpoint included in this repository:
   ```text
   checkpoints/glitch_detector_v1/best_glitch_detector.pt
   ```
   Optional retraining is provided by `examples/train_glitch_detector.py`.
   File hashes for the paper runs are recorded under `results/stress_test_*/`.

3. A Python environment consistent with `requirements.txt` (including
   `dingo-gw`, PyTorch, GWpy, Bilby, and related dependencies).

## Installation

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

If DINGO is installed from a local source tree, prepend that path:
`export PYTHONPATH=/path/to/dingo:src:examples`.

## Glitch-detector training (optional)

The paper checkpoint is already present. To retrain:

```bash
python -u examples/train_glitch_detector.py \
  --epochs 30 --batch-size 16 --steps-per-epoch 100 \
  --outdir checkpoints/glitch_detector_v1
```

This writes `best_glitch_detector.pt` and `train_summary.json`.

## Reproduction of paper experiments

The following commands regenerate the canonical artefact directories. On CPU with
`N=512` posterior samples, the full suite typically requires of order one to two
hours. Resume behaviour may be overridden with `--overwrite`.

### 1. Official control (clean, poisoned, and gated)

```bash
python -u examples/official_control.py \
  --outdir results/dingo_official_control
```

### 2. Honest excision diagnostics

```bash
python -u examples/honest_excision.py \
  --outdir results/excision_honest
```

### 3. GW170817 stress grid (240 cells)

```bash
python -u examples/stress_gw170817.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1
```

### 4. Synthetic BNS stress panel (160 cells)

```bash
python -u examples/stress_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1
```

Additional flags for an exact match to the archived run are recorded in
`results/stress_test_synthetic_bns_v1/synth_config.json`.

### 5. Method hardening (ablation, oracle gap, and runtime)

```bash
python -u examples/method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

### 6. Clean-gate cost control (25 forced placements)

```bash
python -u examples/clean_gate_control.py \
  --seed 0 --num-samples 512 --device cpu \
  --outdir results/clean_gate_control_v1
```

### 7. Real Gravity Spy glitch panel

```bash
python -u examples/stress_real_glitches.py \
  --seed 0 --num-samples 512 --device cpu \
  --outdir results/stress_real_glitches_v1
```

Downloads the Gravity Spy H1 O3 tables (Zenodo 5649212, ~190 MB) into
`data/gravity_spy/raw/` and fetches 8 s H1 excerpts from GWOSC on first run.

### Restricted smoke runs (development / CI only)

The newer drivers (`clean_gate_control.py`, `stress_real_glitches.py`,
`paper_figures_2_3.py`) accept `--smoke` (2 cells, 64 samples); the original
drivers expose `--max-cells`, e.g.

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
python -u examples/clean_gate_control.py --smoke --outdir results/clean_gate_control_smoke
```

Equivalent instructions appear in [`paper/REPRODUCE.md`](paper/REPRODUCE.md) and
[`examples/README.md`](examples/README.md).

## Results at a glance

| Experiment | Directory | Cells | Principal finding |
|---|---|---|---|
| Official control | [`results/dingo_official_control/`](results/dingo_official_control/) | 1 | clean vs poisoned vs gated; max 1-D JS (clean vs gated) 0.012 nat, verdict A |
| Honest excision | [`results/excision_honest/`](results/excision_honest/) | 1 | matched-delta + original ASD recovers; Welch ASD or FFT replacement collapses |
| GW170817 stress grid | [`results/stress_test_excision_v1/`](results/stress_test_excision_v1/) | 240 | poisoned collapse 46.3%; gated recovery **85.8%** (held-out families 95.6%); clean false-positive rate 0/50 |
| Synthetic BNS panel | [`results/stress_test_synthetic_bns_v1/`](results/stress_test_synthetic_bns_v1/) | 160 | gated recovery **91.9%**; detector fire rate 100% |
| Method hardening (ablation) | [`results/journal_method_hardening_v1/`](results/journal_method_hardening_v1/) | 16 × 5 arms | `adapt_full` 81%; `gate_welch` 13%; `fft_replace` **0%**; front-end overhead ≈ 3 s |
| Clean-gate cost control | [`results/clean_gate_control_v1/`](results/clean_gate_control_v1/) | 25 | forced gate on clean data: recovery 92%, no collapse, mean \|Δ d_L\| 3.8 Mpc |
| Real Gravity Spy glitches (capped SNR) | [`results/stress_real_glitches_v1/`](results/stress_real_glitches_v1/) | 24 glitches × (native + 3 severities) + noise-only controls | native loudness: 0/20 collapse, 95 % pass-through; synthetic severity ladder: 86 % collapse, 72 % gated recovery |
| Real Gravity Spy glitches (wide SNR, native) | [`results/stress_real_glitches_v2/`](results/stress_real_glitches_v2/) | 72 glitches × 3 t_rel + 10 noise-only | collapse and recovery vs Omicron-SNR bin; see `summary.json` |
| Loudness audit | [`results/loudness_audit_v1/`](results/loudness_audit_v1/) | 240 + 1 + 24 | whitened optimal SNR $\rho_w$ of every injected glitch; synthetic severities 3/6/10 span $\rho_w \approx 150$–$6\times10^4$ (medians 4.2k / 7.3k / 10.4k); real glitches at native loudness $\rho_w \le 140$ |
| Collapse threshold | [`results/collapse_threshold_v1/`](results/collapse_threshold_v1/) | 2 families × 8 $\rho_w$ × 5 seeds | collapse first exceeds 50 % at $\rho_w \approx 2.7\times10^3$ (grid point 3.7k); gated recovery 95 % overall |

Large HDF5 posterior sample files are omitted from version control; JSON, CSV,
and PDF summaries are retained.

## Paper figures

All figures in `paper/figures/` regenerate from the archived `results/`:

```bash
python paper/make_figures.py --repo .            # Figs 1, 4, 5, 6, 7, S1
python -u examples/paper_figures_2_3.py          # Figs 2, 3 (needs DINGO-BNS demo assets + sample HDF5s)
python -u examples/clean_gate_control.py         # Fig S2 (re-runs the control)
python -u examples/stress_real_glitches.py --wide-snr-panel --native-only \
  --selected-csv data/gravity_spy/selected_h1_o3_wide.csv --outdir results/stress_real_glitches_v2   # Fig 8
python -u examples/loudness_audit.py             # Fig S3 (no PE)
python -u examples/collapse_threshold.py         # Fig 9
```

## Library interface

```python
from adapt.glitch_excision import rebuild_event_from_gated_td
from adapt.models import GlitchDetectorSTFT
from adapt.event_glitch_io import inject_h1_glitch_into_event
from adapt.dingo_bns_demo import discover_assets, run_baseline_sampling
```

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).

Provisional title: *Transient-glitch resilience in neural gravitational-wave
parameter estimation without network retraining*.

Software citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders
a "Cite this repository" button from it). Tagged releases are listed under
[Releases](https://github.com/Jayant-Kumar17/glitch-robust-dingo-bns/releases);
steps for minting a Zenodo DOI are in [`docs/ZENODO.md`](docs/ZENODO.md).

Users of this software are requested to cite the published article when
available and to acknowledge [DINGO](https://github.com/dingo-gw/dingo) as the
underlying neural parameter-estimation framework. Real-glitch experiments use
the Gravity Spy classifications of Glanzer et al. (2023), Zenodo record
[5649212](https://doi.org/10.5281/zenodo.5649212).

## Licence

MIT Licence (`LICENSE`). Upstream DINGO software and LIGO/Virgo data products
remain subject to their respective terms of use.

## Contact

Jayant Kumar, Karachi Grammar School

j.kumar16224@kgs.edu.pk
