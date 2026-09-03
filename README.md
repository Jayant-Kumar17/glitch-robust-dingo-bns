# Glitch mitigation for neural BNS parameter estimation with frozen DINGO-BNS

This repository provides the software accompanying a methods study on restoring
[DINGO-BNS](https://github.com/dingo-gw/dingo) posterior inference in the presence
of short-duration transient glitches, without retraining the neural posterior
estimator.

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

### Restricted smoke run (development only)

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

Equivalent instructions appear in [`paper/REPRODUCE.md`](paper/REPRODUCE.md) and
[`examples/README.md`](examples/README.md).

## Archived results

| Experiment | Directory | Principal finding |
|---|---|---|
| Official control | `results/dingo_official_control/` | clean / poisoned / gated summaries |
| Honest excision | `results/excision_honest/` | matched-delta with original ASD recovers inference |
| GW170817 stress | `results/stress_test_excision_v1/` | gated recovery ≈ 85.8% |
| Synthetic BNS | `results/stress_test_synthetic_bns_v1/` | gated recovery ≈ 91.9% |
| Method hardening | `results/journal_method_hardening_v1/` | full recipe ≈ 81%; FFT replacement 0% |

Large HDF5 posterior sample files are omitted from version control; JSON, CSV,
and PDF summaries are retained.

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

Users of this software are requested to cite the published article when
available and to acknowledge [DINGO](https://github.com/dingo-gw/dingo) as the
underlying neural parameter-estimation framework.

## Licence

MIT Licence (`LICENSE`). Upstream DINGO software and LIGO/Virgo data products
remain subject to their respective terms of use.

## Contact

Jayant Kumar, Karachi Grammar School
