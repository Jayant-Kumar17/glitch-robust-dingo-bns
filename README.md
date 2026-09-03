# Glitch-robust DINGO-BNS inference

Companion code for restoring [DINGO-BNS](https://github.com/dingo-gw/dingo)
posteriors under short-duration transient glitches **without retraining** the
neural network.

Frozen official DINGO-BNS can collapse when a glitch contaminates the analysis
segment. This repository implements a small front-end:

1. detect candidate glitch intervals,
2. apply Tukey gates in the time domain,
3. rebuild frequency-domain strain with **matched-delta** reconstruction,
4. **keep the original analysis ASD**,
5. sample with the frozen DINGO-BNS model.

Claim / method / headline numbers: [`paper/SCOPE.md`](paper/SCOPE.md).  
**Exact “what we changed vs stock DINGO” recipe:** [`paper/METHOD.md`](paper/METHOD.md).

## How this uses official DINGO (read this)

We **do not modify** the official DINGO-BNS network weights or architecture.

Replication means:

1. Fetch the **same official** GW170817 DINGO-BNS demo model as everyone else.
2. Apply **our front-end** to the event package before sampling:
   detect → Tukey gate → **matched-delta** FD rebuild → **keep original ASD**.
3. Call the **same frozen** DINGO sampler on that cleaned package.

The only trained artifact of ours is the small STFT glitch detector in
`checkpoints/glitch_detector_v1/`. Details and the critical API call are in
[`paper/METHOD.md`](paper/METHOD.md).

## Prerequisites (required to re-run)

This repo ships **code + paper result tables/PDFs**. It does **not** redistribute
DINGO weights or LIGO strain frames. To regenerate results you need:

1. **DINGO-BNS GW170817 demo** under the repo (or symlink) at:
   ```text
   DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/
     downloads/                 # dingo-bns-model_*.pt, H/L/V *.gwf, PSD txt
     inference-dingo-pipe/      # GW170817.ini + outdir/data/*_event_data.hdf5
   ```
   Follow the upstream DINGO-BNS demo / Zenodo instructions from
   [dingo-gw/dingo](https://github.com/dingo-gw/dingo).

2. **Glitch detector** — weights **are in this repo**:
   ```text
   checkpoints/glitch_detector_v1/best_glitch_detector.pt
   ```
   Training recipe: `examples/train_glitch_detector.py` (see below).
   SHA-256 of the paper run is also recorded in `results/stress_test_*/` configs.

3. A scientific Python env with the pins in `requirements.txt` (includes
   `dingo-gw`, `torch`, `gwpy`, `bilby`, …).

**Important:** the published method does **not** fine-tune DINGO-BNS. The only
trained component here is the small STFT **glitch detector**. Official DINGO
weights stay frozen and come from upstream. See [`paper/METHOD.md`](paper/METHOD.md).

## Installation

```bash
conda activate adapt_env   # or your env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

Optional: if your DINGO install is a local checkout, prepend it:
`export PYTHONPATH=/path/to/dingo:src:examples`.

## Train the glitch detector (optional — weights already shipped)

```bash
python -u examples/train_glitch_detector.py \
  --epochs 30 --batch-size 16 --steps-per-epoch 100 \
  --outdir checkpoints/glitch_detector_v1
```

Writes `best_glitch_detector.pt` + `train_summary.json`. You can skip this if
you use the committed checkpoint.

## Reproduce all paper experiments

Commands below match the published artifact directories. Wall time is roughly
**~1–2 hours on CPU** for the full suite (N=512). Use `--overwrite` to ignore
resume caches.

### 1) Official control (clean / poison / gated)
```bash
python -u examples/official_control.py \
  --outdir results/dingo_official_control
```

### 2) Honest excision diagnostics

```bash
python -u examples/honest_excision.py \
  --outdir results/excision_honest
```

### 3) GW170817 stress grid (240 cells)

```bash
python -u examples/stress_gw170817.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1
```

### 4) Synthetic BNS stress (160 cells)

```bash
python -u examples/stress_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1
```

Match any extra flags from `results/stress_test_synthetic_bns_v1/synth_config.json`
for an exact rerun.

### 5) Method hardening (ablation + oracle + runtime)

```bash
python -u examples/method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

### Optional smoke (debug only — not paper numbers)

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

Same commands are also listed in [`paper/REPRODUCE.md`](paper/REPRODUCE.md) and
[`examples/README.md`](examples/README.md).

## What is already in the repo (no re-run needed to read)

| Study | Directory | Headline |
|---|---|---|
| Official control | `results/dingo_official_control/` | clean / poison / gated summaries |
| Honest excision | `results/excision_honest/` | matched-delta + orig ASD recovers |
| GW170817 stress | `results/stress_test_excision_v1/` | gated recovery ≈ **85.8%** |
| Synthetic BNS | `results/stress_test_synthetic_bns_v1/` | gated recovery ≈ **91.9%** |
| Method hardening | `results/journal_method_hardening_v1/` | full recipe ≈ **81%**; FFT-replace **0%** |

Large HDF5 PE sample dumps are gitignored; JSON/CSV/PDF summaries are tracked.

## Library API

```python
from adapt.glitch_excision import rebuild_event_from_gated_td
from adapt.models import GlitchDetectorSTFT
from adapt.event_glitch_io import inject_h1_glitch_into_event
from adapt.dingo_bns_demo import discover_assets, run_baseline_sampling
```

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).

Working title: *Transient-glitch resilience in neural gravitational-wave
parameter estimation without network retraining*.

Please cite the paper when available and acknowledge
[DINGO](https://github.com/dingo-gw/dingo) for the underlying neural PE model.

## License

MIT — see [`LICENSE`](LICENSE). Upstream DINGO and LIGO/Virgo data products
remain under their own terms.

## Contact

Jayant Kumar — Karachi Grammar School
