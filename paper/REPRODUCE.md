# Reproduce paper experiments

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

Requires official DINGO-BNS demo assets (event HDF5, ASDs, strain frames) and,
for detector-gated runs, `checkpoints/glitch_detector_v1/best_glitch_detector.pt`.

## 1) Ablation + oracle gap + runtime

```bash
python -u examples/method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

Smoke:

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

## 2) GW170817 stress grid

```bash
python -u examples/stress_gw170817.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1
```

## 3) Synthetic BNS stress

```bash
python -u examples/stress_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1
```

Match flags in `results/stress_test_synthetic_bns_v1/synth_config.json` for an exact rerun.

## 4) Official control (clean / poison / gated)

```bash
python -u examples/official_control.py \
  --outdir results/dingo_official_control
```

Large HDF5 sample dumps are not stored in git (JSON/PDF summaries are).

## Success criteria (as implemented in the examples)

- **Poison collapsed:** distance posterior rails (hi < 15 or lo > 90 Mpc-scale cuts used in code).
- **Recovers like clean:** median / CI overlap rules vs the clean reference in each script.
