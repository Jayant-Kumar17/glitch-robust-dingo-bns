# Reproduce paper experiments

Same commands as the root [`README.md`](../README.md) (kept here for the paper
folder). Prerequisites: DINGO-BNS GW170817 demo assets + glitch-detector
checkpoint (see README).

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Full paper suite

```bash
python -u examples/official_control.py \
  --outdir results/dingo_official_control

python -u examples/honest_excision.py \
  --outdir results/excision_honest

python -u examples/stress_gw170817.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1

python -u examples/stress_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1

python -u examples/method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

## Smoke (not paper numbers)

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

## Success criteria (as implemented)

- **Poison collapsed:** `d_L` hi < 15 or lo > 90 (Mpc-scale rails in code).
- **Recovers like clean:** median / CI overlap rules vs the clean reference.
