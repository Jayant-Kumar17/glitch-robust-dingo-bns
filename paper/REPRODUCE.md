# Reproduction commands

Commands identical to those in the root [`README.md`](../README.md) are collected
here for convenience. Required external assets are the DINGO-BNS GW170817
demonstration packaging and the glitch-detector checkpoint described in the
README.

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Full experimental suite

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

## Restricted smoke configuration

Intended for software verification only; reported paper metrics use the full
suite above.

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

## Success criteria (as coded)

- Poisoned collapse: luminosity-distance credible interval with upper edge
  below 15 Mpc or lower edge above 90 Mpc.
- Recovery relative to clean: median and interval-overlap criteria implemented
  in the respective example drivers.
