# Reproduce synthetic BNS ADAPT stress

Frozen GW170817 DINGO + detect-and-gate on synthetic CBC injections.

## Environment

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
```

## Run

```bash
python -u examples/stress_synthetic_bns.py \
  --seed 0 --n-events 20 \
  --num-samples 512 --batch-size 256 \
  --device cpu \
  --outdir results/stress_test_synthetic_bns_v1
```

Resume skips completed `(event_id, cell_id)` pairs. Use `--overwrite` to restart.

## Outputs

- `synth_config.json`, `events.csv`, `results.csv`, `summary.json`,
  `failures.csv`, `stress_test_report.pdf`
