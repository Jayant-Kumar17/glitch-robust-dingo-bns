# Reproduce detect-and-gate stress test

Event: **GW170817** only (in-repo DINGO-BNS Zenodo model).
Glitches: synthetic families from `adapt.glitch_augmentation` (held-in + held-out).
PE: frozen official DINGO; gated path = detect-and-gate + original ASD.

## Environment

```bash
conda activate adapt_env
cd /Users/jayantkumar/Desktop/ADAPT-Project
export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
```

## Run

```bash
python scripts/stress_test_glitch_excision.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1
```

Resume is automatic if `results.csv` exists (skip completed `cell_id`s).
Use `--overwrite` to restart.

## Outputs

- `stress_config.json` — locked recipe + ckpt hashes
- `results.csv` — per-cell metrics
- `summary.json` — aggregates
- `failures.csv` — gated non-recoveries
- `tier_b_results.json` / `tier_c_clean_fp.json`
- `stress_test_report.pdf`
- Gold 20k+IS control (canonical SG): `results/dingo_official_control/`

## Success criteria

- Poison collapsed: `d_L` hi < 15 or lo > 90
- Gated recovers: med in [20, 50], CI overlaps clean, |med − clean_med| ≤ 10 Mpc
