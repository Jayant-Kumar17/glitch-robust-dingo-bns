# Reproduce paper experiments

All commands assume a working DINGO-BNS install, GW170817 demo event packaging, and:

```bash
conda activate adapt_env
cd /path/to/this/repo
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE
# If needed for local demo data:
# export PYTHONPATH=/path/to/dingo/src:src:scripts
```

Device defaults below match the paper runs (`cpu`, N=512). Use `--overwrite` to ignore resume caches.

## 1) Method hardening (ablation + oracle + runtime)

```bash
python -u scripts/journal_method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

Smoke:

```bash
python -u scripts/journal_method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

Artifacts: `ablation_*.csv/json`, `oracle_gap_*.json/csv`, `runtime_summary.json`, `method_hardening_report.pdf`.

## 2) GW170817 glitch stress grid

```bash
python -u scripts/stress_test_glitch_excision.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --hf-samples 2000 \
  --batch-size 256 --device cpu \
  --outdir results/stress_test_excision_v1
```

## 3) Synthetic BNS stress

```bash
python -u scripts/stress_test_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1
```

(Use the flags recorded in `results/stress_test_synthetic_bns_v1/synth_config.json` / `REPRODUCE.md` for an exact match.)

## 4) Official control (clean / poison / gated)

```bash
python -u scripts/compare_official_vs_gated.py \
  --outdir results/dingo_official_control
```

Large HDF5 sample files are **not** stored in git. JSON summaries and the corner PDF are.

## Success criteria (stress / ablation)

- **Poison collapsed:** `d_L` hi < 15 or lo > 90 (Mpc-scale rails used in scripts).
- **Recovers like clean:** median in the clean-compatible window and CI overlap rules implemented in the stress scripts.

## What is not redistributed

- Official DINGO-BNS network weights (obtain from upstream / Zenodo).
- Local `DINGO-BNS/` source+env trees and multi-GB checkpoints.
- Raw GWOSC `.gwf` caches (downloaded on demand).
