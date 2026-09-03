# Reproduce journal method hardening

```bash
conda activate adapt_env
cd /path/to/glitch-robust-dingo-bns
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python -u examples/method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

Outputs: ablation_results.csv, ablation_summary.json,
oracle_gap_real.json, oracle_gap_synthetic.csv/json,
runtime_summary.json, method_hardening_report.pdf
