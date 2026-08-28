# Reproduce journal method hardening

```bash
conda activate adapt_env
cd /Users/jayantkumar/Desktop/ADAPT-Project
export PYTHONPATH=DINGO-BNS/dingo:src:scripts KMP_DUPLICATE_LIB_OK=TRUE
python -u scripts/journal_method_hardening.py \
  --num-samples 512 --batch-size 256 --device cpu \
  --outdir results/journal_method_hardening_v1
```

Outputs: ablation_results.csv, ablation_summary.json,
oracle_gap_real.json, oracle_gap_synthetic.csv/json,
runtime_summary.json, method_hardening_report.pdf
