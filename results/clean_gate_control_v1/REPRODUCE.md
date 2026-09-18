# Reproduce the clean-gate cost control

Forced `adapt_full` gates (Tukey, half-width 0.4 s) on *clean* GW170817
data; N=20 placements on H1 with gate centre `t_rel` ~ U(-2.0, -0.2) s
relative to the trigger, plus 5 placements gating H1 and L1 at the same `t_rel`.
Posterior sampled with the frozen official DINGO-BNS (512 samples per placement).

```bash
conda activate adapt_env
cd /Users/jayantkumar/Desktop/ADAPT-Project/.worktrees/task2
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/clean_gate_control.py --seed 0 --num-samples 512 --outdir results/clean_gate_control_v1
# smoke: python examples/clean_gate_control.py --smoke --outdir results/clean_gate_control_smoke
```

## Outputs

- `results.csv` -- per placement: d_L / chirp_mass / mass_ratio / theta_jn at 5/50/95 %,
  `gated_recovers` (same criterion as the stress grids), JS divergence vs clean for the five
  core parameters, fraction of in-band packaged-strain power removed by the gate.
- `summary.json` -- recovery rate, max JS, mean |delta d_L median|, mean power removed.
- `paper/figures/figS2_clean_gate_cost.pdf` -- d_L median +/- 90 % CI vs gate centre.
