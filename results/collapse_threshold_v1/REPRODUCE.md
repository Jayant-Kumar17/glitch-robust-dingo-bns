# Reproduce the collapse-threshold sweep

```bash
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/collapse_threshold.py --seed 0 --num-samples 512 --n-seeds 5 --outdir results/collapse_threshold_v1
```

Families: sine_gaussian, broadband_burst; rho_w grid: 10, 27, 72, 193, 518, 1389, 3728, 10000;
stationary ASD; arms poisoned + adapt_full (detector threshold 0.728).
Each seed fixes t_rel and the waveform realisation; only the amplitude is rescaled to hit rho_w.
