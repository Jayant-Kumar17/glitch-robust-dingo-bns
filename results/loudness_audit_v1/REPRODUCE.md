# Reproduce the loudness audit

```bash
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
python examples/loudness_audit.py --outdir results/loudness_audit_v1 [--threshold-rho <rho_w from collapse_threshold_v1>]
python examples/loudness_audit.py --repair-real --outdir results/loudness_audit_v1
python examples/loudness_audit.py --snr-tail-only --outdir results/loudness_audit_v1
```

Re-synthesises every archived cell of `results/stress_test_excision_v1/results.csv`
from `params_json` + `seed` (identical RNG consumption to `stress_gw170817.inject_spec_into_event`),
the official-control sine-Gaussian, and the cached real Gravity Spy excerpts at native loudness,
and reports the whitened optimal SNR `rho_w` of each injected glitch. No posterior sampling is run.

`--repair-real` keeps synthetic/control rows and re-extracts only the real Gravity Spy
excerpts (taper-aware excess + denoised fallback). Unrecoverable catalogue rows are
recorded in `summary.json` under `dropped_rows`.
