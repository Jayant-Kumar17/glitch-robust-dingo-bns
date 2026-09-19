# Experimental drivers

Executable studies for the frozen DINGO-BNS glitch front-end.

```bash
pip install -e .
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

| Script | Purpose | Canonical output |
|---|---|---|
| `official_control.py` | Clean, poisoned, and gated control | `results/dingo_official_control/` |
| `honest_excision.py` | Matched-delta versus Welch and FFT-replace diagnostics | `results/excision_honest/` |
| `stress_gw170817.py` | GW170817 stress grid (240 cells) | `results/stress_test_excision_v1/` |
| `stress_synthetic_bns.py` | Synthetic BNS stress panel | `results/stress_test_synthetic_bns_v1/` |
| `method_hardening.py` | Ablation, oracle gap, and runtime | `results/journal_method_hardening_v1/` |
| `train_glitch_detector.py` | Detector training (checkpoint also distributed) | `checkpoints/glitch_detector_v1/` |
| `clean_gate_control.py` | Cost of a false-positive gate on clean GW170817 data (25 forced placements) | `results/clean_gate_control_v1/` |
| `paper_figures_2_3.py` | Paper Figs 4–5 (filenames `fig2` injection / `fig3` posteriors) | `paper/figures/fig2_*`, `paper/figures/fig3_*` |
| `stress_real_glitches.py` | Real Gravity Spy O3 H1 glitches transplanted into GW170817 (capped-SNR ladder panel; `--wide-snr-panel --native-only` for the wide-SNR native panel) | `results/stress_real_glitches_v1/`, `results/stress_real_glitches_v2/` |
| `loudness_audit.py` | Whitened optimal SNR $\rho_w$ of every injected glitch (archived synthetic cells, control SG, real excerpts); population fraction above a threshold | `results/loudness_audit_v1/` |
| `collapse_threshold.py` | Collapse fraction and gated recovery vs $\rho_w$ (8 log-spaced values, 2 families, 5 seeds) | `results/collapse_threshold_v1/` |

Command-line flags for paper-matching runs are listed in
[`../paper/REPRODUCE.md`](../paper/REPRODUCE.md) and the root README.

DINGO-BNS weights remain frozen throughout; only the glitch detector is trained.
