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

Command-line flags for paper-matching runs are listed in
[`../paper/REPRODUCE.md`](../paper/REPRODUCE.md) and the root README.

DINGO-BNS weights remain frozen throughout; only the glitch detector is trained.
