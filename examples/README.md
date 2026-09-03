# Examples

End-to-end studies for the frozen DINGO-BNS glitch front-end.

```bash
pip install -e .
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

| Script | What it does | Paper output |
|---|---|---|
| `official_control.py` | Clean / poison / gated control | `results/dingo_official_control/` |
| `honest_excision.py` | Matched-delta vs Welch / replace | `results/excision_honest/` |
| `stress_gw170817.py` | GW170817 240-cell stress grid | `results/stress_test_excision_v1/` |
| `stress_synthetic_bns.py` | Synthetic BNS stress panel | `results/stress_test_synthetic_bns_v1/` |
| `method_hardening.py` | Ablation + oracle + runtime | `results/journal_method_hardening_v1/` |

Full flags: [`../paper/REPRODUCE.md`](../paper/REPRODUCE.md) and the root README.
