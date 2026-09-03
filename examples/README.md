# Examples

End-to-end studies for the frozen DINGO-BNS glitch front-end.

All commands assume:

```bash
pip install -e .
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

| Script | What it does | Default output |
|---|---|---|
| `official_control.py` | Clean / poison / gated control on official DINGO-BNS | `results/dingo_official_control/` |
| `honest_excision.py` | Matched-delta vs Welch / replace diagnostics | `results/excision_honest/` |
| `stress_gw170817.py` | Locked GW170817 glitch stress grid | `results/stress_test_excision_v1/` |
| `stress_synthetic_bns.py` | Synthetic BNS stress panel | `results/stress_test_synthetic_bns_v1/` |
| `method_hardening.py` | Ablation + oracle gap + runtime | `results/journal_method_hardening_v1/` |

Shared TD/FD inject helpers live in the installable package:
`adapt.event_glitch_io`.

Exact flags for paper-matching runs: [`../paper/REPRODUCE.md`](../paper/REPRODUCE.md).
