# Results

Archived summary artefacts corresponding to [`paper/SCOPE.md`](../paper/SCOPE.md).

| Directory | Contents |
|---|---|
| `dingo_official_control/` | Clean, poisoned, and gated control summaries and corner figure |
| `excision_honest/` | Honest matched-delta excision report and comparison figure |
| `stress_test_excision_v1/` | GW170817 stress grid (240 cells) |
| `stress_test_synthetic_bns_v1/` | Synthetic BNS stress panel and figures |
| `journal_method_hardening_v1/` | Ablation, oracle-gap, and runtime summaries |
| `clean_gate_control_v1/` | Clean-gate cost control: forced `adapt_full` gates on glitch-free data (Fig. S2) |
| `stress_real_glitches_v1/` | Real Gravity Spy glitch panel, capped SNR 8-30, native + severity ladder |
| `stress_real_glitches_v2/` | Real Gravity Spy glitch panel, wide SNR bins, native loudness only (Fig. 8) |
| `loudness_audit_v1/` | $\rho_w$ of every injected glitch on one scale (Fig. S3); population fraction above the collapse threshold |
| `collapse_threshold_v1/` | Collapse-threshold sweep in $\rho_w$ (Fig. 9) |

Reproduction commands are given in [`paper/REPRODUCE.md`](../paper/REPRODUCE.md).
Large regenerable products (HDF5 sample dumps, smoke runs, and logs) are
excluded from version control.
