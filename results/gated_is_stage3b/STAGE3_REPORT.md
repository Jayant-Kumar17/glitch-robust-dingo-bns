# Stage 3 report: gated importance sampling on real glitches

Clean reference (DINGO IS): d_L median 34.6, 90% [19.5, 45.7] Mpc; efficiency 0.1134.
A cell counts as recovered by gated IS only if n_eff >= 100 AND the d_L criterion passes (docs/STAGE3_PLAN.md).

| group | gate pad | cells | Phase C localiser recovers | network only | naive IS | gated IS (trusted) | gated IS untrusted (n_eff<100) | no gate | median eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| damaged | ±0.1 s | 8 | 1/8 | 1/8 | 0/8 | 5/8 (62%; 31–86%) | 3 | 0 | 0.0017 |
| broken | ±0.1 s | 10 | 0/10 | 0/10 | 0/10 | 2/10 (20%; 6–51%) | 8 | 0 | 0.0005 |

Adaptive rule (pre-registered): per cell, take the pad with the larger gated-IS n_eff.


Damaged cells, ±0.1 s, by morphology / Omicron SNR bin (gated IS trusted recoveries; Phase C localiser in brackets):

- Koi_Fish: 4/6 [1]; Whistle: 1/2 [0]
- [100,300): 1/1 [0]; [30,100): 1/2 [0]; [300,inf): 3/5 [1]
- trusted cells: d_L JS vs clean median 0.025, max 0.047

Injection reproduction: 18 matched, 0 mismatched (skipped).
