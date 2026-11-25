# Stage 3 report: gated importance sampling on real glitches

Clean reference (DINGO IS): d_L median 34.2, 90% [18.9, 45.9] Mpc; efficiency 0.1000.
A cell counts as recovered by gated IS only if n_eff >= 100 AND the d_L criterion passes (docs/STAGE3_PLAN.md).

| group | gate pad | cells | Phase C localiser recovers | network only | naive IS | gated IS (trusted) | gated IS untrusted (n_eff<100) | no gate | median eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| damaged | ±0.1 s | 41 | 31/41 | 31/41 | 29/41 | 32/41 (78%; 63–88%) | 8 | 0 | 0.0824 |
| damaged | ±0.2 s | 41 | 31/41 | 25/41 | 21/41 | 30/41 (73%; 58–84%) | 10 | 0 | 0.0455 |
| broken | ±0.1 s | 14 | 0/14 | 0/14 | 0/14 | 4/14 (29%; 12–55%) | 10 | 0 | 0.0030 |
| broken | ±0.2 s | 14 | 0/14 | 0/14 | 0/14 | 0/14 (0%; 0–22%) | 14 | 0 | 0.0013 |

Adaptive rule (pre-registered): per cell, take the pad with the larger gated-IS n_eff.

- damaged: 33/41 (80%; 66–90%) recovered (trusted), 7 untrusted, over 41 cells with both pads.
- broken: 4/14 (29%; 12–55%) recovered (trusted), 10 untrusted, over 14 cells with both pads.

Damaged cells, ±0.1 s, by morphology / Omicron SNR bin (gated IS trusted recoveries; Phase C localiser in brackets):

- Blip: 15/15 [14]; Koi_Fish: 17/24 [17]; Whistle: 0/2 [0]
- [100,300): 20/22 [19]; [30,100): 5/7 [5]; [300,inf): 6/11 [6]; [8,30): 1/1 [1]
- trusted cells: d_L JS vs clean median 0.018, max 0.474

Damaged cells, ±0.2 s, by morphology / Omicron SNR bin (gated IS trusted recoveries; Phase C localiser in brackets):

- Blip: 14/15 [14]; Koi_Fish: 16/24 [17]; Whistle: 0/2 [0]
- [100,300): 18/22 [19]; [30,100): 5/7 [5]; [300,inf): 6/11 [6]; [8,30): 1/1 [1]
- trusted cells: d_L JS vs clean median 0.021, max 0.137

Injection reproduction: 110 matched, 0 mismatched (skipped).
