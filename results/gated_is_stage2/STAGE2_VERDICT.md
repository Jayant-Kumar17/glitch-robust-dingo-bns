# Stage 2 verdict

Likelihood checks: A max rel diff 1.9487657061011784e-16, B {"gate": {"max_delta_gated": 3.4924596548080444e-10, "min_delta_ungated": 498698.81244072504}, "misplaced": {"max_delta_gated": 498718.01535654184, "min_delta_ungated": 498698.81244072504}} -> PASS

Decisive set: 10 clean cells with +-0.4 s H1 gates.
- V1 likelihood checks: PASS
- V2 route-A efficiency >= 1%: 3/10 (need 7) -> FAIL
- V3 d_L JS(A,B) <= 0.05 among 3 cells with n_eff >= 100 in both routes: 3 -> PASS
- V4 gated-IS d_L recovers: 9/10 (need 8) -> PASS
  (same cells, network only: 3/10; naive IS: 3/10)
- V5 (reported) eps sensitivity of d_L median, max over cells: 0.016555404427585074 Mpc
- route-A efficiency: median 0.0083, min 0.0005106, max 0.0546

## Decision: PARTIAL

## Other cells

| cell | network recovers | naive IS recovers | gated IS recovers | eff A | n_eff A | d_L med A | JS(A,B) d_L | JS(A,clean) d_L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_h0.2_00 | True | True | True | 0.0690 | 690 | 35.6 | 0.017 | 0.011 |
| clean_h0.2_01 | True | True | True | 0.0892 | 892 | 34.1 | 0.013 | 0.017 |
| clean_h0.2_02 | True | True | True | 0.0897 | 897 | 34.7 | 0.013 | 0.012 |
| clean_h0.2_03 | True | True | True | 0.0759 | 759 | 34.7 | 0.013 | 0.015 |
| clean_h0.2_04 | True | True | True | 0.0624 | 624 | 35.0 | 0.015 | 0.014 |
| clean_h0.2_05 | False | False | True | 0.0137 | 137 | 33.4 | 0.044 | 0.042 |
| clean_h0.2_06 | True | True | True | 0.0553 | 553 | 35.5 | 0.028 | 0.017 |
| clean_h0.2_07 | True | True | True | 0.0812 | 812 | 35.4 | 0.017 | 0.016 |
| clean_h0.2_08 | True | True | True | 0.0521 | 521 | 34.7 | 0.025 | 0.017 |
| clean_h0.2_09 | False | False | True | 0.0057 | 57 | 30.3 | 0.079 | 0.092 |
| clean_h0.8_00 | False | False | False | 0.0002 | 2 | 20.2 | 0.580 | 0.593 |
| clean_h0.8_01 | False | False | True | 0.0018 | 18 | 35.6 | 0.181 | 0.179 |
| clean_h0.8_02 | False | False | True | 0.0043 | 43 | 34.0 | 0.068 | 0.069 |
| clean_h0.8_03 | False | False | True | 0.0083 | 83 | 34.8 | 0.069 | 0.071 |
| clean_h0.8_04 | False | False | False | 0.0004 | 4 | 15.1 | 0.634 | 0.661 |
| clean_h0.8_05 | False | False | False | 0.0001 | 1 | 15.4 | 0.613 | 0.659 |
| clean_h0.8_06 | False | False | True | 0.0002 | 2 | 26.5 | 0.533 | 0.548 |
| clean_h0.8_07 | False | False | False | 0.0003 | 3 | 15.7 | 0.522 | 0.628 |
| clean_h0.8_08 | False | False | True | 0.0005 | 5 | 28.6 | 0.460 | 0.442 |
| clean_h0.8_09 | False | False | False | 0.0003 | 3 | 14.8 | 0.567 | 0.638 |
| rawL1_h0.1 | True | True | True | 0.1074 | 1074 | 34.9 | nan | 0.014 |
| rawL1_h0.4 | True | True | True | 0.0114 | 114 | 34.9 | nan | 0.036 |
