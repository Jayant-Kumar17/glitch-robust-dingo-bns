# Facts for the referee revision

Every number below was read from the file named next to it. Nothing here was re-fit or re-sampled. Stage 3b was still running, so `results/gated_is_stage3b/` was not opened.

`n_eff` is quoted from the file when the file stores it. Where a clean reference stores only an efficiency, `n_eff` is efficiency times the proposal count in that run's config, and the product is shown.

## F1. DINGO-IS on the clean GW170817 package

These are importance-sampling draws on the un-gated event, not network-only quantiles and not gated or poisoned packages. A repeated efficiency across many cell files is one draw, reused.

| folder / file | proposal samples | efficiency | n_eff | seed | date |
| --- | --- | --- | --- | --- | --- |
| `results/dingo_official_control/control_summary.json` | 20000 (`n_samples`) | 0.11128324827509177 | 2225.6649655018355 | not in the file | file mtime 2026-09-04 08:45:39 |
| `results/gw170817_raw_l1/lvk_subtracted_ref_efficiency.json` | 20000 | 0.1089559193193174 | 2179.118386386348 | not in the file | file mtime 2026-11-22 23:31:20 |
| `results/gw170817_raw_l1_v3_band_zero/lvk_subtracted_ref_efficiency.json` | 20000 | 0.11498638447207625 | 2299.727689441525 | not in the file | file mtime 2026-11-23 23:50:48 |
| `results/gw170817_raw_l1_v3_band_noise/lvk_subtracted_ref_efficiency.json` | 20000 | 0.11357314738175703 | 2271.4629476351406 | not in the file | file mtime 2026-11-23 23:52:50 |
| `results/multi_param_validation_v1/real_clean_efficiency.json` | 10000, from `n_eff / sample_efficiency` (the file does not store `n_samples`) | 0.10676655236112835 | 1067.6655236112836 | not in the file | file mtime 2026-11-22 23:14:07 |
| `results/gated_is_stage2/clean_reference.json` plus `results/gated_is_stage2/config.json` | 10000 (`num_samples`) | 0.10184321123277702 | 1018.4321123277702 = efficiency × 10000 | 0 (`config.json` `seed`) | run start `results/stage2_runner.log` line `2026-11-24 01:06:14 [3/3] full run`; file mtime 2026-11-24 03:06:34 |
| `results/gw170817_raw_l1_v3_final/lvk_subtracted_ref_efficiency.json` | 20000 | 0.11136830200522499 | 2227.3660401045 | not in the file | file mtime 2026-11-24 04:40:10; log line `results/phase_c_logs/gw170817_raw_l1_v3_final.log`: `Sampling lvk_subtracted_ref N=20000` |
| `results/multi_param_validation_v3/real_clean_efficiency.json` | 10000, from `n_eff / sample_efficiency` | 0.11606093336509912 | 1160.6093336509912 | not in the file | file mtime 2026-11-24 04:41:51 |
| `results/gated_is_stage3/clean_reference.json` plus `results/gated_is_stage3/config.json` | 10000 (`num_samples`; this run predates `--clean-samples`, and the saved config has no separate clean count) | 0.0999865440944742 (`clean_eff`) | 999.865440944742 = efficiency × 10000 | not in `config.json` | run start `results/stage3_runner.log` line `2026-11-24 08:24:34 [4/4] full run`; file mtime 2026-11-24 08:25:04 |

`lvk_subtracted_ref` is the official demonstration package (`examples/gw170817_raw_l1.py` lines 12 and 317: the key is `event.data`). Stage 2 and Stage 3 draw on `c.event.data` after `examples/gated_is_stage2.py::setup` loads the LOSC frames (`examples/gated_is_stage2.py` lines 188–220 and 539). The paper's 10.0% is the Stage 3 efficiency 0.0999865440944742 rounded to one decimal as a percentage. The paper's 11.1% is the Phase C final `lvk_subtracted_ref` efficiency 0.11136830200522499. They are different draws: different proposal counts (10000 vs 20000), different efficiencies, and different packages. They are not the same run.

## F2. Unsubtracted-L1 and the subtracted reference

All three Phase C final rows are `n_samples` 20000, from the matching `*_efficiency.json` under `results/gw170817_raw_l1_v3_final/`.

| run | file | proposal samples | efficiency | n_eff |
| --- | --- | --- | --- | --- |
| `raw_l1_poisoned` | `results/gw170817_raw_l1_v3_final/raw_l1_poisoned_efficiency.json` | 20000 | 5.0025508781825375e-05 | 1.0005101756365076 |
| `raw_l1_adapt_full` | `results/gw170817_raw_l1_v3_final/raw_l1_adapt_full_efficiency.json` | 20000 | 0.13615854131051414 | 2723.170826210283 |
| `lvk_subtracted_ref` | `results/gw170817_raw_l1_v3_final/lvk_subtracted_ref_efficiency.json` | 20000 | 0.11136830200522499 | 2227.3660401045 |
| Stage 2 `rawL1_h0.1` | `results/gated_is_stage2/cells/rawL1_h0.1.json` key `gated_is_A` | 10000 (`results/gated_is_stage2/config.json` `num_samples`) | 0.10738909713197627 | 1073.8909713197627 |
| Stage 2 `rawL1_h0.4` | `results/gated_is_stage2/cells/rawL1_h0.4.json` key `gated_is_A` | 10000 | 0.011418073955249365 | 114.18073955249365 |

## F3. The Stage 2 ±0.4 s and ±0.8 s cells

Route A is `gated_is_A` in `results/gated_is_stage2/cells/`.

Median efficiency over the ten `clean_h0.4_*` cells, `numpy.median` of the ten `gated_is_A.efficiency` values: **0.00825319788608578**. The ten values are 0.0075388132246871934, 0.03872562768693967, 0.054624979714474636, 0.0162337386801936, 0.005363838044874293, 0.0006309030569845745, 0.00424216425279025, 0.009146664795112319, 0.008967582547484367, 0.0005106018243971858.

Largest-n_eff `clean_h0.8_*` cell: **`clean_h0.8_03`**, `gated_is_A.n_eff` = **83.33765367957683** (`results/gated_is_stage2/cells/clean_h0.8_03.json`).

## F4. Counts

### (a) Fifteen rows with `injected_rho_w` ≤ 1

From `results/stress_real_glitches_v3_final/results.csv`, joined to `glitch_catalogue.csv` on `gs_event_time` = `event_time`. There is no Gravity Spy string id on `results.csv`; the id below is `gravityspy_id`. Each excerpt is three placements. `scale_k` is 0 and `native_snr` is 0 on every one of these 15 rows. Omicron SNR is `gs_snr`.

| gravityspy_id | cells | label | Omicron SNR | catalogue `excerpt_excess_energy_w` | catalogue `excerpt_denoised_energy_w` |
| --- | --- | --- | --- | --- | --- |
| `5GyW2I8RTC` | 45, 46, 47 | Blip | 344.380798339844 | -5707.330448563732 | 49.02781138643231 |
| `0PLIgSeAk2` | 120, 121, 122 | Scattered_Light | 30.0440902709961 | -24.50518006828497 | 223.0833834920236 |
| `hz6EfDcDTj` | 180, 181, 182 | Tomte | 24.6868495941162 | -1209.319879310121 | 849.5290024829467 |
| `8aiQCtoZkf` | 183, 184, 185 | Tomte | 24.2844390869141 | -1689.6712909092855 | 844.4359319901394 |
| `K2mPAUToX3` | 186, 187, 188 | Tomte | 24.1168003082275 | -762.5526270319706 | 816.8946840844894 |

Why the injected power is zero: native mode scales the transplanted waveform by `excerpt.excess_energy_w`, not by `native_energy_w()`.

```348:349:examples/stress_real_glitches.py
    if mode == "native":
        k = transplanter.scale_native(g0, excerpt.excess_energy_w)
```

```301:303:examples/stress_real_glitches.py
    def scale_native(self, g0: np.ndarray, excess_energy_w: float) -> float:
        e0 = self.whitened_energy(g0)
        return float(np.sqrt(max(excess_energy_w, 0.0) / max(e0, 1e-300)))
```

All five catalogue excesses are negative, so `k` is 0, the added strain is zero, and `injected_rho_w` is 0. `GlitchExcerpt.native_energy_w` would have fallen back to the positive denoised energy (`src/adapt/gravity_spy_io.py` lines 598–606), and the catalogue `excerpt_native_snr` column is that fallback. The real-panel injector does not call it.

### (b) Loudness sweep

`results/collapse_threshold_v3/summary.json`: `n_rows` 139, `n_ok` 138, `n_errors` 1. `results/collapse_threshold_v3/results.csv` has 139 rows. One row failed: `cell_id` 7, `family` sine_gaussian, `rho_w_target` 10000, `error` `AssertionError:`. The other 138 rows have an empty error. Counts of those 138, per family and target `rho_w`:

| family | target ρ_w | cells with empty error |
| --- | --- | --- |
| broadband_burst | 10 | 15 |
| broadband_burst | 26.826957952797258 | 5 |
| broadband_burst | 30 | 10 |
| broadband_burst | 71.96856730011521 | 5 |
| broadband_burst | 100 | 10 |
| broadband_burst | 193.06977288832496 | 5 |
| broadband_burst | 517.9474679231213 | 5 |
| broadband_burst | 1389.4954943731375 | 5 |
| broadband_burst | 3727.593720314938 | 5 |
| broadband_burst | 10000 | 5 |
| sine_gaussian | 10 | 14 |
| sine_gaussian | 26.826957952797258 | 5 |
| sine_gaussian | 30 | 10 |
| sine_gaussian | 71.96856730011521 | 5 |
| sine_gaussian | 100 | 10 |
| sine_gaussian | 193.06977288832496 | 5 |
| sine_gaussian | 517.9474679231213 | 5 |
| sine_gaussian | 1389.4954943731375 | 5 |
| sine_gaussian | 3727.593720314938 | 5 |
| sine_gaussian | 10000 | 4 |

70 + 68 = 138 successful rows. The paper's 139 is `n_rows`, including the failed cell. The Phase C report table (`results/PHASE_C_REPORT.md` section 3) prints the successful denominators and sums to 138. The sine-Gaussian ρ_w = 10000 line there is `4/4`, which is this failed cell left out.

## F5. Prior and sky

Loaded with `adapt.dingo_bns_demo.load_bns_checkpoint` (`src/adapt/dingo_bns_demo.py` lines 51–55) from `DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/downloads/dingo-bns-model_GW170817.pt`. The object is a dict. Priors are under `metadata`.

- `metadata.dataset_settings.intrinsic_prior["luminosity_distance"]` = `100.0` (a fixed number, not a distribution string).
- `metadata.train_settings.data.extrinsic_prior["luminosity_distance"]` = `bilby.core.prior.Uniform(minimum=10.0, maximum=100.0)`.
- `metadata.dataset_settings.intrinsic_prior["chirp_mass"]` = `bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.0, maximum=2.2)`. There is no `chirp_mass` entry in `extrinsic_prior`.
- `metadata.dataset_settings.intrinsic_prior["geocent_time"]` = `0.0`.
- `metadata.train_settings.data.extrinsic_prior["geocent_time"]` = `bilby.core.prior.Uniform(minimum=-0.030, maximum=0.030)`.
- `metadata.train_settings.data.extrinsic_prior["ra"]` = `default`, and the same for `dec` and `psi`.
- `metadata.train_settings.data.context_parameters` = `["ra", "dec", "chirp_mass_proxy"]`.
- GNPE chirp kernel, `metadata.train_settings.data.gnpe_chirp`: `{"kernel": {"chirp_mass": "bilby.core.prior.Uniform(minimum=-0.005, maximum=0.005)"}, "order": 0}`. `gnpe_time_shifts` is null.

`build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})` (`src/adapt/dingo_bns_demo.py` lines 420–424) lets the extrinsic dict overwrite the intrinsic one. The sampling prior on luminosity distance is therefore the extrinsic uniform on [10, 100] Mpc, and the sampling prior on chirp mass is the intrinsic uniform-in-components prior on [1.0, 2.2].

Fixed values, not priors. `DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/inference-dingo-pipe/GW170817.ini` line 17:

`fixed-context-parameters = {chirp_mass_proxy: 1.19786, ra: 3.44616, dec: -0.408084}`

`discover_assets` reads that line (`src/adapt/dingo_bns_demo.py` lines 147–151 and 75–101). No other parameter is in the fixed-context dict.

The GW170817 BNS settings that this repo actually loads are that demo ini. `DINGO-BNS/dingo/examples/binary_neutron_stars/inference/GW170817.ini` is a different example (its `trigger-chirp-mass` is 1.487 and `num-gnpe-iterations` is 30) and is not the frozen checkpoint's ini.

## F6. Recovery criterion, quoted

`examples/gated_is_stage3.py` lines 40–41, used to label the archived real-panel rows as damaged or not:

```python
def _recovers(lo, med, hi, clean) -> pd.Series:
    return (med >= 20) & (med <= 50) & ((med - clean["med"]).abs() <= 10) & (hi >= clean["lo"]) & (lo <= clean["hi"])
```

`examples/stress_gw170817.py` lines 104–115, used by Stage 2, Stage 3 and the Phase C real panel:

```python
def _poison_collapsed(ci: Dict[str, float]) -> bool:
    return bool(ci["hi"] < 15.0 or ci["lo"] > 90.0)


def _gated_recovers(ci: Dict[str, float], clean: Dict[str, float]) -> bool:
    if not (20.0 <= ci["med"] <= 50.0):
        return False
    if not _ci_overlap(ci, clean):
        return False
    if abs(ci["med"] - clean["med"]) > 10.0:
        return False
    return True
```

`_ci_overlap` is lines 100–101 of the same file: the intervals overlap unless one lies entirely to the left of the other.

Clean d_L interval each run compared against:

- Phase C real panel. `examples/stress_real_glitches.py` lines 496–499 and 706 write `clean_reference.json` from `_sample_dl` (a network draw, `n_samples` 512 in this archive) and set `gated_recovers` with `_gated_recovers(gated_ci, clean_ci)`. The stored interval, `results/stress_real_glitches_v3_final/clean_reference.json`, is lo 18.47731285095215, med 34.025936126708984, hi 45.645494079589845, `n` 512. This is not an importance-sampled reference.
- Stage 2. `examples/gated_is_stage2.py` lines 539–544 build `clean_ci` from the weighted DINGO-IS samples and pass it into the cell runner. Stored in `results/gated_is_stage2/clean_reference.json` under `d_L`: lo 19.822062491746202, med 34.2196848494327, hi 45.69732668790281.
- Stage 3. `examples/gated_is_stage3.py` draws a fresh clean DINGO-IS reference and stores `clean_ci`. `results/gated_is_stage3/clean_reference.json`: lo 18.87725950524704, med 34.22833370150656, hi 45.854318587868605.

## F7. `naive_is.efficiency`

All 110 files in `results/gated_is_stage3/cells/*.json` contain `naive_is.efficiency`. For each, the archived `weight_naive` column in `results/gated_is_stage3/<id>_samples.csv.gz` was reduced to `(sum w)^2 / sum(w^2) / n`. That matched `naive_is.efficiency` on all 110 files (relative tolerance 1e-6, absolute 1e-8). None were missing the field.

It is assigned from DINGO's `sample_efficiency` and is not recomputed later:

```165:168:examples/gated_is_stage3.py
    rec["naive_is"] = {
        "efficiency": float(resA.sample_efficiency) if resA.sample_efficiency is not None else None,
        "d_L": S2._ci(resA.samples["luminosity_distance"].to_numpy(float), wN),
    }
```

`resA` is the object returned by `S2.draw`, which calls `run_importance_sampling` (`examples/gated_is_stage2.py` lines 215–221). DINGO's `sample_efficiency` is `n_eff / num_samples` on that result.

## F8. Deviations from pre-registration

`git log --follow --format='%h %ad %s' --date=iso`.

### (a) Plan commits against run starts

| document | only commit touching it | run start |
| --- | --- | --- |
| `docs/STAGE2_CRITERIA.md` | `51f2ddd` 2026-11-24 00:53:26 +0500 | `results/stage2_runner.log`: 2026-11-24 01:06:14 |
| `docs/STAGE3_PLAN.md` | `416256d` 2026-11-24 08:19:15 +0500 | `results/stage3_runner.log`: 2026-11-24 08:24:34 |
| `docs/STAGE3B_PLAN.md` | `bc09a21` 2026-11-24 17:55:09 +0500 | `results/stage3b_runner.log`: 2026-11-24 17:55:28, after pytest at 17:55:10 |

Each plan commit is before the corresponding full-run line.

### (b) Edits after the run started

`git log --follow` shows one commit for each of those three files. None was changed by a later commit. So none of the three documents changed after its run started.

### (c) The nine extra v3 excerpts

`native_energy_w` was added in `cc96a9b` 2026-09-19 11:43:09 +0500 (`git log -S native_energy_w -- src/adapt/gravity_spy_io.py`), message `Task 5 follow-ups: repair zero rho_w, low-rho extension, figS4, v2 panel, O3 SNR tail`. Phase C started at 2026-11-24 03:15:07 (`results/overnight_chain.log`). `git log --since='2026-11-24 03:15:00'` on `src/adapt/gravity_spy_io.py` and `examples/stress_real_glitches.py` is empty. The extraction change is before the Phase C run, not during it.

The v3 panel still injected with `excess_energy_w` (F4). The nine excerpts that the diagnosis counts as newly usable are a catalogue-energy change, not a change made after Phase C started. `results/PHASE_C_REPORT.md` section 2a lists nine ids only in v3.

### (d) `_gated_recovers` against release v1.2.0

`git log -S "_gated_recovers"` oldest hit in this clone is `4ff77a5` 2026-08-28 11:18:51 +0500, `Publish glitch-robust DINGO-BNS methods package for public reproducibility.` Later commits move or edit the call sites (2026-09-18 through 2026-11-24). `git tag -l 'v1.2*'` prints nothing: the tag object is not in this clone, so this history cannot say whether `4ff77a5` is before or after the tag. The paper text (`paper/main.tex`) identifies v1.2.0 as a release that still had the packaging error. That is the paper's statement, not a tag date.

### (e) Choices made with dates

- Whitened-excess localiser and the 8σ default: `7ffdf81` 2026-11-24 01:13:17 +0500, `Add whitened-excess localiser and Phase C runner`. Phase C started at 03:15:07 the same day, so the threshold was in the committed code before that run. The false-trigger counts in `results/PHASE_C_REPORT.md` section 1 were produced by the run, not before the choice was committed.
- Stage 3 pads 0.1 s and 0.2 s, and the rule that a recovery counts only if `n_eff` ≥ 100: both are in `docs/STAGE3_PLAN.md`, committed `416256d` at 08:19:15, and the run started at 08:24:34. `TRUST_N_EFF = 100` enters the driver in that same commit (`git log -S TRUST_N_EFF`).
- Stage 2's pre-registered trust use of `n_eff` ≥ 100 is the V3 clause of `docs/STAGE2_CRITERIA.md` (the JS comparison is restricted to cells where both routes have `n_eff` ≥ 100), committed `51f2ddd` at 00:53:26, before the 01:06:14 run.
- Stage 3b's decision to rerun only the untrusted ±0.1 s cells at 10^5 samples, and to keep `n_eff` ≥ 100, is `docs/STAGE3B_PLAN.md`, committed `bc09a21` at 17:55:09, before the 17:55:28 run.
