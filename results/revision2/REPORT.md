# Revision 2 checks

`paper/main.tex` was not edited. The r8 zip was installed into `paper/` (`main.tex`, `refs.bib`, `main.bbl`, and the figure and script files that the zip contains). `paper/figures/fig5_trust_scatter.pdf` was already present and was not in the zip, so it was left as it was. The sentences below are the ones to paste. Archived numbers are in `results/revision2/archived.json`. The residual spectrum is in `results/revision2/residual.json` and `results/revision2/residual_bands.json`. Route B cell records are in `results/revision2/routeb/`.

## 1. Which subset the 0.018 nat median uses

It uses all 33 trusted damaged cells at ±0.1 s, not the 32 trusted passes.

`examples/gated_is_stage3.py` `report()` takes every damaged ±0.1 s cell with `gated_is.trusted` and prints the median of `js_gated_vs_clean["luminosity_distance"]` against the pre-registered 10^4-sample clean reference. That set has 33 cells. The one trusted failure is `damaged_092_h0.1` (JS 0.4736 nat, n_eff 143.7). The median of the 33 is 0.01758 nat, which prints as 0.018. The median of the 32 passes is 0.01755 nat, which also prints as 0.018. The Stage 3 line “median 0.018, max 0.474” exists only if the failure is included.

The installed r8 sentence says “The 32 trusted passes among damaged cells”. Replace that opening with:

> The 33 trusted damaged cells

## 2. Table IV split by group

Same comparison as Table IV: 64-bin weighted JS against the 10^5-sample clean reference (`results/revision/clean_ref_v2`), pooled over pads within each group. Rows are gated IS with n_eff ≥ 100, ordinary IS with n_eff ≥ 100, and the unweighted network (no trust flag). “Unaffected” is the code’s `fine` group: the ungated network passes and the localiser-gated network passes. “Damaged” is an ungated-network failure. “Broken” is an ungated-network pass that the localiser-gated network fails. Entries are median / 90th percentile, in nat.

The floor is the median, over all trusted gated analyses (every group, not damaged only), of the JS between two resamples of the same posterior at the comparison’s effective sample sizes. Unrounded floors: chirp mass 0.00910, mass ratio 0.00984, theta_JN 0.00977, d_L 0.00972, lambda_1 0.00716, lambda_2 0.00758, lambda_tilde 0.00866, t_c 0.00880.

| Group | Row | n | M_c | q | theta_JN | d_L | Lambda_1 | Lambda_2 | Lambda_tilde | t_c | max |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Damaged | Gated, trusted | 64 | 0.0099/0.0303 | 0.0115/0.0310 | 0.0132/0.0427 | 0.0117/0.0322 | 0.0030/0.0110 | 0.0033/0.0114 | 0.0058/0.0199 | 0.0048/0.0195 | 0.0172/0.0491 |
| Damaged | Ordinary, trusted | 81 | 0.0083/0.0277 | 0.0092/0.0150 | 0.0812/0.5411 | 0.0598/0.4381 | 0.0027/0.0091 | 0.0032/0.0085 | 0.0064/0.0233 | 0.0061/0.0249 | 0.0812/0.5411 |
| Damaged | Network | 82 | 0.0226/0.0543 | 0.0033/0.0058 | 0.0443/0.4621 | 0.0330/0.3956 | 0.0741/0.0934 | 0.0797/0.0966 | 0.2108/0.2571 | 0.2342/0.2706 | 0.2504/0.4621 |
| Broken | Gated, trusted | 4 | 0.0232/0.0345 | 0.0312/0.0491 | 0.0332/0.0413 | 0.0328/0.0386 | 0.0081/0.0194 | 0.0131/0.0177 | 0.0106/0.0454 | 0.0137/0.0473 | 0.0498/0.0641 |
| Broken | Ordinary, trusted | 28 | 0.0087/0.0186 | 0.0100/0.0174 | 0.6026/0.6840 | 0.5149/0.6330 | 0.0058/0.0260 | 0.0069/0.0279 | 0.0212/0.0759 | 0.0226/0.0919 | 0.6026/0.6840 |
| Broken | Network | 28 | 0.0297/0.0438 | 0.0029/0.0046 | 0.5807/0.6827 | 0.5134/0.6397 | 0.0757/0.1093 | 0.0804/0.1023 | 0.2153/0.2626 | 0.2407/0.2875 | 0.5807/0.6827 |
| Unaffected | Gated, trusted | 139 | 0.0066/0.0117 | 0.0089/0.0139 | 0.0091/0.0147 | 0.0084/0.0144 | 0.0022/0.0053 | 0.0025/0.0061 | 0.0040/0.0122 | 0.0037/0.0116 | 0.0119/0.0235 |
| Unaffected | Ordinary, trusted | 139 | 0.0061/0.0099 | 0.0084/0.0115 | 0.0151/0.0680 | 0.0134/0.0498 | 0.0020/0.0049 | 0.0023/0.0049 | 0.0042/0.0126 | 0.0041/0.0125 | 0.0186/0.0686 |
| Unaffected | Network | 140 | 0.0248/0.0385 | 0.0037/0.0047 | 0.0070/0.0434 | 0.0067/0.0313 | 0.0741/0.0872 | 0.0777/0.0906 | 0.2088/0.2408 | 0.2299/0.2582 | 0.2299/0.2582 |

Trusted gated IS on damaged cells is not at or below the floor on every parameter. The median is above the floor for chirp mass (0.00993 > 0.00910), mass ratio (0.01153 > 0.00984), theta_JN (0.01324 > 0.00977), and d_L (0.01174 > 0.00972). It is at or below the floor for lambda_1 (0.00297), lambda_2 (0.00325), lambda_tilde (0.00581), and t_c (0.00479). The pooled Table IV row stays at or below the floor because the 139 unaffected trusted analyses dominate the 64 damaged ones.

## 3. Glitch-clustered permutation test

This replaces the gated-IS-versus-ordinary-IS McNemar test on damaged cells. The outcomes are unchanged: gated IS counts only if it is trusted and passes; ordinary IS counts if it passes, with no trust requirement. That is the comparison in `examples/revision_analysis.py` `mcnemar_rows`.

Cells are clustered by `gs_event_time` (18 glitches, 41 cells at each pad). One fair coin per glitch swaps the two methods’ outcomes on every cell of that glitch. Cells from different glitches are not exchanged. The statistic is (gated-IS only) minus (ordinary-IS only). With 10^5 permutations and seed 20261125, the two-sided p-value is (1 + number of permutations with |S*| ≥ |S|) / (10^5 + 1).

| Pad | Gated only | Ordinary only | Gated total | Ordinary total | p | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.1 s | 3 | 0 | 32 | 29 | 0.250 | 0.25 |
| ±0.2 s | 9 | 0 | 30 | 21 | 0.00779 | 0.00391 |

The unrounded p-values are 0.25028749712502874 (25,028 permutations at least as extreme) and 0.007789922100778992 (778 permutations). At ±0.1 s the clustered test and the exact binomial agree. At ±0.2 s the clustered p is about twice the exact McNemar p of 0.00390625, which the paper rounds to 0.004. The gain of 9 cells at ±0.2 s is still unlikely under this null, but the cells are not independent.

## 4. Route B

Route B reweights the first 10^4 rows of the R1 clean-data proposal (`results/revision/clean_ref_v2/samples.csv.gz`) to the gated likelihood of each damaged analysis. It does not draw a new network proposal. The gate is the Stage 3 gate: excess power at 8 sigma, the cell’s pad, zero-fill, and the 20–23 Hz gate band. Trust is n_eff ≥ 100. The d_L median shift is the route B median minus the Stage 3 gated median. The JS is a 16-bin weighted divergence of those two d_L samples. No cutoff on the shift or the JS was used to define agreement. The synthetic phase stored with the clean proposal is passed to the waveform generator; it is not a prior column. Eight network samples are drawn only to build the likelihood object.

All 82 analyses finished with no injection or localiser error. The trust flag agrees in 62 of 82 analyses: 31 of 41 at ±0.1 s and 31 of 41 at ±0.2 s.

| Pad | Agree | Both trusted | Both untrusted | Route B only | Stage 3 only | Route B trusted | Stage 3 trusted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.1 s | 31/41 | 31 | 0 | 8 | 2 | 39 | 33 |
| ±0.2 s | 31/41 | 31 | 0 | 10 | 0 | 41 | 31 |
| Both | 62/82 | 62 | 0 | 18 | 2 | 80 | 64 |

The 18 route-B-only cases are Stage 3 analyses with n_eff below 100 whose clean-data proposal clears 100. The two Stage-3-only cases are `damaged_092_h0.1` (Stage 3 n_eff 143.7, route B 24.4) and `damaged_101_h0.1` (174.2 and 31.0). Route B passes the distance criterion in 80 of 82 analyses. The two failures are both `damaged_092` (section 5).

d_L median shift, route B minus Stage 3, in Mpc:

| Pad | Median | Mean | Minimum | 5th | 95th | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.1 s | −0.23 | −0.17 | −3.64 | −1.25 | +2.18 | +2.87 |
| ±0.2 s | −0.47 | −0.75 | −17.51 | −2.81 | +1.41 | +7.43 |
| Both | −0.30 | −0.46 | −17.51 | −2.81 | +1.61 | +7.43 |

The median absolute shift is 0.59 Mpc. Twenty-eight analyses move by more than 1 Mpc, and two by more than 5 Mpc (`damaged_094_h0.2`, +7.43 Mpc; `damaged_097_h0.2`, −17.51 Mpc). Both of those are Stage 3 results with n_eff below 4.

16-bin d_L JS, in nat:

| Pad | Median | Mean | Minimum | 5th | 95th | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.1 s | 0.0042 | 0.0150 | 0.0017 | 0.0019 | 0.050 | 0.165 |
| ±0.2 s | 0.0068 | 0.0441 | 0.0016 | 0.0030 | 0.277 | 0.433 |
| Both | 0.0057 | 0.0296 | 0.0016 | 0.0020 | 0.162 | 0.433 |

Twenty-three analyses exceed 0.01 nat, ten exceed 0.05, and seven exceed 0.1. The large shifts and JS values are the low-n_eff Stage 3 posteriors, not the trusted ones. Where both flags are trusted, the posteriors are the closer ones; the disagreements are almost all an untrusted Stage 3 flag against a trusted route B flag.

Every analysis has exactly 6 non-finite gated log-likelihoods (492 in total). They are the same clean-proposal rows: the waveform generator rejects samples with a negative tidal deformability. Those six weights are zero in every cell.

## 5. Leakage on the two trusted failures

`damaged_092` at both pads. The same 10^4 clean-proposal samples are scored twice with the gated log-likelihood. One score uses the injected H1 strain, glitch included inside the gate. The other uses that strain with the injected glitch zeroed only on the gate mask, tapers included (`gate_mask_from_windows`). Outside the mask the two strains match. The difference is log L(with glitch) minus log L(glitch removed inside the gate). The packaged strain is the archived clean waveform plus `package_fd` of the added time series, which is how the panel injection was written.

|  | ±0.1 s | ±0.2 s |
| --- | ---: | ---: |
| Gate-mask samples | 1310 | 2130 |
| Non-zero glitch samples inside the mask | 1310 | 2130 |
| Non-zero glitch samples outside the mask | 17120 | 16300 |
| Fraction of glitch energy inside the mask | 0.977 | 0.990 |
| Finite differences | 9994 | 9994 |
| Mean (nat) | −0.01977 | −0.00713 |
| Standard deviation (nat) | 6.9×10⁻⁶ | 3.5×10⁻⁵ |
| Minimum (nat) | −0.01981 | −0.00845 |
| Median (nat) | −0.01977 | −0.00713 |
| Maximum (nat) | −0.01959 | −0.00701 |
| Maximum absolute value (nat) | 0.01981 | 0.00845 |
| Fraction with \|Δ\| > 0.1 | 0 | 0 |
| Fraction with \|Δ\| > 1 | 0 | 0 |

The coloured glitch is non-zero on its whole placed window, so the inside count equals the mask length. Almost all of its energy is on the mask. Removing that in-gate glitch shifts every finite log-likelihood by nearly the same amount, a few hundredths of a nat, with a scatter of 10⁻⁵ nat. A constant shift cancels in the normalized weights. The in-gate glitch is not what moves the posterior.

Route B does not rescue the failure. At ±0.1 s it is untrusted (n_eff 24.4) and fails, with d_L median 17.63 Mpc against the Stage 3 median 15.45 Mpc. At ±0.2 s it is trusted (n_eff 296.5) and still fails, with median 23.40 Mpc against 22.34 Mpc (shift +1.06 Mpc, 16-bin JS 0.013 nat). The two trusted failures are a property of the gated target, not of the network proposal and not of leakage through the gate.

## 6. Weights, seeds, the glitch-only estimate, and the 0.26 residual

### Non-finite log weights

The Stage 3, unaffected, and ungated-IS sample files store normalized weights, not log weights. `reweight` maps a non-finite log weight to zero. An exact zero can also be underflow of a finite log weight. Stored non-finite weights and exact zeros are counted separately.

| Method | Files | Weights | Stored non-finite | Exact zeros |
| --- | ---: | ---: | ---: | ---: |
| Gated IS | 213 | 2,130,000 | 0 | 3,180 |
| Ordinary IS, gated package | 213 | 2,130,000 | 0 | 3,180 |
| Ordinary IS, ungated package | 192 | 1,920,000 | 0 | 40,528 |

On 20 Stage 3 files (200,000 weights) the 337 exact zeros are the same rows in both weight columns: 0 gated-only zeros and 0 ordinary-only zeros. The gated likelihood does not add zeros of its own. The rate, 3,180 / 2,130,000 = 0.149%, matches the clean reference.

The 10^5-sample clean reference does store the log columns. Its log weight, reconstructed as log prior + log likelihood − log prob + delta_log_prob_target, is non-finite for 139 rows. The same 139 rows are non-finite in log likelihood, log prior, and log prob. The stored weight column has 0 non-finite values and 139 exact zeros. That is the shared DINGO failure mode behind the gated and ordinary zeros.

The ungated-IS files have more exact zeros, 40,528 / 1,920,000 = 2.11%. Their log weights were not stored, so this count mixes non-finite log weights with underflow. It is higher than the clean-data rate.

Route B, above, has a separate and uniform 6 non-finite gated log-likelihoods per damaged analysis, from negative tidal deformabilities in the clean proposal.

### Trust flips for n_eff below 300

Fifty-six Stage 3 gated analyses have n_eff below 300. The only second draw in the archive is R4 route A: one new `draw()` of the same gated package, and `draw()` sets no seed. It covers 8 cells, and all 8 have Stage 3 n_eff below 300. Trust flips in 2 of those 8, both from trusted to untrusted:

| Cell | Stage 3 n_eff | Route A n_eff |
| --- | ---: | ---: |
| damaged_087_h0.2 | 133.0 | 46.6 |
| damaged_092_h0.1 | 143.7 | 25.3 |

The other six do not flip: `damaged_037_h0.2` (137 to 128), `damaged_092_h0.2` (154 to 218), and `damaged_106_h0.1` (153 to 135) stay trusted; `damaged_093_h0.1`, `damaged_169_h0.1`, and `damaged_170_h0.1` stay untrusted. The remaining 48 analyses with n_eff below 300 have no second seed in the archive. Stage 3b is not a seed comparison: it changes the sample size as well as the draw.

### Glitch-only estimate used for the panel

The panel is `results/stress_real_glitches_v3_final`. Its `stress_config.json` records `panel = wide_snr_native` and `severity_modes = ["native"]`. Glitch cells call `inject_excerpt_into_event` with `use_denoised=True` (`use_denoised = not cell["noise"]`). The estimate is `extract_glitch_excerpt` in `src/adapt/gravity_spy_io.py`, then the native branch of `inject_excerpt_into_event` in `examples/stress_real_glitches.py`.

1. Read 8 s of H1 open strain centred on the Gravity Spy `event_time`.
2. Whiten it with `whiten_with_own_background`. The Welch PSD uses the off-window samples, median average, `nperseg = min(4096, background_length // 4)`, and a Tukey window with alpha `2 * 0.1 * f_s / n`. The band is `max(20, f_min)` to `min(1500, f_max)`. The inverse FFT is divided by the off-window standard deviation, so the background has unit variance.
3. Cut `event_time ± max(duration, 0.5 s)` and apply a Tukey window with alpha 0.1.
4. Build the glitch-only series with `stft_denoise_window`: STFT with `nperseg` 512 and hop 128, keep pixels whose power exceeds 8 times the per-frequency median off-window power, and only inside the trigger duration plus 0.15 s. Invert the STFT and apply the same Tukey window. This product is `excerpt.denoised`.
5. Record the whitened excess energy `sum((w * taper)^2) − sum(taper^2) * background_variance`. If that excess is not positive, the native scale below is zero.
6. Colour with `colour_with_asd`: a 2 Hz running-median smooth of the GW170817 H1 ASD, zero-pad by a factor of 4, multiply the in-band FFT by the smoothed ASD, inverse FFT, and re-taper with alpha 0.1. Bins outside `f_min` to `f_max` are zero.
7. Native scale: whiten the coloured series in a 4 s pad and multiply by `sqrt(excess_energy / that whitened energy)`. This is `Transplanter.scale_native`. It does not use a separate `native_energy_w` field.
8. Place the scaled series with `place_series` at `t_peak = (T − time_buffer) + t_rel`. Add it to the H1 time series. Add `package_fd` of it (`td_to_fd_strain`) to the archived H1 waveform. The analysis ASD is left unchanged.

The prose copy of this recipe is `excerpt_recipe` in `examples/stress_real_glitches.py` and in the panel’s `stress_config.json`.

### Cause of the 0.26 residual in Equation (1)

Equation (1) is `package_fd`: Tukey window with alpha `2 * 0.4 / 128`, real FFT divided by `f_s`, then multiplication by `exp(−2 π i f t_b)` with `t_b = 2 s`. On the strain loaded by `load_full_event_td` (gwpy read of the whole LOSC frame, then crop to `[t_event − 126 s, t_event + 2 s]`), that transform matches DINGO’s current `data_to_domain` to a whitened signal-to-noise ratio of 1.7×10⁻⁹ in H1, 1.6×10⁻⁹ in L1, and 5.0×10⁻¹⁰ in V1. The 0.26 is not a bug in that implementation.

The same transform differs from the strain stored in the event HDF5 by a whitened residual of 0.2634 in H1, 0.2696 in L1, and 0.2773 in V1. The paper’s 0.26 is the H1 number. `data_to_domain` versus the archive equals `package_fd` versus the archive. The H1 archive has whitened norm 620.7, so the relative in-band mismatch is 5.4×10⁻⁴. A whitened complex least-squares scale of `package_fd` over the archive is 0.99999991 + 0.000269 i, absolute value 0.99999995. It is not a scale error. The phase slope of the ratio is 3.2×10⁻⁷ rad/Hz, which is a delay of −56 ns. A one-sample shift would be 244 μs and would leave a residual of order the data norm, not 0.26.

The H1 residual by frequency band is 0.0006 (23–30 Hz), 0.0014 (30–50), 0.0041 (50–100), 0.0214 (100–300), 0.0987 (300–800), and 0.2433 (800–1535.3). The power above 800 Hz is 85% of the residual. In segment time the in-band residual is spread out: in 4 s chunks its energy fraction ranges from 1.5% to 4.1%, against 3.1% for a uniform split, including the chunks at the ends and at merger.

The HDF5 strain was written by dingo-pipe `DataGeneration.save_hdf5` from Bilby’s `frequency_domain_strain`, then `FrequencyDomain.update_data` (zero below `f_min`, truncate above `f_max`) and `time_translate_data(post_trigger_duration)`. Bilby’s `nfft` is `rfft / f_s`, and its Tukey alpha is `2 * roll_off / T`, the same operations as Equation (1). Re-reading the H1 frame today with `bilby.gw.utils.read_frame_file(start, end, channel="H1:LOSC-STRAIN", resample=4096)` does not reproduce the archive: the returned series has `t0 = 1187008756.4201663` rather than the requested `1187008756.42`, and its windowed FFT has whitened residual 732 against the archive. The requested start falls 0.32 of a sample off the 4096 Hz grid of a frame that begins at integer GPS 1187007040 (`1716.42 * 4096 = 7030451.2`). The crop this code uses matches the archive to 0.26; the current start/end read does not match it at all.

So the 0.26 is the high-frequency difference, spread through the segment, between the gwpy crop Equation (1) is applied to now and the strain samples Bilby Fourier-transformed when the demo file was written. It is not the cyclic shift, not a normalisation, and not a disagreement between `package_fd` and current DINGO.
