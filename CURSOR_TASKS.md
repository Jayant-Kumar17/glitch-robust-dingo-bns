# Cursor tasks — glitch-robust-dingo-bns (paper finalisation)

Repo: `glitch-robust-dingo-bns`. Read `paper/SCOPE.md`, `paper/METHOD.md`, `examples/README.md`
and `results/README.md` before starting. Conventions: `pip install -e .`,
`export PYTHONPATH=src:examples`, `export KMP_DUPLICATE_LIB_OK=TRUE`. DINGO-BNS weights are
frozen — never fine-tune them. Every new driver goes in `examples/`, writes JSON summaries +
CSV rows + a `REPRODUCE.md` into its own `results/<name>/` dir, and is added to the tables in
`examples/README.md` and `results/README.md`. Reuse existing library functions
(`adapt.glitch_excision.rebuild_event_from_gated_td`, `apply_excision_to_event_data`,
`adapt.event_glitch_io.inject_h1_glitch_into_event`, `adapt.dingo_bns_demo.run_baseline_sampling`)
rather than re-implementing. Do the tasks in order; each is independent and can be committed
separately. Do not touch archived `results/*_v1/` directories.

---

## Task 1 — Paper figures 2 and 3 (`examples/paper_figures_2_3.py`)

Goal: two publication figures that the archived results cannot produce because they need raw
strain and posterior samples.

Inputs: the outputs of `examples/official_control.py` in `results/dingo_official_control/`
(`poison_nn_samples.hdf5`, `gated_nn_samples.hdf5`, `gated_is_samples.hdf5` if present) and the
official clean IS result (`--control-hdf5`, default `DEMO_RESULT` in `official_control.py`).
Load samples with dingo's `Result` / `GwResult` classes, as `official_control.py` already does.
Regenerate the injected glitch and detector output deterministically with the same flags as the
archived control run (`--seed 0 --f0 100 --q 5 --t-rel -1.0 --snr-amp-scale 8.0`).

Figure 3 (`fig3_posteriors.pdf/.png`, 7.0 in wide, 2×3 panels): 1-D marginals of
`luminosity_distance`, `chirp_mass`, `mass_ratio`, `theta_jn`, `ra`, `dec` for three runs —
clean (blue `#1f77b4`), poisoned (red `#c0392b`), gated (green `#2a9d8f`). Step histograms,
60 bins, common x-range per panel from the 0.1–99.9 percentiles across runs, weights applied
for IS results. Annotate the d_L panel: "poisoned: saturates at 10 Mpc prior floor". Print the
JS divergence (nat) clean-vs-gated per panel in the corner, taken from
`comparison_report.json`. No titles; axis labels with units.

Figure 2 (`fig2_injection_gate.pdf/.png`, 3.5 in wide, 3 stacked panels sharing time axis,
4 s H1 analysis crop centred on the trigger):
 (a) whitened H1 strain with the glitch, gate window shaded green (alpha 0.25);
 (b) the whitened spectrogram the detector sees (use `build_event_spectrogram_stack`, log
     colour, 32×128 grid, y in Hz);
 (c) detector per-time-bin probability with the event threshold as a dashed line.
Use `analysis_crop_bounds` / `load_event_td_crops` for the crop and the released detector
checkpoint in `checkpoints/glitch_detector_v1/`.

Both figures: `matplotlib`, `font.size 8`, `pdf.fonttype 42`, `savefig.bbox tight`, no
top/right spines, saved to `paper/figures/`. Add a `--outdir` flag. Add a `make figures`
target or a line in `paper/REPRODUCE.md`.

Acceptance: both PDFs render, d_L panel visibly shows collapse vs. recovery, the script runs
from a clean shell with only the existing assets.

---

## Task 2 — Clean-gate cost control (`examples/clean_gate_control.py`)

Goal: show that applying a gate to *clean* (glitch-free) GW170817 data does not change the
posterior — i.e. a false-positive gate is harmless. This is the missing counterpart to the
existing `tier_c_clean_fp` (which only shows the detector stays silent) and
`clean_noop_bit_exact` (no-fire path is bit-exact).

Procedure:
1. Load the clean GW170817 event package (no injection).
2. For each of N=20 gate placements: forced `GateWindow` on H1, half-width 0.4 s, centre
   `t_rel` drawn uniformly in [−2.0, −0.2] s relative to the trigger (i.e. inside the
   analysis crop, over the inspiral). Apply the full `adapt_full` recipe
   (Tukey gate → matched-delta rebuild → original ASD) via `rebuild_event_from_gated_td`.
3. Sample n=512 with the frozen baseline (`run_baseline_sampling`) for each placement, plus
   once for the untouched clean package.
4. Also run 5 placements with the gate on **both** H1 and L1 at the same `t_rel`.

Record per placement: d_L (lo/med/hi at 5/50/95%), chirp_mass, mass_ratio, theta_jn same
percentiles; `gated_recovers` using the same criterion as `stress_gw170817._gated_recovers`
against the clean reference; JS divergence vs. clean for the five core params (reuse the JS
helper from `official_control.py`); fraction of in-band signal power removed by the gate.

Outputs: `results/clean_gate_control_v1/{results.csv, summary.json, REPRODUCE.md}` and a
figure `paper/figures/figS2_clean_gate_cost.pdf` — d_L median ± 90% CI vs. gate centre
`t_rel`, with the clean CI as a horizontal band. Summary must report: recovery rate, max JS,
mean |Δ d_L median|.

Acceptance: script runs end-to-end on CPU in < 10 min; summary.json has the four numbers
above. Report them back verbatim when done.

---

## Task 3 — Real Gravity Spy glitch panel (`examples/stress_real_glitches.py`)

Goal: replace "synthetic glitches only" with a small panel of **real** LIGO glitches
injected into the GW170817 package. This is the single biggest robustness claim we can add.

Data acquisition (write `src/adapt/gravity_spy_io.py`):
1. Use the public Gravity Spy O3 classification table
   (Zenodo record for "Gravity Spy machine learning classifications of LIGO glitches from O3",
   Glanzer et al. 2023; CSV columns include `event_time`, `ifo`, `ml_label`, `ml_confidence`,
   `snr`, `duration`, `peak_frequency`). Download once into `data/gravity_spy/` (gitignored),
   cache the filtered subset as a small CSV that *is* committed.
2. Select glitches: `ifo == H1`, `ml_confidence >= 0.95`, `snr` in [8, 30],
   labels: `Blip` (10), `Scattered_Light` (5), `Whistle` (5), `Koi_Fish` (5) → 25 total.
   Require `duration <= 3 s`. Take the top-confidence ones per label.
3. For each: fetch 8 s of H1 open strain centred on `event_time` with
   `gwpy.timeseries.TimeSeries.fetch_open_data('H1', t0-4, t0+4, sample_rate=4096)`.
   Whiten *for extraction only* is NOT wanted — we need the raw glitch. Instead:
   high-pass at 20 Hz, take the segment, and **estimate the glitch waveform as
   raw strain minus a smooth background** is unreliable, so use the simpler, defensible
   approach: treat the 8 s raw segment (band-passed 20–1500 Hz, Tukey 0.1 taper) as the
   "glitch + local noise" excerpt and inject it *additively* into the GW170817 H1 strain at
   `t_rel` drawn from the same range as the synthetic grid. Document in the summary that
   this adds real O3 noise as well as the glitch (it does; say so).
   Scale amplitude so in-band severity matches the synthetic midpoints {3, 6, 10} using the
   same `rms_inband` convention as `inject_h1_glitch_into_event`.
4. If fetch_open_data fails for an event (data gap), skip and log it; keep going.

Experiment: 25 glitches × 3 severities × 1 seed × ASD policy `stationary` = 75 cells
(add `welch` policy only if runtime allows). Per cell, run exactly the same three arms as
`stress_gw170817.py`: poisoned, detector-gated (`adapt_full`, event threshold 0.728),
centred-oracle (±0.4 s at the Gravity Spy `event_time`). Same recovery/collapse criteria,
same CSV columns as `results/stress_test_excision_v1/results.csv`, plus columns
`gs_label`, `gs_event_time`, `gs_snr`, `gs_duration`, `gs_confidence`.

Outputs: `results/stress_real_glitches_v1/{results.csv, summary.json, failures.csv,
glitch_catalogue.csv, REPRODUCE.md}`; summary.json follows the schema of
`stress_test_excision_v1/summary.json` with `by_family` keyed on `gs_label`.
Figure `paper/figures/fig8_real_glitches.pdf` (3.5 in): grouped bars per label —
poison collapse, gated recovery, oracle recovery; detector fire rate as a tick.

Acceptance: ≥ 60 cells complete; summary reports overall poison collapse, gated recovery,
oracle recovery, detector fire rate, mean |Δ d_L| on successes. Report these back verbatim.
Add a unit test in `tests/` that the catalogue filter returns ≥ 20 rows from the cached CSV.

---

## Task 4 — Release hygiene

1. Ensure the repo is public. Confirm `pip install -e .` + the commands in
   `paper/REPRODUCE.md` run from a fresh clone (do it in a temp dir; fix anything broken).
2. Add `paper/figures/` and `paper/make_figures.py` (provided separately; regenerates
   Figs 1, 4, 5, 6, 7, S1 from `results/`). Add a `--repo` default of `.`.
3. Add a `CITATION.cff` (author, title from `paper/SCOPE.md`, version 1.0.0, repo URL).
4. Update `README.md`: one-paragraph summary with the headline numbers (46% collapse →
   86% recovery on GW170817; 92% synthetic; 0% for FFT replacement), a "Results at a glance"
   table linking each `results/` dir, and the figure-regeneration command.
5. Tag `v1.0.0` and create a GitHub release with the detector checkpoint attached.
6. Do NOT create a Zenodo deposit automatically — leave a `docs/ZENODO.md` note with the
   steps (link GitHub → Zenodo, publish release, paste DOI into README and CITATION.cff).

---

## Reporting back

For Tasks 2 and 3, paste the `summary.json` contents into the PR description. For Task 1,
attach the two PNGs. Keep every new script runnable with `--smoke` (2 cells / 64 samples)
for CI.
