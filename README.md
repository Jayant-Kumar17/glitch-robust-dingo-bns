# ADAPT

**A**daptive **D**istributed **A**strophysics for **P**arameter Es**t**imation

A continuous hub-and-spoke inference framework for gravitational-wave parameter
estimation. See the preliminary report for the full design; this repository
tracks the implementation as it's built out over time.

## Setup

The project uses the `adapt_env` conda environment, which already has the
scientific dependencies installed (`gwpy`, `torch`, `torchdiffeq`, `PyCBC`,
`lalsuite`, `bilby`, etc.).

```bash
conda activate adapt_env
pip install -e .
```

Note: `PyCBC 2.8.2` is incompatible with very recent `astropy`/`setuptools`
releases (it imports the now-removed `astropy.cosmology.core` shim and the
now-optional `pkg_resources`). If `import pycbc.waveform` fails, pin:

```bash
pip install "astropy==7.2.2" "setuptools==69.5.1"
```

## Project layout

```
src/adapt/
  physics.py         # chirp mass / total mass / mass ratio / effective spin helpers
  router.py          # component-mass boundary router (BNS / BBH / AMBIGUOUS)
  injection.py        # synthetic waveform injection into real noise (Section 4.3 preview)
  gwosc_events.py     # live GWOSC lookups: published event parameters + real strain files
  noise_analytics.py  # isolated single-detector rich noise profiling (local encoder)
tests/               # standalone test/verification scripts
plot_results.py      # publication-quality figure from the simulation-batch CSV
results/             # generated CSVs and figures (timestamped per run)
```

## Routing scheme

`MatchedFilterRouter` classifies strictly on component masses, targeting the
two source classes with mature deep-learning parameter-estimation networks:

- **BNS** -- both component masses `<= ns_max` (default 2.2 M_sun).
- **BBH** -- both component masses `>= bh_min` (default 5.0 M_sun).
- **AMBIGUOUS** -- everything else (asymmetric NSBH systems, lower-mass-gap
  objects). NSBH has no widely accepted PE network, so such triggers are
  deliberately routed to traditional, non-ML offline analysis rather than
  forced into a BBH/BNS pathway.

`route_event(m1, m2, chi_eff=0.0)` returns `{"route", "confidence"}` where
confidence is `1.0` for a clean BNS/BBH classification and `0.5` for
AMBIGUOUS. (`chi_eff` is accepted for pipeline compatibility but does not
change the boundary decision.)

## Local noise profiling (isolated module)

`src/adapt/noise_analytics.py` is a standalone single-detector subsystem
(not yet wired into the router). It builds a fixed-length **Rich Noise
Profile** from real GWOSC strain (default 256 s @ 4096 Hz after an
explicit `.resample()`), combining:

- Welch PSD features on a log-spaced grid from **20 Hz** to Nyquist
- Windowed std / skewness / excess kurtosis (default 4 s windows)

If GWOSC is unavailable, `fetch_background_strain` falls back to colored
Gaussian noise from `pycbc.psd.aLIGOZeroDetHighPower` + `noise_from_psd`.
`LocalNoiseTracker` maintains a sliding history and an environmental
drift delta; `plot_rich_profile` writes a timestamped multi-panel PDF to
`results/rich_noise_profile_<ts>.pdf`.

### Long-duration real noise validation campaign

```bash
python scripts/run_noise_validation.py
```

Fetches a continuous **2048 s** H1 block from GWOSC (GPS `1240559616`),
injects a loud BBH (40+30 M☉) at the stream midpoint, then rolls
`LocalNoiseTracker` over **256 s** windows stepped by **32 s** (57
updates). Writes `results/real_validation_profile_<ts>.pdf`. The first
GWOSC fetch may download a large continuous-archive file (~500 MB);
subsequent runs use the cache when available. If GWOSC fails or the
fetch exceeds 600 s (override with `ADAPT_FETCH_TIMEOUT_S`), the script
continues on the colored-noise fallback.

## Running tests

Test scripts are plain runnable Python scripts (no `pytest` required):

```bash
conda activate adapt_env
python tests/test_router.py                      # fast, no network needed
python tests/test_noise_analytics.py             # offline noise-profile unit tests
                                                    # (fallback, glitch sensitivity, tracker + PDF)
python tests/test_noise.py                        # downloads ~10s of real noise from GWOSC
python tests/test_injection.py                     # downloads a real 64s noise segment, then
                                                    # injects synthetic BNS/BBH waveforms and
                                                    # verifies the router; downloads a large
                                                    # (~500MB) continuous-archive GWOSC file
python tests/test_known_answer_validation.py       # (a) synthetic full-metadata self-consistency
                                                    # check, (b) router vs. published parameters
                                                    # for 4 real confirmed events (fast, small API calls)
python tests/test_real_event_full_validation.py    # downloads real raw strain (small ~1MB per-event
                                                    # files, not the continuous archive) + published
                                                    # parameters for GW150914/GW170817, runs the
                                                    # router, and checks it against the verified
                                                    # classification
python tests/test_large_scale_validation.py        # runs the router against every confirmed event
                                                    # with published masses from GWOSC's cumulative
                                                    # GWTC catalog (through GWTC-5.0, ~280+ events).
                                                    # Takes several minutes (per-event GraceDB calls).
python tests/test_simulation_batch.py              # Section 4.3-style campaign: 1000 synthetic
                                                    # BNS/BBH draws. For EVERY sample: real
                                                    # IMRPhenomD/LALSimulation waveform + mocked
                                                    # matched-filter noise + boundary router
                                                    # score. Writes a timestamped
                                                    # results/simulation_batch_<ts>.csv
                                                    # (includes peak/RMS strain columns).
python plot_results.py                             # reads the latest simulation_batch_<ts>.csv and
                                                    # writes a timestamped two-panel figure to
                                                    # results/router_performance_<ts>.pdf (vector)
```

Generated CSVs and figures are timestamped (e.g. `simulation_batch_20260716_185310.csv`,
`router_performance_20260716_185318.pdf`), so each run adds a new file
rather than overwriting the previous one.

Note on GWOSC downloads: the continuous archive (`test_noise.py`,
`test_injection.py`) always serves ~4096-second (~500MB) files regardless
of the window requested. The event API (`test_known_answer_validation.py`,
`test_real_event_full_validation.py`) instead serves small, pre-cut ~32s
segments for confirmed events, which is much faster -- prefer it when
possible.

## Progress

- [x] Component-mass boundary routing (BNS / BBH / AMBIGUOUS), targeting the
      two source classes with mature PE networks and routing NSBH/mass-gap
      systems to offline analysis -- `src/adapt/router.py`
- [x] Synthetic injection pipeline for end-to-end router verification against
      real detector noise -- `src/adapt/injection.py`
- [x] Known-answer validation: synthetic full-metadata self-consistency checks,
      plus router decisions checked against real confirmed events (live
      published parameters and real raw strain from GWOSC) -- `src/adapt/gwosc_events.py`,
      `tests/test_known_answer_validation.py`, `tests/test_real_event_full_validation.py`
- [x] Large-scale validation against all confirmed events with published masses
      from GWOSC's cumulative GWTC catalog (through GWTC-5.0, ~280+ events),
      checked against LVK's real-time p_astro classification where public:
      0 hard mismatches, NSBH/MassGap events routed to AMBIGUOUS as designed
      -- `tests/test_large_scale_validation.py`
- [x] Distributed simulation campaign (Section 4.3): 1000 synthetic BNS/BBH
      draws, each with a real IMRPhenomD waveform + mocked matched-filter
      noise into the boundary router (99.6% exact match, 0 hard mismatches,
      100% safe-path rate) -- `tests/test_simulation_batch.py`, `plot_results.py`
- [x] Isolated single-detector rich noise profiling: Welch PSD (>= 20 Hz) +
      windowed higher-order moments, GWOSC fetch with colored-noise fallback,
      epoch-aware waveform injection sandbox, LocalNoiseTracker drift, and
      timestamped vector diagnostics -- `src/adapt/noise_analytics.py`,
      `tests/test_noise_analytics.py`
- [ ] NSBH-specific routing refinement
- [ ] Wire noise profiler into the dual-pathway backbone (integration phase)
- [ ] Dual-pathway global backbone
- [ ] Continuous training hub
