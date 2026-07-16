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
  physics.py       # chirp mass / total mass / mass ratio / effective spin helpers
  router.py        # component-mass boundary router (BNS / BBH / AMBIGUOUS)
  injection.py      # synthetic waveform injection into real noise (Section 4.3 preview)
  gwosc_events.py   # live GWOSC lookups: published event parameters + real strain files
tests/             # standalone test/verification scripts
plot_results.py    # publication-quality figure from the simulation-batch CSV
results/           # generated CSVs and figures (timestamped per run)
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

## Running tests

Test scripts are plain runnable Python scripts (no `pytest` required):

```bash
conda activate adapt_env
python tests/test_router.py                      # fast, no network needed
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
python tests/test_large_scale_validation.py        # runs the router against every confident event
                                                    # published across GWTC-1/2.1/3 (~90 real
                                                    # detections). For each, fetches LVK's own
                                                    # real-time source classification (p_astro:
                                                    # BNS/NSBH/BBH/MassGap/Terrestrial) from GraceDB
                                                    # where public (O3 onward), falling back to a
                                                    # labeled mass-threshold heuristic for older
                                                    # (O1/O2) events with no public classifier.
                                                    # Takes a few minutes (per-event network calls).
python tests/test_simulation_batch.py              # Section 4.3-style campaign: 1000 synthetic
                                                    # BNS/BBH draws. For EVERY sample: real
                                                    # IMRPhenomD/LALSimulation waveform + mocked
                                                    # matched-filter noise + boundary router
                                                    # score. Writes a timestamped
                                                    # results/simulation_batch_<ts>.csv
                                                    # (includes peak/RMS strain columns).
python plot_results.py                             # reads the latest simulation_batch_<ts>.csv and
                                                    # writes a timestamped two-panel figure to
                                                    # results/router_performance_<ts>.png
```

Generated CSVs and figures are timestamped (e.g. `simulation_batch_20260716_185310.csv`,
`router_performance_20260716_185318.png`), so each run adds a new file
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
- [x] Large-scale validation against all ~90 confident confirmed events across
      GWTC-1/2.1/3, checked against LVK's own real-time source classification
      (p_astro, from GraceDB) where public, not a self-invented threshold:
      0 hard mismatches, with NSBH/MassGap events routed to AMBIGUOUS as
      designed -- `tests/test_large_scale_validation.py`
- [x] Distributed simulation campaign (Section 4.3): 1000 synthetic BNS/BBH
      draws, each with a real IMRPhenomD waveform + mocked matched-filter
      noise into the boundary router (99.6% exact match, 0 hard mismatches,
      100% safe-path rate) -- `tests/test_simulation_batch.py`, `plot_results.py`
- [ ] NSBH-specific routing refinement
- [ ] Local noise adaptation layer
- [ ] Dual-pathway global backbone
- [ ] Continuous training hub
