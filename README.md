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
  router.py        # hierarchical matched-filter routing heuristic (Section 3.1)
  injection.py      # synthetic waveform injection into real noise (Section 4.3 preview)
  gwosc_events.py   # live GWOSC lookups: published event parameters + real strain files
tests/             # standalone test/verification scripts
```

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
```

Note on GWOSC downloads: the continuous archive (`test_noise.py`,
`test_injection.py`) always serves ~4096-second (~500MB) files regardless
of the window requested. The event API (`test_known_answer_validation.py`,
`test_real_event_full_validation.py`) instead serves small, pre-cut ~32s
segments for confirmed events, which is much faster -- prefer it when
possible.

## Progress

- [x] Hierarchical matched-filter routing heuristic (chirp mass, total mass,
      mass ratio, spin confidence modifier) -- `src/adapt/router.py`
- [x] Synthetic injection pipeline for end-to-end router verification against
      real detector noise -- `src/adapt/injection.py`
- [x] Known-answer validation: synthetic full-metadata self-consistency checks,
      plus router decisions checked against real confirmed events (live
      published parameters and real raw strain from GWOSC) -- `src/adapt/gwosc_events.py`,
      `tests/test_known_answer_validation.py`, `tests/test_real_event_full_validation.py`
- [ ] NSBH-specific routing refinement
- [ ] Local noise adaptation layer
- [ ] Dual-pathway global backbone
- [ ] Continuous training hub
