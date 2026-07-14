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
  physics.py     # chirp mass / total mass / mass ratio helpers
  router.py      # hierarchical matched-filter routing heuristic (Section 3.1)
  injection.py   # synthetic waveform injection into real noise (Section 4.3 preview)
tests/           # standalone test/verification scripts
```

## Running tests

Test scripts are plain runnable Python scripts (no `pytest` required):

```bash
conda activate adapt_env
python tests/test_router.py       # fast, no network needed
python tests/test_noise.py        # downloads ~10s of real noise from GWOSC
python tests/test_injection.py    # downloads a real 64s noise segment, then
                                   # injects synthetic BNS/BBH waveforms and
                                   # verifies the router; first run downloads
                                   # a large (~500MB) GWOSC file, cached after that
```

## Progress

- [x] Hierarchical matched-filter routing heuristic (chirp mass, total mass,
      mass ratio, spin confidence modifier) -- `src/adapt/router.py`
- [x] Synthetic injection pipeline for end-to-end router verification against
      real detector noise -- `src/adapt/injection.py`
- [ ] NSBH-specific routing refinement
- [ ] Local noise adaptation layer
- [ ] Dual-pathway global backbone
- [ ] Continuous training hub
