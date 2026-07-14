# ADAPT

**A**daptive **D**istributed **A**strophysics for **P**arameter Es**t**imation

A continuous hub-and-spoke inference framework for gravitational-wave parameter
estimation. See the preliminary report for the full design; this repository
tracks the implementation as it's built out over time.

## Setup

The project uses the `adapt_env` conda environment, which already has the
scientific dependencies installed (`gwpy`, `torch`, `torchdiffeq`, etc.).

```bash
conda activate adapt_env
pip install -e .
```

## Project layout

```
src/adapt/       # the adapt package
  router.py      # matched-filter routing heuristic (Section 3.1)
tests/           # standalone test/verification scripts
```

## Running tests

Test scripts are plain runnable Python scripts (no `pytest` required):

```bash
conda activate adapt_env
python tests/test_router.py
python tests/test_noise.py
```

## Progress

- [x] Matched-filter routing heuristic (BBH/BNS threshold) -- `src/adapt/router.py`
- [ ] NSBH mass-ratio routing refinement
- [ ] Local noise adaptation layer
- [ ] Dual-pathway global backbone
- [ ] Continuous training hub
