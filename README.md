# Glitch-robust DINGO-BNS inference

Companion code for restoring [DINGO-BNS](https://github.com/dingo-gw/dingo)
posteriors under short-duration transient glitches **without retraining** the
neural network.

Frozen official DINGO-BNS can collapse when a glitch contaminates the analysis
segment. This repository implements a small front-end:

1. detect candidate glitch intervals,
2. apply Tukey gates in the time domain,
3. rebuild frequency-domain strain with **matched-delta** reconstruction,
4. **keep the original analysis ASD**,
5. sample with the frozen DINGO-BNS model.

Full claim, method, and canonical result table: [`paper/SCOPE.md`](paper/SCOPE.md).

## Installation

Create / activate a scientific Python environment (the development work used a
conda env named `adapt_env`), then from the repository root:

```bash
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

This package provides the front-end utilities. You still need the upstream
**DINGO-BNS** demo weights and GW170817 event packaging (not redistributed here).
Detector-gated runs expect:

```text
checkpoints/glitch_detector_v1/best_glitch_detector.pt
```

## Usage

Runnable studies live under [`examples/`](examples/). See
[`examples/README.md`](examples/README.md) and [`paper/REPRODUCE.md`](paper/REPRODUCE.md).

Quick smoke (ablation subset):

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

Library entry points (after `pip install -e .`):

```python
from adapt.glitch_excision import rebuild_event_from_gated_td
from adapt.models import GlitchDetectorSTFT
from adapt.event_glitch_io import inject_h1_glitch_into_event
```

## Results

Canonical paper artifacts:

| Study | Directory | Headline |
|---|---|---|
| Official control | `results/dingo_official_control/` | clean / poison / gated |
| Honest excision | `results/excision_honest/` | matched-delta + orig ASD recovers |
| GW170817 stress | `results/stress_test_excision_v1/` | gated recovery ≈ 85.8% |
| Synthetic BNS | `results/stress_test_synthetic_bns_v1/` | gated recovery ≈ 91.9% |
| Method hardening | `results/journal_method_hardening_v1/` | ablation / oracle / runtime |

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).

Working title: *Transient-glitch resilience in neural gravitational-wave
parameter estimation without network retraining*.

If you use this code, please cite the published paper when available and
acknowledge [DINGO](https://github.com/dingo-gw/dingo) for the underlying
neural PE model.

## License

MIT — see [`LICENSE`](LICENSE). Upstream DINGO and LIGO/Virgo data products
remain under their own terms.

## Contact

Jayant Kumar — Karachi Grammar School
