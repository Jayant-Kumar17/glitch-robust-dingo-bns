# Glitch mitigation for neural BNS parameter estimation with frozen DINGO-BNS

[![CITATION.cff](https://img.shields.io/badge/cite-CITATION.cff-blue)](CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the software accompanying a methods study on restoring
[DINGO-BNS](https://github.com/dingo-gw/dingo) posterior inference in the presence
of short-duration transient glitches, without retraining the neural posterior
estimator.

**Summary.** The frozen official DINGO-BNS network is robust to ordinary
glitches: **0% collapse below Omicron SNR 100**. Collapse reaches **60% for
Koi Fish at SNR ≥ 300**, a regime that contains **11% of confident O3 H1
Gravity Spy triggers**. A detect–gate–rebuild front-end restores clean-like
posteriors in **99% of 183 real-glitch cells** (including every collapsed
one) without retraining. A synthetic loudness sweep places 50% collapse at
**ρ_w ≈ 2.7×10³**. On a 240-cell GW170817 grid and a 160-cell synthetic BNS
panel, gated recovery is 86% and 92%. Ablations show the reconstruction
step is load-bearing: full FFT replacement recovers 0%. Archived result
directories are linked in [Results at a glance](#results-at-a-glance).

When the analysis segment is contaminated by a transient, official DINGO-BNS
posteriors can collapse (most conspicuously in luminosity distance). The
procedure implemented here is a preprocessing front-end applied to the event
data package prior to sampling:

1. detection of candidate glitch intervals;
2. Tukey gating in the time domain;
3. frequency-domain reconstruction via a matched-delta update;
4. retention of the original analysis amplitude spectral densities (ASDs);
5. sampling with the frozen official DINGO-BNS model.

Scientific scope and summary metrics are given in [`paper/SCOPE.md`](paper/SCOPE.md).
The precise relationship to stock DINGO-BNS inference is specified in
[`paper/METHOD.md`](paper/METHOD.md).

## Relation to official DINGO-BNS

The official DINGO-BNS network weights and architecture are left unchanged.
Reproduction therefore consists of:

1. obtaining the official GW170817 DINGO-BNS demo model and event packaging from
   upstream sources;
2. applying the front-end defined in this repository to the event data package
   (detection, gating, matched-delta reconstruction, original ASD retention);
3. sampling with the same frozen DINGO-BNS posterior model.

The sole trained component contributed here is a compact STFT glitch detector
(`checkpoints/glitch_detector_v1/`). Algorithmic detail and the reconstruction
API are documented in [`paper/METHOD.md`](paper/METHOD.md).

## External dependencies

The repository distributes source code, the glitch-detector checkpoint, and
summary result artefacts (JSON/CSV/PDF). Official DINGO-BNS weights and LIGO
strain frames are not redistributed. Regeneration of numerical results requires:

1. The DINGO-BNS GW170817 demonstration tree (or an equivalent layout), e.g.
   ```text
   DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/
     downloads/                 # dingo-bns-model_*.pt, H/L/V *.gwf, PSD files
     inference-dingo-pipe/      # GW170817.ini and outdir/data/*_event_data.hdf5
   ```
   Installation follows the upstream DINGO-BNS demo documentation
   ([dingo-gw/dingo](https://github.com/dingo-gw/dingo)).

2. The glitch-detector checkpoint included in this repository:
   ```text
   checkpoints/glitch_detector_v1/best_glitch_detector.pt
   ```
   Optional retraining is provided by `examples/train_glitch_detector.py`.
   File hashes for the paper runs are recorded under `results/stress_test_*/`.

3. A Python environment consistent with `environment.yml` / `requirements.txt`
   (pinned `numpy`, `scipy`, `torch`, `gwpy`, and `dingo-gw`).

The STFT detector weights are 566 KB and are tracked at
`checkpoints/glitch_detector_v1/best_glitch_detector.pt`. The same file is
attached to GitHub Release
[v1.1.1](https://github.com/Jayant-Kumar17/glitch-robust-dingo-bns/releases/tag/v1.1.1).

JSON/CSV summaries cited in the paper live under `results/` (see
[Results at a glance](#results-at-a-glance)). They are enough to regenerate
Tables 1–5 and Figures 1–3 and 6–9 via `paper/make_figures.py` without rerunning
the GPU/CPU inference grid. Figures 4–5 additionally need the official-control
HDF5 posterior dumps (gitignored; regenerate with `examples/official_control.py`).

## Installation

```bash
git clone https://github.com/Jayant-Kumar17/glitch-robust-dingo-bns.git
cd glitch-robust-dingo-bns
conda env create -f environment.yml   # or: conda activate adapt_env
conda activate adapt_env
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE
```

If DINGO is installed from a local source tree, prepend that path:
`export PYTHONPATH=/path/to/dingo:src:examples`.
Pinned versions of the critical stack (`numpy==1.26.4`, `scipy==1.13.1`,
`torch==2.13.0`, `gwpy==4.0.1`, `dingo-gw @ fede5c01`) are in
`pyproject.toml`, `requirements.txt`, and `environment.yml`.

## Quick-start / Reproducibility (Appendix C)

These are the commands from Appendix C of the paper. A user who has the
official DINGO-BNS demonstration assets and the released detector checkpoint
can clone, install, and reproduce Tables 1–5 with:

```bash
conda activate adapt_env
pip install -e .
export PYTHONPATH=src:examples
export KMP_DUPLICATE_LIB_OK=TRUE

python -u examples/official_control.py \
  --outdir results/dingo_official_control
python -u examples/honest_excision.py \
  --outdir results/excision_honest
python -u examples/stress_gw170817.py \
  --seed 0 --n-seeds-per-cell 5 \
  --num-samples 512 --outdir results/stress_test_excision_v1
python -u examples/stress_synthetic_bns.py \
  --outdir results/stress_test_synthetic_bns_v1
python -u examples/method_hardening.py \
  --num-samples 512 --outdir results/journal_method_hardening_v1
python -u examples/clean_gate_control.py \
  --outdir results/clean_gate_control_v1
python -u examples/collapse_threshold.py --seed 0 \
  --num-samples 512 --n-seeds 5 \
  --outdir results/collapse_threshold_v1
python -u examples/collapse_threshold.py --extend-low-rho \
  --outdir results/collapse_threshold_v1
python -u examples/stress_real_glitches.py --seed 0 \
  --num-samples 512 --outdir results/stress_real_glitches_v2
python -u examples/loudness_audit.py \
  --outdir results/loudness_audit_v1
```

Archived summaries under `results/*/` are the numerical source for Tables 1–5;
re-running the drivers is only required if you want to regenerate posterior
samples. Then:

```bash
python paper/make_figures.py --repo .            # fig1, fig4–fig9, S1–S4
python paper/make_figures_t5.py --repo .         # same suite (Appendix C alias)
python -u examples/paper_figures_2_3.py          # paper Figs 4–5 (filenames fig2, fig3)
```

All drivers resolve input and output paths from the repository root via
`pathlib.Path` (`Path(__file__).resolve().parents[1]`), so they run from any
working directory. Equivalent notes: [`paper/REPRODUCE.md`](paper/REPRODUCE.md),
[`examples/README.md`](examples/README.md).

## Paper-to-code mapping

| Paper item | Driver | Archive / figure script |
|---|---|---|
| Figure 1 (pipeline) | — | `paper/make_figures.py` → `fig1_*` |
| Figure 2 (threshold) | `examples/collapse_threshold.py` | `results/collapse_threshold_v1/` → `fig9_*` |
| Figure 3 (real glitches) | `examples/stress_real_glitches.py` | `results/stress_real_glitches_v2/` → `fig8_*` |
| Figure 4 (injection) | `examples/official_control.py` | `examples/paper_figures_2_3.py` → `fig2_*` |
| Figure 5 (posteriors) | `examples/official_control.py` | `examples/paper_figures_2_3.py` → `fig3_*` |
| Figure 6 (heatmap) | `examples/stress_gw170817.py` | `results/stress_test_excision_v1/` → `fig4_*` |
| Figure 7 (synthetic) | `examples/stress_synthetic_bns.py` | `results/stress_test_synthetic_bns_v1/` → `fig5_*` |
| Figure 8 (ablation) | `examples/method_hardening.py` | `results/journal_method_hardening_v1/` → `fig6_*` |
| Figure 9 (oracle) | `examples/method_hardening.py` | `results/journal_method_hardening_v1/` → `fig7_*` |
| Table 1 (real Gravity Spy) | `examples/stress_real_glitches.py` | `results/stress_real_glitches_v2/` |
| Table 2 (JS divergences) | `examples/official_control.py` | `results/dingo_official_control/` |
| Table 3 (GW170817 grid) | `examples/stress_gw170817.py` | `results/stress_test_excision_v1/` |
| Table 4 (synthetic BNS) | `examples/stress_synthetic_bns.py` | `results/stress_test_synthetic_bns_v1/` |
| Table 5 (ablation) | `examples/method_hardening.py` | `results/journal_method_hardening_v1/` |

## Glitch-detector training (optional)

The paper checkpoint is already present. To retrain:

```bash
python -u examples/train_glitch_detector.py \
  --epochs 30 --batch-size 16 --steps-per-epoch 100 \
  --outdir checkpoints/glitch_detector_v1
```

This writes `best_glitch_detector.pt` and `train_summary.json`.

### Restricted smoke runs (development / CI only)

The newer drivers (`clean_gate_control.py`, `stress_real_glitches.py`,
`paper_figures_2_3.py`) accept `--smoke` (2 cells, 64 samples); the original
drivers expose `--max-cells`, e.g.

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
python -u examples/clean_gate_control.py --smoke --outdir results/clean_gate_control_smoke
```

## Results at a glance

Headline numbers (abstract / release v1.1.1): **0% collapse** below Omicron SNR 100;
**60% collapse** for Koi Fish at SNR ≥ 300; **99% gated recovery** on 183 real
cells; **50% collapse** at ρ_w ≈ 2.7×10³; **11%** of confident O3 H1 Gravity Spy
triggers have SNR ≥ 300.

| Experiment | Directory | Cells | Principal finding |
|---|---|---|---|
| Official control | [`results/dingo_official_control/`](results/dingo_official_control/) | 1 | clean vs poisoned vs gated; max 1-D JS (clean vs gated) 0.012 nat, verdict A |
| Honest excision | [`results/excision_honest/`](results/excision_honest/) | 1 | matched-delta + original ASD recovers; Welch ASD or FFT replacement collapses |
| GW170817 stress grid | [`results/stress_test_excision_v1/`](results/stress_test_excision_v1/) | 240 | poisoned collapse 46.3%; gated recovery **85.8%** (held-out families 95.6%); clean false-positive rate 0/50 |
| Synthetic BNS panel | [`results/stress_test_synthetic_bns_v1/`](results/stress_test_synthetic_bns_v1/) | 160 | gated recovery **91.9%**; detector fire rate 100% |
| Method hardening (ablation) | [`results/journal_method_hardening_v1/`](results/journal_method_hardening_v1/) | 16 × 5 arms | `adapt_full` 81%; `gate_welch` 13%; `fft_replace` **0%**; front-end overhead ≈ 3 s |
| Clean-gate cost control | [`results/clean_gate_control_v1/`](results/clean_gate_control_v1/) | 25 | forced gate on clean data: recovery 92%, no collapse, mean \|Δ d_L\| 3.8 Mpc |
| Real Gravity Spy glitches (capped SNR) | [`results/stress_real_glitches_v1/`](results/stress_real_glitches_v1/) | 24 glitches × (native + 3 severities) + noise-only controls | native loudness: 0/20 collapse, 95 % pass-through; synthetic severity ladder: 86 % collapse, 72 % gated recovery |
| Real Gravity Spy glitches (wide SNR, native) | [`results/stress_real_glitches_v2/`](results/stress_real_glitches_v2/) | 183 native cells + noise-only | **0% collapse** below Omicron SNR 100; **60%** for Koi Fish ≥ 300; **99%** gated recovery; every collapse is a Koi Fish |
| Loudness audit | [`results/loudness_audit_v1/`](results/loudness_audit_v1/) | 240 + 1 + 24 | ρ_w of every injected glitch; **11%** of confident O3 H1 triggers have SNR ≥ 300 (`o3_snr_tail.json`) |
| Collapse threshold | [`results/collapse_threshold_v1/`](results/collapse_threshold_v1/) | 2 families × ρ_w ladder | 50% collapse at **ρ_w ≈ 2.7×10³**; gated recovery ≥ 97% at every loudness |

Large HDF5 posterior sample files are omitted from version control; JSON, CSV,
and PDF summaries are retained.

## Paper figures

All figures in `paper/figures/` regenerate from the archived `results/`:

```bash
python paper/make_figures.py --repo .            # fig1, fig4–fig9, S1–S4
python paper/make_figures_t5.py --repo .         # same suite (Appendix C alias)
python -u examples/paper_figures_2_3.py          # paper Figs 4–5 (filenames fig2, fig3; needs DINGO-BNS demo + sample HDF5s)
```

Print order in `paper/main.tex`: 1 pipeline, 2 threshold (`fig9`), 3 real glitches (`fig8`), 4 injection (`fig2`), 5 posteriors (`fig3`), 6 heatmap (`fig4`), 7 synthetic (`fig5`), 8 ablation (`fig6`), 9 oracle (`fig7`).

## Library interface

```python
from adapt.glitch_excision import rebuild_event_from_gated_td
from adapt.models import GlitchDetectorSTFT
from adapt.event_glitch_io import inject_h1_glitch_into_event
from adapt.dingo_bns_demo import discover_assets, run_baseline_sampling
```

## Citation

Manuscript in preparation for *Astronomy and Computing* (Elsevier).

Provisional title: *Transient-glitch resilience in neural gravitational-wave
parameter estimation without network retraining*.

Software citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub renders
a "Cite this repository" button from it). Tagged releases are listed under
[Releases](https://github.com/Jayant-Kumar17/glitch-robust-dingo-bns/releases);
steps for minting a Zenodo DOI are in [`docs/ZENODO.md`](docs/ZENODO.md).

Users of this software are requested to cite the published article when
available and to acknowledge [DINGO](https://github.com/dingo-gw/dingo) as the
underlying neural parameter-estimation framework. Real-glitch experiments use
the Gravity Spy classifications of Glanzer et al. (2023), Zenodo record
[5649212](https://doi.org/10.5281/zenodo.5649212).

## Licence

MIT Licence (`LICENSE`). Upstream DINGO software and LIGO/Virgo data products
remain subject to their respective terms of use.

## Contact

Jayant Kumar, Karachi Grammar School

j.kumar16224@kgs.edu.pk
