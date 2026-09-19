# Reproduction commands

Commands identical to Appendix C of the paper and the root [`README.md`](../README.md).
Required external assets are the DINGO-BNS GW170817 demonstration packaging and
the glitch-detector checkpoint described in the README. Drivers resolve paths
from the repository root (`pathlib.Path`), independent of the working directory.

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

Archived summaries under `results/*/` are the numerical source for Tables 1–5.
Optional extra flags (`--device cpu`, `--batch-size`, `--hf-samples`,
`--wide-snr-panel`) are recorded in each run's `*config.json` / `REPRODUCE.md`.

## Restricted smoke configuration

Intended for software verification only; reported paper metrics use the full
suite above.

```bash
python -u examples/method_hardening.py \
  --max-cells 2 --num-samples 128 --batch-size 64 --device cpu \
  --outdir results/journal_method_hardening_v1_smoke
```

## Paper figures

Figures 1–3 and 6–9 regenerate from archived JSON/CSV under `results/` (no PE):

```bash
python paper/make_figures.py --repo . --out paper/figures
python paper/make_figures_t5.py --repo . --out paper/figures   # alias
```

Paper Figures 4–5 (filenames `fig2`, `fig3`) need the raw GW170817 strain and
the posterior-sample HDF5 dumps written by `examples/official_control.py`
(`poison_nn_samples.hdf5`, `gated_nn_samples.hdf5` / `gated_is_samples.hdf5` in
`results/dingo_official_control/`, plus the official importance-sampling
result of the DINGO-BNS demo). Once those exist:

```bash
python -u examples/paper_figures_2_3.py --outdir paper/figures --device cpu
# Fig 3 / paper Figure 5 only (no strain/detector assets needed):
python -u examples/paper_figures_2_3.py --only fig3 --outdir paper/figures
```

Writes `paper/figures/fig2_injection_gate.{pdf,png}` and
`paper/figures/fig3_posteriors.{pdf,png}`. The injected glitch and detector
output are regenerated deterministically with the archived control flags
(`--seed 0 --f0 100 --q 5 --t-rel -1.0 --snr-amp-scale 8.0`). RA/Dec are
fixed context in the GW170817 demo network and are therefore replaced in
paper Figure 5 by geocentric time and $\Lambda_1$.

## Success criteria (as coded)

- Poisoned collapse: luminosity-distance credible interval with upper edge
  below 15 Mpc or lower edge above 90 Mpc.
- Recovery relative to clean: median and interval-overlap criteria implemented
  in the respective example drivers.
