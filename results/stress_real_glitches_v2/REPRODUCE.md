# Reproduce the real Gravity Spy glitch panel

Event: **GW170817** (frozen official DINGO-BNS). Glitches: real O3 H1 excerpts
from Gravity Spy (Zenodo 5649212, Glanzer et al. 2023), selected by
`adapt.gravity_spy_io` (H1, ml_confidence >= 0.95, 8 <= snr <= 30, duration <= 3 s;
Blip 10 / Scattered_Light 5 / Whistle 5 / Koi_Fish 5). Cached selection:
`data/gravity_spy/selected_h1_o3.csv`.

## Environment

```bash
conda activate adapt_env
cd /Users/jayantkumar/Desktop/ADAPT-Project/.worktrees/task5
export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
```

## Run

```bash
python examples/stress_real_glitches.py --seed 0 \
  --num-samples 512 --outdir results/stress_real_glitches_v2
# smoke: python examples/stress_real_glitches.py --smoke --outdir results/stress_real_glitches_smoke
```

Raw Zenodo tables download to `data/gravity_spy/raw/` (git-ignored) on first
run; H1 strain is fetched from GWOSC with `gwpy` (8 s per glitch).

## Excerpt recipe

8 s H1 open strain (GWOSC, chunk-level HTTP range read of the bulk HDF5) centred on the Gravity Spy event_time; whitened with a Welch PSD from its own off-window part (unit-variance background, 20-1500 Hz); cut to event_time +/- max(duration, 0.5 s) with Tukey(alpha=0.1) taper; STFT-thresholded (8x per-frequency median off-window power) within the trigger duration (+0.15 s) to a glitch-only estimate; coloured with the line-smoothed (2 Hz running median) GW170817 H1 analysis ASD (zero-padded, re-tapered) and injected additively into the H1 analysis segment. Scaling: 'native' = same whitened energy (SNR^2) as observed in O3; 'ladder' = raw in-band RMS of the injected window equals severity x inband_rms(clean H1), the synthetic-grid convention. Working in the whitened domain is required because Gravity Spy glitches of Omicron SNR 8-30 are not measurable as a raw in-band RMS excess over the 20-40 Hz noise wall.

## Outputs

- `results.csv`, `summary.json`, `failures.csv`, `glitch_catalogue.csv`,
  `fetch_failures.csv`, `clean_reference.json`, `stress_config.json`
- `paper/figures/fig8_real_glitches.pdf`

## Success criteria

- Poison collapsed: `d_L` hi < 15 or lo > 90
- Gated recovers: med in [20, 50], CI overlaps clean, |med - clean_med| <= 10 Mpc
