# Method specification relative to official DINGO-BNS

This document defines the inference procedure used in the accompanying study
and its precise relationship to stock DINGO-BNS sampling.

The official DINGO-BNS network parameters are not edited, fine-tuned, or
replaced. The checkpoint employed for posterior sampling is the upstream
GW170817 demonstration model. The intervention acts exclusively on the event
data package (`waveform` and `asds`) presented to that frozen sampler.

## Stock baseline

```text
event packaging (HDF5 / demo conditioning)
    → GWSamplerGNPE(official DINGO-BNS checkpoint)
    → posterior samples
```

Implementation: `adapt.dingo_bns_demo.run_baseline_sampling`.

## Front-end procedure

```text
contaminated event package
    → STFT glitch detector (checkpoint distributed with this repository)
    → Tukey gates on affected interferometers (default half-width 0.4 s)
    → matched-delta frequency-domain reconstruction of gated strain
    → restoration of the original analysis ASDs
      (Welch estimates formed on the gated segment are not retained)
    → GWSamplerGNPE(same official DINGO-BNS checkpoint)
    → posterior samples
```

### Correspondence to source modules

| Stage | Operation | Implementation |
|---|---|---|
| 1 | Locate official model and GW170817 packaging; load event dataset | `adapt.dingo_bns_demo.discover_assets`, `load_event_dataset` |
| 2 | Synthetic glitch injection (evaluation panels only) | `adapt.event_glitch_io.inject_h1_glitch_into_event` |
| 3 | Glitch detection (time-bin logits → gate windows) | `adapt.models.GlitchDetectorSTFT` and example gate utilities |
| 4 | Time-domain Tukey gating | `adapt.glitch_excision` |
| 5 | Frequency-domain rebuild with `mode="matched_delta"` on the contaminated package | `rebuild_event_from_gated_td` |
| 6 | ASD policy: supply `original_asds` from the analysis ASD | same call |
| 7 | Posterior sampling with the unchanged official model | `run_baseline_sampling` |

### Canonical reconstruction call

```python
from adapt.glitch_excision import rebuild_event_from_gated_td

excised = rebuild_event_from_gated_td(
    glitchy_event_data,
    gates=gates,
    td_map=td_full,
    sample_rate=4096.0,
    settings=event.settings,
    original_asds=analysis_asds,
    mode="matched_delta",
)
# Posterior sampling is then performed on excised.data with the frozen
# official DINGO-BNS checkpoint.
```

The combination `mode="matched_delta"` with `original_asds` constitutes the
published recipe. Full FFT replacement (`mode="replace"`) and retention of
Welch ASDs estimated on gated data are treated as negative controls.

### Ablation arms

| Arm | Configuration | Observed behaviour |
|---|---|---|
| `poison_welch` | Contaminated strain with Welch ASD; no gate | Posterior collapse |
| `glitch_orig_asd` | Contaminated strain with original ASD; no gate | Generally unrecovered |
| `gate_welch` | Gate and matched-delta rebuild; Welch ASD retained | Generally unrecovered |
| `fft_replace` | Gate with full FFT replacement and original ASD | Posterior collapse |
| `adapt_full` | Gate, matched-delta rebuild, and original ASD | Recovery consistent with clean reference |

These configurations are exercised by `examples/method_hardening.py`.

## Trained component

The only network trained for this study is the STFT glitch detector:

- checkpoint: `checkpoints/glitch_detector_v1/best_glitch_detector.pt`
- training entry point: `examples/train_glitch_detector.py`

No DINGO embedding network, neural spline flow, or alternative PE head was
trained for the reported results.

## Numerical reproduction

1. Install the upstream DINGO-BNS GW170817 demonstration assets (official
   checkpoint, strain frames, and event HDF5).
2. Employ the distributed detector checkpoint, or retrain it with the supplied
   script.
3. Execute the example drivers listed in the repository README with the stated
   flags.
4. Compare summary statistics against the archived files under `results/`.

Discrepancies should first be checked against (i) SHA-256 hashes of the
official and detector checkpoints recorded in `results/*/`, (ii) use of
matched-delta reconstruction with original ASDs rather than FFT replacement or
Welch ASDs on gated data.
