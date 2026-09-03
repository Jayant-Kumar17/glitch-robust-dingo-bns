# Exact method: what changes relative to official DINGO-BNS

**You do not edit, fine-tune, or replace the official DINGO-BNS network.**  
The `.pt` weights stay bit-identical to the upstream GW170817 demo model.

What changes is the **event data package** (`waveform` + `asds`) that you pass into
frozen DINGO sampling.

## Baseline (stock DINGO)

```text
event HDF5 / demo packaging  →  GWSamplerGNPE(official .pt)  →  posterior
```

Implemented by `adapt.dingo_bns_demo.run_baseline_sampling`.

## Paper front-end (what we did)

```text
glitchy event package
    → STFT glitch detector (our small network; weights in this repo)
    → Tukey gates (±0.4 s default) on contaminated IFO(s)
    → matched-delta FD rebuild of gated strain
    → restore original analysis ASDs (do NOT keep Welch from the gated segment)
    → same frozen GWSamplerGNPE(official .pt)
    → posterior
```

### Step-by-step mapping to code

| Step | Exact operation | Code |
|---|---|---|
| 1. Load official model + GW170817 package | discover demo assets; load event HDF5 | `adapt.dingo_bns_demo.discover_assets`, `load_event_dataset` |
| 2. (Eval only) inject glitch | add synthetic glitch FD + often Welch ASD | `adapt.event_glitch_io.inject_h1_glitch_into_event` |
| 3. Detect | STFT time-bin logits → gate windows | `adapt.models.GlitchDetectorSTFT` + gate helpers in examples |
| 4. Gate | Tukey window in TD around each gate | `adapt.glitch_excision` |
| 5. Rebuild FD | **`mode="matched_delta"`** on the **glitchy** package | `rebuild_event_from_gated_td(...)` |
| 6. ASD policy | pass **`original_asds=...`** from the clean/analysis ASD | same call |
| 7. Sample | unchanged official DINGO sampler | `run_baseline_sampling(baseline_ckpt, excised_event, ...)` |

### The one call that is the “modification”

```python
from adapt.glitch_excision import rebuild_event_from_gated_td

excised = rebuild_event_from_gated_td(
    glitchy_event_data,          # poisoned package
    gates=gates,                 # from detector or oracle
    td_map=td_full,              # full analysis-segment TD
    sample_rate=4096.0,
    settings=event.settings,
    original_asds=clean_asds,    # critical: keep analysis ASD
    mode="matched_delta",        # critical: not "replace"
)
# then sample with frozen official DINGO on excised.data
```

### What fails (ablation controls — do not use as the method)

| Arm | What you change | Typical outcome |
|---|---|---|
| `poison_welch` | glitch FD + Welch ASD, no gate | posterior collapses |
| `glitch_orig_asd` | glitch FD + original ASD, no gate | usually still bad |
| `gate_welch` | gate + matched-delta but **Welch ASD kept** | usually still bad |
| `fft_replace` | gate + **full FFT replace** + original ASD | collapses (known-bad) |
| `adapt_full` | gate + **matched-delta** + **original ASD** | recovers (paper recipe) |

These arms are run in `examples/method_hardening.py`.

## What we trained

Only the **glitch detector**:

- weights: `checkpoints/glitch_detector_v1/best_glitch_detector.pt`
- trainer: `examples/train_glitch_detector.py`

We did **not** train a new DINGO embedding / NSF / “glitch-robust PE head” for
the published results.

## How to get the same results

1. Install upstream DINGO-BNS GW170817 demo (official `.pt` + strain + event HDF5).
2. Use the committed detector checkpoint (or retrain with the shipped script).
3. Run the five `examples/*.py` commands in the root README with the listed flags.
4. Compare JSON/CSV summaries to `results/*/`.

If your numbers disagree, first check: (a) official `.pt` SHA matches configs under
`results/`, (b) detector SHA matches, (c) you used `matched_delta` + original ASD,
not FFT-replace / Welch.
