# Checkpoints

## Included (paper front-end)

| File | Role |
|---|---|
| `glitch_detector_v1/best_glitch_detector.pt` | Trained STFT glitch detector used for detect-and-gate |
| `glitch_detector_v1/train_summary.json` | Training / calibration log (threshold, metrics) |

Retrain from scratch:

```bash
python -u examples/train_glitch_detector.py \
  --epochs 30 --batch-size 16 --steps-per-epoch 100 \
  --outdir checkpoints/glitch_detector_v1
```

## Not included (and not part of the paper claim)

- **Official DINGO-BNS weights** (~2 GB): obtain from the upstream DINGO-BNS
  GW170817 demo / Zenodo. The paper uses this network **frozen** (no NSF /
  embedding retrain).
- Older experimental “glitch-robust” fine-tuned DINGO heads (multi‑GB): **not**
  used in the manuscript; do not confuse them with the published method.
