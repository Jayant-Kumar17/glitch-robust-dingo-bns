# Checkpoints

## Distributed with this study

| File | Description |
|---|---|
| `glitch_detector_v1/best_glitch_detector.pt` | Trained STFT glitch detector used for detect-and-gate inference |
| `glitch_detector_v1/train_summary.json` | Training and calibration summary (threshold and validation metrics) |

Optional retraining:

```bash
python -u examples/train_glitch_detector.py \
  --epochs 30 --batch-size 16 --steps-per-epoch 100 \
  --outdir checkpoints/glitch_detector_v1
```

## Not distributed

- Official DINGO-BNS weights (approximately 2 GB) are obtained from the upstream
  GW170817 demonstration release. In this study the network is used with frozen
  parameters (no NSF or embedding retraining).
- Earlier exploratory fine-tuned DINGO variants are not part of the reported
  method and are not required for reproduction.
