"""Per-IFO STFT glitch detector (time-bin segmentation).

Takes a whitened spectrogram ``(B, C, T, F)`` and predicts per-detector,
per-time-bin glitch logits ``(B, n_ifo, T)``. Designed to drive Tukey gating
around frozen DINGO — not to modify the NSF context.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Avoid importing adapt.stft_context here (circular via train_t1 → models).
_DEFAULT_N_TIME = 32
_DEFAULT_N_FREQ = 128


class GlitchDetectorSTFT(nn.Module):
    """Small Conv encoder → per-IFO time-bin logits.

    Architecture:
      stem Conv2d over (time, freq) → residual blocks with freq pooling only
      → collapse freq → Linear to n_ifo logits per time bin.

    At init the final bias is −4 so sigmoid≈0.018 (silent on clean).
    """

    def __init__(
        self,
        *,
        in_channels: int = 6,
        n_ifo: int = 3,
        n_time: int = _DEFAULT_N_TIME,
        n_freq: int = _DEFAULT_N_FREQ,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_ifo = int(n_ifo)
        self.n_time = int(n_time)
        self.n_freq = int(n_freq)
        c = int(base_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ELU(inplace=True),
        )
        # Pool frequency only so the time axis stays aligned with STFT bins.
        self.block1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ELU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ELU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ELU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ELU(inplace=True),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Conv1d(c * 4, self.n_ifo, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Return logits ``(B, n_ifo, n_time)``."""
        if spectrogram.ndim != 4:
            raise ValueError(
                f"spectrogram must be (B, C, T, F); got {tuple(spectrogram.shape)}"
            )
        b, c, t, f = spectrogram.shape
        if c != self.in_channels:
            raise ValueError(
                f"expected in_channels={self.in_channels}; got {c}"
            )
        h = self.stem(spectrogram)
        h = self.block1(h)
        h = self.block2(h)
        h = self.dropout(h)
        # Collapse frequency; keep / restore time axis to input length.
        h = h.mean(dim=-1)  # (B, C', T')
        if h.shape[-1] != t:
            h = F.interpolate(h, size=t, mode="linear", align_corners=False)
        logits = self.head(h)  # (B, n_ifo, T)
        return logits

    def predict_mask(
        self,
        spectrogram: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Boolean mask ``(B, n_ifo, T)`` from sigmoid(logits) > threshold."""
        probs = torch.sigmoid(self.forward(spectrogram))
        return probs > float(threshold)

    def predict_probs(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(spectrogram))


def segmentation_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """BCE-with-logits over ``(B, n_ifo, T)`` targets in {0,1}."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} != targets {tuple(targets.shape)}"
        )
    kw: Dict[str, torch.Tensor] = {}
    if pos_weight is not None:
        kw["pos_weight"] = torch.tensor(
            [float(pos_weight)], device=logits.device, dtype=logits.dtype
        )
    # BCEWithLogits pos_weight broadcasts over last dim; flatten for simplicity.
    return F.binary_cross_entropy_with_logits(
        logits.reshape(-1),
        targets.reshape(-1).to(dtype=logits.dtype),
        pos_weight=(
            torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
            if pos_weight is not None
            else None
        ),
    )


def detector_fires(
    probs: torch.Tensor,
    *,
    threshold: float = 0.5,
    min_bins: int = 1,
) -> torch.Tensor:
    """Per-sample, per-IFO fire flag ``(B, n_ifo)``."""
    hits = (probs > float(threshold)).to(dtype=torch.int64).sum(dim=-1)
    return hits >= int(min_bins)
