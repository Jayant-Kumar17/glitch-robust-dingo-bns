"""DINGO-T1 Transformer wrapper for ADAPT Neural Posterior Estimation.

This module defines a journal-grade conditioning network that maps
frequency-domain strain together with a real-time global noise context
tensor into a fixed-length feature vector for a normalizing-flow (NF)
engine. The intended probabilistic target is the conditional density

    q(θ | d, S_n)

where ``d`` is the frequency-domain data and ``S_n`` is the active noise
context (PSD / rich noise profile). The Transformer operates on a sequence
of per-frequency tokens, so variable detector and frequency configurations
are handled natively by choosing ``n_freq`` and ``context_dim`` at
construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DingoT1Network(nn.Module):
    """Transformer NPE conditioner: frequency strain + noise context → features.

    Each frequency bin is tokenized as the concatenation of real strain,
    imaginary strain, and an aligned noise-power token derived from the
    pipeline ``noise_context_tensor``. A learnable summary token aggregates
    the sequence via self-attention; a linear head projects that state to
    ``num_params`` conditioning features for a downstream normalizing flow
    approximating ``q(θ | d, S_n)``.

    The sequence layout adapts to different detector/frequency setups by
    configuring ``n_freq`` (bins) and ``context_dim`` (global noise vector
    length, e.g. 640 for the default H1+L1 hub profile).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_params: int = 15,
        n_freq: int = 128,
        context_dim: int = 640,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ):
        """Build tokenizer, noise aligner, Transformer encoder, and projection head.

        Parameters
        ----------
        embed_dim :
            Token embedding dimension (must be divisible by ``num_heads``).
        num_heads :
            Multi-head attention heads in each encoder layer.
        num_layers :
            Number of ``TransformerEncoderLayer`` stacks.
        num_params :
            Output conditioning feature dimension for the NF engine.
        n_freq :
            Expected number of frequency bins ``F`` in ``strain_frequencies``.
        context_dim :
            Expected length ``C`` of ``noise_context_tensor`` (default 640 =
            320 × 2 for the H1+L1 ``GlobalNoiseHub`` rich-profile length).
        dropout :
            Dropout used inside ``TransformerEncoderLayer``.
        dim_feedforward :
            Transformer FFN hidden width. Defaults to ``4 * embed_dim``;
            industrial DINGO backbones typically use ``2 * embed_dim`` (2048).
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if n_freq < 1 or context_dim < 1 or num_params < 1:
            raise ValueError("n_freq, context_dim, and num_params must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        ffn_dim = int(dim_feedforward) if dim_feedforward is not None else 4 * int(embed_dim)
        if ffn_dim < 1:
            raise ValueError("dim_feedforward must be >= 1")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_params = int(num_params)
        self.n_freq = int(n_freq)
        self.context_dim = int(context_dim)
        self.dim_feedforward = ffn_dim

        # Tokenizer: [Re(d_f), Im(d_f), S_n(f)] -> embed_dim
        self.tokenizer = nn.Linear(3, self.embed_dim)

        # Map global noise context (B, C) -> per-bin noise power (B, F)
        self.noise_to_freq = nn.Linear(self.context_dim, self.n_freq)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Learnable summary / CLS token prepended to the frequency sequence.
        self.summary_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Project summary state to NF conditioning features.
        self.proj_head = nn.Linear(self.embed_dim, self.num_params)

    def forward(
        self,
        strain_frequencies: torch.Tensor,
        noise_context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``(d, S_n)`` into conditioning features for ``q(θ | d, S_n)``.

        Parameters
        ----------
        strain_frequencies :
            Frequency-domain strain with shape ``(B, F, 2)`` where the last
            axis is ``[real, imag]``.
        noise_context_tensor :
            Batched global noise context with shape ``(B, C)``, typically the
            ``context_tensor`` from ``ADAPTPipelineManager.prepare_dingo_context``
            (possibly squeezed from ``(B, C)`` after adapter batching).

        Returns
        -------
        torch.Tensor
            Conditioning features of shape ``(B, num_params)`` for a
            normalizing-flow posterior engine.

        Notes
        -----
        Per-bin tokens let the Transformer handle variable detector and
        frequency configurations by changing ``n_freq`` / ``context_dim`` at
        construction rather than hard-coding a single IFO layout.
        """
        if not isinstance(strain_frequencies, torch.Tensor):
            raise TypeError(
                f"strain_frequencies must be a torch.Tensor, got {type(strain_frequencies)}"
            )
        if not isinstance(noise_context_tensor, torch.Tensor):
            raise TypeError(
                f"noise_context_tensor must be a torch.Tensor, got {type(noise_context_tensor)}"
            )

        summary_state = self._encode_summary(strain_frequencies, noise_context_tensor)
        features = self.proj_head(summary_state)  # (B, num_params)
        batch_size = summary_state.shape[0]
        if features.shape != (batch_size, self.num_params):
            raise RuntimeError(
                f"output shape {tuple(features.shape)} != "
                f"({batch_size}, {self.num_params})"
            )
        return features

    def _encode_summary(
        self,
        strain_frequencies: torch.Tensor,
        noise_context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenizer → Transformer → summary token state ``(B, embed_dim)``."""
        if noise_context_tensor.ndim == 3 and noise_context_tensor.shape[1] == 1:
            noise_context_tensor = noise_context_tensor.squeeze(1)

        if strain_frequencies.ndim != 3 or strain_frequencies.shape[-1] != 2:
            raise ValueError(
                f"strain_frequencies must have shape (B, F, 2); "
                f"got {tuple(strain_frequencies.shape)}"
            )
        batch_size, n_freq, _ = strain_frequencies.shape
        if n_freq != self.n_freq:
            raise ValueError(
                f"strain_frequencies F={n_freq} != network n_freq={self.n_freq}"
            )
        if noise_context_tensor.ndim != 2:
            raise ValueError(
                f"noise_context_tensor must have shape (B, C); "
                f"got {tuple(noise_context_tensor.shape)}"
            )
        if noise_context_tensor.shape[0] != batch_size:
            raise ValueError(
                f"batch mismatch: strain B={batch_size}, "
                f"noise B={noise_context_tensor.shape[0]}"
            )
        if noise_context_tensor.shape[1] != self.context_dim:
            raise ValueError(
                f"noise_context_tensor C={noise_context_tensor.shape[1]} "
                f"!= network context_dim={self.context_dim}"
            )

        sn = self.noise_to_freq(noise_context_tensor).unsqueeze(-1)  # (B, F, 1)
        tokens_in = torch.cat([strain_frequencies, sn], dim=-1)  # (B, F, 3)
        emb = self.tokenizer(tokens_in)  # (B, F, embed_dim)
        summary = self.summary_token.expand(batch_size, -1, -1)
        seq = torch.cat([summary, emb], dim=1)
        encoded = self.encoder(seq)
        summary_state = encoded[:, 0, :]
        if summary_state.shape != (batch_size, self.embed_dim):
            raise RuntimeError(
                f"summary_state shape {tuple(summary_state.shape)} != "
                f"({batch_size}, {self.embed_dim})"
            )
        return summary_state

    def encode_flow_context(
        self,
        strain_frequencies: torch.Tensor,
        noise_context_tensor: torch.Tensor,
        final_proj: nn.Module | None = None,
    ) -> torch.Tensor:
        """Return NSF conditioning context ``(B, 128)`` (or ``embed_dim`` if no proj)."""
        summary_state = self._encode_summary(strain_frequencies, noise_context_tensor)
        if final_proj is None:
            return summary_state
        return final_proj(summary_state)
