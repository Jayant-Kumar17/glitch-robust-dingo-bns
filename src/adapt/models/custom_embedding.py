"""Custom 2-D spectrogram embedding for DINGO-BNS adaptation.

Fuses an H1–L1 3-channel mag/coherence STFT image with frequency-domain Re/Im
strain and log-energy into a fixed 128-D context vector matching the BNS NSF
embedding output dimension.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class SpectrogramEncoder2D(nn.Module):
    """Light Conv2d stack (``cnn_base``): ``(N, C, T, F)`` → ``(N, out_features)``."""

    def __init__(self, out_features: int = 64, in_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_features)
        self.out_features = int(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (N, {self.in_channels}, T, F); got {tuple(x.shape)}"
            )
        h = self.net(x).flatten(1)  # (N, 64)
        return self.proj(h)


class ResidualConvBlock(nn.Module):
    """Basic residual block: Conv-BN-ELU-Conv-BN + skip, then ELU.

    The second BN is zero-initialized so each block starts as an identity
    (stable early training with deep residual stacks).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ELU(inplace=True)
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class SpectrogramResNet2D(nn.Module):
    """Deeper residual 2-D encoder (``resnet_deep``).

    Stem → residual stages with configurable channel widths → AdaptiveAvgPool → Linear.
    """

    def __init__(
        self,
        out_features: int = 64,
        channels: Sequence[int] = (64, 128, 256, 512),
        in_channels: int = 3,
    ):
        super().__init__()
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("encoder channels must be non-empty")
        self.channels = chans
        self.out_features = int(out_features)
        self.in_channels = int(in_channels)

        layers: list[nn.Module] = [
            nn.Conv2d(self.in_channels, chans[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chans[0]),
            nn.ELU(inplace=True),
            ResidualConvBlock(chans[0]),
        ]
        for i in range(len(chans) - 1):
            c_in, c_out = chans[i], chans[i + 1]
            layers.extend(
                [
                    nn.Conv2d(
                        c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(c_out),
                    nn.ELU(inplace=True),
                    ResidualConvBlock(c_out),
                ]
            )
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(chans[-1], self.out_features)
        # Small projection so spectrogram branch starts near-zero contribution.
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (N, {self.in_channels}, T, F); got {tuple(x.shape)}"
            )
        h = self.net(x).flatten(1)
        return self.proj(h)


def build_spectrogram_encoder(
    encoder_type: str = "resnet_deep",
    *,
    out_features: int = 64,
    encoder_channels: Sequence[int] | None = None,
    in_channels: int = 3,
) -> nn.Module:
    """Factory for 2-D spectrogram encoders."""
    et = str(encoder_type).lower().strip()
    if et in ("cnn_base", "cnn", "base"):
        return SpectrogramEncoder2D(
            out_features=out_features, in_channels=in_channels
        )
    if et in ("resnet_deep", "resnet", "deep"):
        chans = encoder_channels if encoder_channels is not None else (64, 128, 256, 512)
        return SpectrogramResNet2D(
            out_features=out_features,
            channels=chans,
            in_channels=in_channels,
        )
    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}; expected 'cnn_base' or 'resnet_deep'"
    )


class Strain1DEncoder(nn.Module):
    """Encode Re/Im FD strain ``(B, n_ifo, 2, n_freq)`` → ``(B, out_features)``."""

    def __init__(
        self,
        n_freq: int = 3324,
        n_ifo: int = 3,
        out_features: int = 128,
    ):
        super().__init__()
        self.n_freq = int(n_freq)
        self.n_ifo = int(n_ifo)
        self.out_features = int(out_features)
        in_features = self.n_ifo * 2 * self.n_freq
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.out_features),
            nn.ELU(inplace=True),
        )

    def forward(self, strain_re_im: torch.Tensor) -> torch.Tensor:
        if strain_re_im.ndim != 4:
            raise ValueError(
                f"strain_re_im must be 4-D (B, n_ifo, 2, n_freq); "
                f"got {tuple(strain_re_im.shape)}"
            )
        b, n_ifo, n_ch, n_freq = strain_re_im.shape
        if n_ifo != self.n_ifo or n_ch != 2 or n_freq != self.n_freq:
            raise ValueError(
                f"expected (B, {self.n_ifo}, 2, {self.n_freq}); "
                f"got {tuple(strain_re_im.shape)}"
            )
        return self.net(strain_re_im.reshape(b, -1))


class Spectrogram2DNet(nn.Module):
    """Fuse 3-ch mag/coherence STFT + FD strain + log-energy → 128-D NSF context.

    Parameters
    ----------
    strain :
        ``(B, 3, 3, 3324)`` official DINGO package (ch0/1 = Re/Im, ch2 ignored)
        or ``(B, 3, 2, 3324)`` Re/Im only.
    spectrogram :
        ``(B, 3, n_time, n_spec_freq)`` H1–L1 mag/coherence image
        (default ``n_time=5``, ``n_spec_freq=128``).
    log_energy :
        ``(B, n_ifo)`` pre-normalization STFT ``log(sum |S|^2 + eps)``.

    Returns
    -------
    context : Tensor
        Shape ``(B, 128)``.
    """

    ENERGY_DIM = 16
    ENERGY_EPS = 1e-8
    SPEC_CHANNELS = 3

    def __init__(
        self,
        n_ifo: int = 3,
        n_freq: int = 3324,
        n_time: int = 5,
        n_spec_freq: int = 128,
        context_dim: int = 128,
        strain_dim: int = 128,
        spec_ifo_dim: int = 64,
        encoder_type: str = "resnet_deep",
        encoder_channels: Sequence[int] | None = None,
        in_channels: int = 3,
    ):
        super().__init__()
        self.n_ifo = int(n_ifo)
        self.n_freq = int(n_freq)
        self.n_time = int(n_time)
        self.n_spec_freq = int(n_spec_freq)
        self.context_dim = int(context_dim)
        self.strain_dim = int(strain_dim)
        self.spec_ifo_dim = int(spec_ifo_dim)
        self.in_channels = int(in_channels)
        self.encoder_type = str(encoder_type)
        self.encoder_channels = (
            list(encoder_channels)
            if encoder_channels is not None
            else [64, 128, 256, 512]
        )
        self.energy_conditioning = True
        self.spectrogram_layout = "hl_coh_3ch"

        self.strain_encoder = Strain1DEncoder(
            n_freq=self.n_freq,
            n_ifo=self.n_ifo,
            out_features=self.strain_dim,
        )
        self.spec_encoder = build_spectrogram_encoder(
            self.encoder_type,
            out_features=self.spec_ifo_dim,
            encoder_channels=self.encoder_channels,
            in_channels=self.in_channels,
        )
        self.spec_fuse = nn.Sequential(
            nn.Linear(self.spec_ifo_dim, 128),
            nn.ELU(inplace=True),
        )
        # Per-IFO STFT log-energy + per-IFO FD strain log-energy → 16-D.
        self.energy_proj = nn.Sequential(
            nn.Linear(2 * self.n_ifo, self.ENERGY_DIM),
            nn.ELU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(self.strain_dim + 128 + self.ENERGY_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Linear(256, self.context_dim),
        )
        # Near-zero context at init keeps the frozen NSF in a finite NLL regime
        # until the embedding starts learning useful features.
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _fd_log_energy(strain_re_im: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-IFO ``log(sum(re^2+im^2) + eps)`` over frequency → ``(B, n_ifo)``."""
        power = (strain_re_im**2).sum(dim=(-2, -1))  # (B, n_ifo)
        return torch.log(power + eps)

    def forward(
        self,
        strain: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
    ) -> torch.Tensor:
        if strain.ndim != 4 or strain.shape[1] != self.n_ifo:
            raise ValueError(
                f"strain must be (B, {self.n_ifo}, 2|3, {self.n_freq}); "
                f"got {tuple(strain.shape)}"
            )
        if strain.shape[2] not in (2, 3) or strain.shape[3] != self.n_freq:
            raise ValueError(
                f"strain must be (B, {self.n_ifo}, 2|3, {self.n_freq}); "
                f"got {tuple(strain.shape)}"
            )
        expected_spec = (self.in_channels, self.n_time, self.n_spec_freq)
        if spectrogram.ndim != 4 or spectrogram.shape[1:] != expected_spec:
            raise ValueError(
                f"spectrogram must be (B, {self.in_channels}, {self.n_time}, "
                f"{self.n_spec_freq}); got {tuple(spectrogram.shape)}"
            )
        if strain.shape[0] != spectrogram.shape[0]:
            raise ValueError(
                f"batch mismatch: strain B={strain.shape[0]}, "
                f"spectrogram B={spectrogram.shape[0]}"
            )

        b = strain.shape[0]
        if log_energy.ndim == 1:
            log_energy = log_energy.unsqueeze(0)
        if log_energy.shape != (b, self.n_ifo):
            raise ValueError(
                f"log_energy must be (B, {self.n_ifo}); got {tuple(log_energy.shape)}"
            )
        stft_e = torch.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)

        strain_re_im = strain[:, :, :2, :]
        strain_feat = self.strain_encoder(strain_re_im)
        fd_e = self._fd_log_energy(strain_re_im, eps=self.ENERGY_EPS)
        energy_feat = self.energy_proj(torch.cat([stft_e, fd_e], dim=-1))

        spec_feat = self.spec_fuse(self.spec_encoder(spectrogram))

        context = self.out(torch.cat([strain_feat, spec_feat, energy_feat], dim=-1))
        if context.shape != (b, self.context_dim):
            raise RuntimeError(
                f"output shape {tuple(context.shape)} != ({b}, {self.context_dim})"
            )
        return context


class SpectrogramBNSEmbedding(nn.Module):
    """Wrap ``Spectrogram2DNet`` and concatenate GNPE context → 131-D NSF input.

    Matches DINGO-BNS ``added_context=True``: ``cat(embed(data), z)`` with
    ``z = [ra, dec, chirp_mass_proxy]`` (already standardized).

    Forward signature (NSF ``*x`` order)::

        strain, spectrogram, log_energy, context_z
    """

    def __init__(
        self,
        spect_net: Spectrogram2DNet | None = None,
        context_param_dim: int = 3,
    ):
        super().__init__()
        self.spect_net = spect_net if spect_net is not None else Spectrogram2DNet()
        self.context_param_dim = int(context_param_dim)
        self.output_dim = int(self.spect_net.context_dim + self.context_param_dim)

    def forward(
        self,
        strain: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
        context_z: torch.Tensor,
    ) -> torch.Tensor:
        h = self.spect_net(strain, spectrogram, log_energy)  # (B, 128)
        if context_z.ndim == 1:
            context_z = context_z.unsqueeze(0)
        if context_z.shape[0] != h.shape[0]:
            raise ValueError(
                f"batch mismatch: embed B={h.shape[0]}, context_z B={context_z.shape[0]}"
            )
        if context_z.shape[-1] != self.context_param_dim:
            raise ValueError(
                f"context_z last dim {context_z.shape[-1]} != {self.context_param_dim}"
            )
        out = torch.cat([h, context_z], dim=-1)
        if out.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"output dim {out.shape[-1]} != expected {self.output_dim}"
            )
        return out


class SpectrogramResidualHead(nn.Module):
    """STFT + log-energy → small 128-D residual (no strain re-encoder).

    Used on top of a frozen DINGO RB embedding. Re-encoding FD strain in the
    residual lets the head shove context off the NSF manifold and explode NLL;
    this head only looks at the spectrogram path.
    """

    ENERGY_DIM = 16
    ENERGY_EPS = 1e-8

    def __init__(
        self,
        *,
        n_ifo: int = 3,
        n_time: int = 5,
        n_spec_freq: int = 128,
        context_dim: int = 128,
        spec_ifo_dim: int = 64,
        encoder_type: str = "resnet_deep",
        encoder_channels: Sequence[int] | None = None,
        in_channels: int = 3,
        init_scale: float = 0.05,
        max_scale: float = 0.25,
        max_delta: float = 0.5,
    ):
        super().__init__()
        import math

        self.n_ifo = int(n_ifo)
        self.n_time = int(n_time)
        self.n_spec_freq = int(n_spec_freq)
        self.context_dim = int(context_dim)
        self.in_channels = int(in_channels)
        self.max_scale = float(max_scale)
        self.max_delta = float(max_delta)
        self.encoder_type = str(encoder_type)
        self.encoder_channels = (
            list(encoder_channels)
            if encoder_channels is not None
            else [64, 128, 256, 512]
        )

        self.spec_encoder = build_spectrogram_encoder(
            self.encoder_type,
            out_features=int(spec_ifo_dim),
            encoder_channels=self.encoder_channels,
            in_channels=self.in_channels,
        )
        self.spec_fuse = nn.Sequential(
            nn.Linear(int(spec_ifo_dim), 128),
            nn.ELU(inplace=True),
        )
        self.energy_proj = nn.Sequential(
            nn.Linear(self.n_ifo, self.ENERGY_DIM),
            nn.ELU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(128 + self.ENERGY_DIM, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, self.context_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        # Learnable scale, starts small so residual cannot dominate early.
        self.log_scale = nn.Parameter(
            torch.tensor(float(math.log(max(float(init_scale), 1e-6))))
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
    ) -> torch.Tensor:
        expected_spec = (self.in_channels, self.n_time, self.n_spec_freq)
        if spectrogram.ndim != 4 or spectrogram.shape[1:] != expected_spec:
            raise ValueError(
                f"spectrogram must be (B, {self.in_channels}, {self.n_time}, "
                f"{self.n_spec_freq}); got {tuple(spectrogram.shape)}"
            )
        b = spectrogram.shape[0]
        if log_energy.ndim == 1:
            log_energy = log_energy.unsqueeze(0)
        if log_energy.shape != (b, self.n_ifo):
            raise ValueError(
                f"log_energy must be (B, {self.n_ifo}); got {tuple(log_energy.shape)}"
            )
        stft_e = torch.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)
        spec_feat = self.spec_fuse(self.spec_encoder(spectrogram))
        energy_feat = self.energy_proj(stft_e)
        delta = self.out(torch.cat([spec_feat, energy_feat], dim=-1))
        scale = torch.exp(self.log_scale).clamp(max=self.max_scale)
        delta = (scale * delta).clamp(-self.max_delta, self.max_delta)
        return delta


class ResidualSpectrogramBNSEmbedding(nn.Module):
    """Frozen DINGO RB embedding + small STFT-only residual → 131-D NSF context.

    Keeps pretrained ``base_embedding(strain, z)`` and adds::

        out[:128] = base[:128] + residual_head(spectrogram, log_energy)

    The residual head does **not** re-encode FD strain (that path already lives
    in the frozen DINGO embedding). Zero-init + small scale keep init ≈ baseline.
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        spect_net: nn.Module | None = None,
        context_param_dim: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.spect_net = (
            spect_net if spect_net is not None else SpectrogramResidualHead()
        )
        self.context_param_dim = int(context_param_dim)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(self.embed_dim + self.context_param_dim)
        for p in self.base_embedding.parameters():
            p.requires_grad = False

    def forward(
        self,
        strain: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
        context_z: torch.Tensor,
    ) -> torch.Tensor:
        if context_z.ndim == 1:
            context_z = context_z.unsqueeze(0)
        base = self.base_embedding(strain, context_z)
        if base.ndim != 2 or base.shape[-1] != self.output_dim:
            raise ValueError(
                f"base embedding must be (B, {self.output_dim}); got {tuple(base.shape)}"
            )
        if context_z.shape[0] != base.shape[0]:
            raise ValueError(
                f"batch mismatch: base B={base.shape[0]}, context_z B={context_z.shape[0]}"
            )
        # STFT-only residual (ignore strain in the head).
        if isinstance(self.spect_net, SpectrogramResidualHead):
            delta = self.spect_net(spectrogram, log_energy)
        else:
            # Legacy Spectrogram2DNet residual path.
            delta = self.spect_net(strain, spectrogram, log_energy)
        if delta.shape != (base.shape[0], self.embed_dim):
            raise ValueError(
                f"spect_net must return (B, {self.embed_dim}); got {tuple(delta.shape)}"
            )
        out = base.clone()
        out[:, : self.embed_dim] = out[:, : self.embed_dim] + delta
        return out

    def trainable_residual_parameters(self):
        """STFT residual parameters (excludes frozen DINGO base)."""
        yield from self.spect_net.parameters()


class ContextAwareGlitchCorrector(nn.Module):
    """Replace-mix corrector: blend frozen DINGO context with STFT-reconstructed context.

    On clean inputs the gate stays near 0 ⇒ output ≈ frozen DINGO.
    On glitchy inputs an **energy-forced** gate opens and the NSF sees
    ``ctx_hat`` trained to match ``DINGO(clean)``, not a tiny residual on the
    poisoned FD embedding (which cannot undo Welch-ASD amplitude collapse).

    Identity-safe at init: replace head zero-init, learned-gate bias −4.
    """

    ENERGY_DIM = 16

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        n_ifo: int = 3,
        n_time: int = 32,
        n_spec_freq: int = 128,
        in_channels: int = 6,
        spec_ifo_dim: int = 64,
        encoder_type: str = "resnet_deep",
        encoder_channels: Sequence[int] | None = None,
        max_delta: float = 8.0,
        energy_gate_center: float = 1.5,
        energy_gate_scale: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_ifo = int(n_ifo)
        self.n_time = int(n_time)
        self.n_spec_freq = int(n_spec_freq)
        self.in_channels = int(in_channels)
        self.max_delta = float(max_delta)
        self.encoder_type = str(encoder_type)
        self.encoder_channels = (
            list(encoder_channels)
            if encoder_channels is not None
            else [64, 128, 256, 512]
        )
        # Monotonic H1-energy gate (buffers — not learned away by accident).
        self.register_buffer(
            "energy_gate_center",
            torch.tensor(float(energy_gate_center)),
            persistent=True,
        )
        self.register_buffer(
            "energy_gate_scale",
            torch.tensor(float(energy_gate_scale)),
            persistent=True,
        )

        self.spec_encoder = build_spectrogram_encoder(
            self.encoder_type,
            out_features=int(spec_ifo_dim),
            encoder_channels=self.encoder_channels,
            in_channels=self.in_channels,
        )
        self.spec_fuse = nn.Sequential(
            nn.Linear(int(spec_ifo_dim), 128),
            nn.ELU(inplace=True),
        )
        self.energy_proj = nn.Sequential(
            nn.Linear(self.n_ifo, self.ENERGY_DIM),
            nn.ELU(inplace=True),
        )
        # Replace head may ignore poisoned base via learned skip (zero at init).
        fuse_in = self.embed_dim + 128 + self.ENERGY_DIM
        self.replace_net = nn.Sequential(
            nn.Linear(fuse_in, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.embed_dim),
        )
        nn.init.zeros_(self.replace_net[-1].weight)
        nn.init.zeros_(self.replace_net[-1].bias)
        # Alias for older checkpoint key patterns / tools.
        self.delta_net = self.replace_net
        self.gate_net = nn.Sequential(
            nn.Linear(128 + self.ENERGY_DIM, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -4.0)

    def forward(
        self,
        base_ctx: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(delta, gate, corrected_ctx)`` with gate in ``(0, 1)``."""
        diag = self.forward_detailed(base_ctx, spectrogram, log_energy)
        return diag["delta"], diag["gate"], diag["corrected"]

    def forward_detailed(
        self,
        base_ctx: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if base_ctx.ndim != 2 or base_ctx.shape[-1] != self.embed_dim:
            raise ValueError(
                f"base_ctx must be (B, {self.embed_dim}); got {tuple(base_ctx.shape)}"
            )
        expected = (self.in_channels, self.n_time, self.n_spec_freq)
        if spectrogram.ndim != 4 or spectrogram.shape[1:] != expected:
            raise ValueError(
                f"spectrogram must be (B, {self.in_channels}, {self.n_time}, "
                f"{self.n_spec_freq}); got {tuple(spectrogram.shape)}"
            )
        b = base_ctx.shape[0]
        if log_energy.ndim == 1:
            log_energy = log_energy.unsqueeze(0)
        if log_energy.shape != (b, self.n_ifo):
            raise ValueError(
                f"log_energy must be (B, {self.n_ifo}); got {tuple(log_energy.shape)}"
            )
        stft_e = torch.nan_to_num(log_energy, nan=0.0, posinf=0.0, neginf=0.0)
        spec_feat = self.spec_fuse(self.spec_encoder(spectrogram))
        energy_feat = self.energy_proj(stft_e)
        aux = torch.cat([spec_feat, energy_feat], dim=-1)

        # ctx_hat ≈ base at init (zero last layer); can fully replace when trained.
        replace_delta = self.replace_net(torch.cat([base_ctx, aux], dim=-1))
        replace_delta = replace_delta.clamp(-self.max_delta, self.max_delta)
        ctx_hat = base_ctx + replace_delta

        gate_logit = self.gate_net(aux)
        gate_learned = torch.sigmoid(gate_logit)
        # H1 energy excess forces the gate open on real Welch+blip events even if
        # the learned gate underfires (domain gap vs synthetic training).
        h1 = stft_e[:, :1]
        energy_gate = torch.sigmoid(
            self.energy_gate_scale * (h1 - self.energy_gate_center)
        )
        gate = 1.0 - (1.0 - gate_learned) * (1.0 - energy_gate)

        corrected = (1.0 - gate) * base_ctx + gate * ctx_hat
        delta = corrected - base_ctx
        return {
            "delta": delta,
            "gate": gate,
            "gate_logit": gate_logit,
            "gate_learned": gate_learned,
            "energy_gate": energy_gate,
            "ctx_hat": ctx_hat,
            "corrected": corrected,
            "replace_delta": replace_delta,
        }


class GlitchRobustBNSEmbedding(nn.Module):
    """Frozen DINGO RB + STFT replace-mix corrector → 131-D NSF input.

    Forward::

        strain, spectrogram, log_energy, context_z

    At init, gate≈0 and replace head is zero ⇒ output matches baseline DINGO.
    """

    def __init__(
        self,
        base_embedding: nn.Module,
        corrector: ContextAwareGlitchCorrector | None = None,
        context_param_dim: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.corrector = corrector if corrector is not None else ContextAwareGlitchCorrector()
        self.context_param_dim = int(context_param_dim)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(self.embed_dim + self.context_param_dim)
        for p in self.base_embedding.parameters():
            p.requires_grad = False
        # Alias for checkpoint helpers that look for spect_net.
        self.spect_net = self.corrector

    def forward(
        self,
        strain: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
        context_z: torch.Tensor,
    ) -> torch.Tensor:
        if context_z.ndim == 1:
            context_z = context_z.unsqueeze(0)
        base = self.base_embedding(strain, context_z)
        if base.ndim != 2 or base.shape[-1] != self.output_dim:
            raise ValueError(
                f"base embedding must be (B, {self.output_dim}); got {tuple(base.shape)}"
            )
        base_ctx = base[:, : self.embed_dim]
        _delta, _gate, corrected = self.corrector(base_ctx, spectrogram, log_energy)
        out = base.clone()
        out[:, : self.embed_dim] = corrected
        return out

    def forward_with_diagnostics(
        self,
        strain: torch.Tensor,
        spectrogram: torch.Tensor,
        log_energy: torch.Tensor,
        context_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return context plus replace/gate diagnostics for distillation losses."""
        if context_z.ndim == 1:
            context_z = context_z.unsqueeze(0)
        base = self.base_embedding(strain, context_z)
        base_ctx = base[:, : self.embed_dim]
        detail = self.corrector.forward_detailed(base_ctx, spectrogram, log_energy)
        out = base.clone()
        out[:, : self.embed_dim] = detail["corrected"]
        return {
            "context": out,
            "base_context": base,
            "base_embed": base_ctx,
            "delta": detail["delta"],
            "gate": detail["gate"],
            "gate_logit": detail["gate_logit"],
            "gate_learned": detail["gate_learned"],
            "energy_gate": detail["energy_gate"],
            "ctx_hat": detail["ctx_hat"],
            "replace_delta": detail["replace_delta"],
            "corrected_embed": detail["corrected"],
        }

    def trainable_residual_parameters(self):
        yield from self.corrector.parameters()
