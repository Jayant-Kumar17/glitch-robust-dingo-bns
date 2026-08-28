"""Unit tests for DINGO-preserving glitch-robust conditioning and corrector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "DINGO-BNS" / "dingo"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_all_glitch_families_finite_and_reproducible():
    from adapt.glitch_augmentation import (
        HELD_IN_FAMILIES,
        HELD_OUT_FAMILIES,
        sample_glitch_spec,
        synthesize_glitch_td,
    )

    n = 4096
    fs = 4096.0
    for held_out, pool in ((False, HELD_IN_FAMILIES), (True, HELD_OUT_FAMILIES)):
        for fam in pool:
            rng_a = np.random.default_rng(123)
            rng_b = np.random.default_rng(123)
            # Force family by resampling until match (bounded).
            spec_a = None
            for _ in range(200):
                s = sample_glitch_spec(
                    rng_a, detectors=["H1", "L1", "V1"], held_out=held_out
                )
                if s.family == fam:
                    spec_a = s
                    break
            assert spec_a is not None, f"failed to sample {fam}"
            # Re-seed and force same family via reconstruct.
            from adapt.glitch_augmentation import GlitchSpec

            spec_b = GlitchSpec(
                family=spec_a.family,
                detectors=list(spec_a.detectors),
                t_rel=spec_a.t_rel,
                severity=spec_a.severity,
                params=dict(spec_a.params),
                asd_policy=spec_a.asd_policy,
                held_out=spec_a.held_out,
            )
            g1 = synthesize_glitch_td(n, fs, spec_a, rng_a, rms=1e-21)
            g2 = synthesize_glitch_td(n, fs, spec_b, rng_b, rms=1e-21)
            assert np.all(np.isfinite(g1))
            assert np.all(np.isfinite(g2))
            # Deterministic families (no RNG inside synth except broadband).
            if fam != "broadband_burst":
                np.testing.assert_allclose(g1, g2, rtol=0, atol=0)


def test_paired_fd_corruption_changes_target_detector():
    from adapt.glitch_augmentation import (
        corrupt_injection_fd_with_glitch,
        make_fixed_eval_glitch,
        td_to_fd,
    )

    n_freq = 2048
    duration = 8.0
    fs = 4096.0
    n_td = int(duration * fs)
    freqs = np.fft.rfftfreq(n_td, d=1.0 / fs)[:n_freq]
    # Synthetic FD: white noise-like.
    rng = np.random.default_rng(7)
    wave = {
        det: (rng.normal(size=n_freq) + 1j * rng.normal(size=n_freq)) * 1e-23
        for det in ("H1", "L1", "V1")
    }
    # Colored stationary ASD so Welch replacement is obviously different.
    asds = {
        det: (1e-23 * (1.0 + (freqs / 100.0) ** 2)).astype(np.float64) for det in wave
    }
    for det in asds:
        asds[det][freqs < 20] = 1.0
    inj = {
        "waveform": {k: v.copy() for k, v in wave.items()},
        "asds": {k: v.copy() for k, v in asds.items()},
        "parameters": {"geocent_time": 0.0, "H1_time": 0.0, "L1_time": 0.0, "V1_time": 0.0},
    }
    # Round-trip clean H1 (same Tukey/IFFT path without glitch) as control.
    fd_full = np.zeros(n_td // 2 + 1, dtype=np.complex128)
    fd_full[:n_freq] = wave["H1"]
    td_clean = np.fft.irfft(fd_full, n=n_td) * fs
    h1_rt = td_to_fd(td_clean, fs, n_freq=n_freq)

    spec = make_fixed_eval_glitch(severity=10.0, asd_policy="welch")
    glitched, td_crop, meta = corrupt_injection_fd_with_glitch(
        inj, sample_rate=fs, duration=duration, spec=spec, rng=rng
    )
    assert meta["family"] == "sine_gaussian"
    # Strain scales ≪ 1e-8: use rtol/atol=0 relative norms, not np.allclose defaults.
    h1_glitch_delta = np.linalg.norm(glitched["waveform"]["H1"] - h1_rt)
    h1_rt_norm = np.linalg.norm(h1_rt) + 1e-30
    assert h1_glitch_delta / h1_rt_norm > 1e-3
    # L1 has no glitch → original FD kept (no TD↔FD round-trip).
    np.testing.assert_allclose(glitched["waveform"]["L1"], wave["L1"], rtol=0, atol=0)
    np.testing.assert_allclose(glitched["waveform"]["V1"], wave["V1"], rtol=0, atol=0)
    assert td_crop["H1"].ndim == 1 and td_crop["H1"].size > 0
    assert np.all(np.isfinite(td_crop["H1"]))
    # Welch ASD policy replaces H1 above f_min; L1 keeps the colored curve.
    assert meta["asd_policy"] == "welch"
    band = freqs >= 50.0
    asd_rel_h1 = np.linalg.norm(
        glitched["asds"]["H1"][band] - asds["H1"][band]
    ) / (np.linalg.norm(asds["H1"][band]) + 1e-30)
    assert asd_rel_h1 > 1e-3
    asd_rel_l1 = np.linalg.norm(glitched["asds"]["L1"] - asds["L1"]) / (
        np.linalg.norm(asds["L1"]) + 1e-30
    )
    assert asd_rel_l1 < 1e-12


def test_robust_stft_shape_and_fixed_norm():
    from adapt.stft_context import (
        build_robust_spectrogram_from_td,
        calibrate_robust_norm_stats,
    )

    fs = 4096.0
    n = int(4.0 * fs)
    rng = np.random.default_rng(0)
    td = {det: rng.normal(size=n).astype(np.float64) for det in ("H1", "L1", "V1")}
    raw_t, raw_e = build_robust_spectrogram_from_td(td, fs, norm_stats=None)
    assert raw_t.shape == (6, 32, 128)
    assert raw_e.shape == (3,)
    stats = calibrate_robust_norm_stats([raw_t, raw_t], [raw_e, raw_e])
    norm_t, norm_e = build_robust_spectrogram_from_td(td, fs, norm_stats=stats)
    assert norm_t.shape == (6, 32, 128)
    assert np.all(np.isfinite(norm_t))
    assert np.all(np.isfinite(norm_e))
    # Same input + stats ⇒ identical tensor (eval matches train preprocess).
    again_t, again_e = build_robust_spectrogram_from_td(td, fs, norm_stats=stats)
    np.testing.assert_allclose(norm_t, again_t)
    np.testing.assert_allclose(norm_e, again_e)


class _FixedBase(torch.nn.Module):
    """Stand-in frozen DINGO embedding returning a constant 128-D context."""

    def __init__(self):
        super().__init__()
        self.register_buffer("fixed", torch.linspace(-1, 1, 128))

    def forward(self, strain, context_z):
        b = strain.shape[0]
        emb = self.fixed.unsqueeze(0).expand(b, -1)
        return torch.cat([emb, context_z], dim=-1)


def _make_embedding():
    from adapt.models import ContextAwareGlitchCorrector, GlitchRobustBNSEmbedding

    emb = GlitchRobustBNSEmbedding(_FixedBase(), corrector=ContextAwareGlitchCorrector())
    emb.eval()
    return emb


def test_corrector_near_identity_at_init():
    emb = _make_embedding()
    b = 2
    strain = torch.zeros(b, 3, 2, 16)
    spec = torch.randn(b, 6, 32, 128)
    # Clean-like normalized STFT energies sit near zero, below the gate center.
    energy = torch.zeros(b, 3)
    z = torch.zeros(b, 3)
    out = emb(strain, spec, energy, z)
    base_out = _FixedBase()(strain, z)
    # Gate≈0 and zero-init replacement ⇒ nearly identical to base DINGO.
    max_diff = (out - base_out).abs().max().item()
    assert max_diff < 0.05, f"init residual too large: {max_diff}"
    diag = emb.forward_with_diagnostics(strain, spec, energy, z)
    # Both branches have a sigmoid floor at clean energy (sigmoid(-4) learned,
    # sigmoid(-3) energy), so the mix leaks a few percent even on pristine input.
    # That is harmless while ctx_hat tracks DINGO(clean); the max_diff check above
    # is the binding constraint.
    assert diag["gate"].mean().item() < 0.10
    assert diag["energy_gate"].mean().item() < 0.05
    assert diag["gate_learned"].mean().item() < 0.05


def test_corrector_is_identity_at_init_even_when_gate_is_open():
    """Zero-init replacement ⇒ ctx_hat == base, so a wide-open gate is harmless."""
    emb = _make_embedding()
    b = 2
    strain = torch.zeros(b, 3, 2, 16)
    spec = torch.randn(b, 6, 32, 128)
    energy = torch.full((b, 3), 8.0)
    z = torch.zeros(b, 3)
    diag = emb.forward_with_diagnostics(strain, spec, energy, z)
    assert diag["gate"].mean().item() > 0.9
    torch.testing.assert_close(diag["ctx_hat"], diag["base_embed"])
    torch.testing.assert_close(diag["corrected_embed"], diag["base_embed"])


def test_h1_energy_spike_forces_gate_open():
    """The learned gate underfires on real events; H1 energy must open it anyway."""
    emb = _make_embedding()
    b = 2
    strain = torch.zeros(b, 3, 2, 16)
    spec = torch.randn(b, 6, 32, 128)
    z = torch.zeros(b, 3)
    clean_e = torch.zeros(b, 3)
    # Only H1 spikes, as for the GW170817 sine-Gaussian blip.
    spike_e = clean_e.clone()
    spike_e[:, 0] = 4.0

    diag_clean = emb.forward_with_diagnostics(strain, spec, clean_e, z)
    diag_spike = emb.forward_with_diagnostics(strain, spec, spike_e, z)
    # Learned branch is untrained and near zero in both cases.
    assert diag_spike["gate_learned"].mean().item() < 0.05
    assert diag_spike["energy_gate"].mean().item() > 0.9
    assert diag_spike["gate"].mean().item() > 0.9
    assert diag_clean["gate"].mean().item() < 0.10
    # The spike must dominate: gate opens by >10x relative to the clean floor.
    assert diag_spike["gate"].mean().item() > 10.0 * diag_clean["gate"].mean().item()


def test_replace_head_can_overwrite_poisoned_context():
    """A trained replacement must be able to leave the base context entirely."""
    emb = _make_embedding()
    corrector = emb.corrector
    # Simulate a trained replace head: non-zero output layer.
    torch.nn.init.normal_(corrector.replace_net[-1].weight, std=0.05)
    torch.nn.init.constant_(corrector.replace_net[-1].bias, 0.5)
    b = 2
    strain = torch.zeros(b, 3, 2, 16)
    spec = torch.randn(b, 6, 32, 128)
    energy = torch.full((b, 3), 4.0)
    z = torch.zeros(b, 3)
    diag = emb.forward_with_diagnostics(strain, spec, energy, z)
    # Gate is open, so the mix follows ctx_hat rather than the poisoned base.
    moved = (diag["corrected_embed"] - diag["base_embed"]).abs().mean().item()
    assert moved > 0.1, f"replacement had no effect: {moved}"
    torch.testing.assert_close(
        diag["corrected_embed"],
        (1.0 - diag["gate"]) * diag["base_embed"] + diag["gate"] * diag["ctx_hat"],
    )
    assert diag["replace_delta"].abs().max().item() <= corrector.max_delta + 1e-6


def test_delta_net_alias_shares_replace_net_weights():
    """Older checkpoint tooling keys off ``delta_net``; keep it bound to one module."""
    emb = _make_embedding()
    assert emb.corrector.delta_net is emb.corrector.replace_net


def test_make_fixed_eval_glitch_defaults():
    from adapt.glitch_augmentation import make_fixed_eval_glitch

    g = make_fixed_eval_glitch()
    assert g.detectors == ["H1"]
    assert g.family == "sine_gaussian"
    assert g.asd_policy == "welch"
    assert g.t_rel == -1.0


def test_stft_whitening_uses_clean_asd_and_sees_glitch():
    """Welch FD ASD must not be used for STFT whitening or the blip vanishes."""
    from adapt.glitch_augmentation import (
        make_fixed_eval_glitch,
        stft_whitening_asds,
        synthesize_glitch_td,
    )
    from adapt.stft_context import (
        SPECTROGRAM_ANALYSIS_SECONDS,
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )

    fs = 4096.0
    n = int(SPECTROGRAM_ANALYSIS_SECONDS * fs)
    rng = np.random.default_rng(11)
    # Quiet Gaussian noise + loud H1 SG blip (matches eval amplitude scale).
    td_clean = {det: rng.normal(size=n) * 1e-22 for det in ("H1", "L1", "V1")}
    spec = make_fixed_eval_glitch(severity=10.0, asd_policy="welch")
    g = synthesize_glitch_td(n, fs, spec, rng, rms=float(np.std(td_clean["H1"])))
    td_g = {k: v.copy() for k, v in td_clean.items()}
    td_g["H1"] = td_g["H1"] + g

    n_freq = n // 2 + 1
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    clean_asds = {
        det: (1e-23 * (1.0 + (freqs / 100.0) ** 2)).astype(np.float64)
        for det in td_clean
    }
    for det in clean_asds:
        clean_asds[det][freqs < 20] = 1.0
    # Contaminated Welch-like ASD: inflate bins near the glitch band.
    welch_asds = {k: v.copy() for k, v in clean_asds.items()}
    band = (freqs > 60) & (freqs < 140)
    welch_asds["H1"][band] *= 20.0

    stft_asds = stft_whitening_asds(clean_asds, welch_asds)
    np.testing.assert_array_equal(stft_asds["H1"], clean_asds["H1"])

    def _h1_stats(td_map, asd_map):
        wh = whiten_td_map_with_asds(
            td_map,
            asd_map,
            sample_rate=fs,
            delta_f=float(freqs[1] - freqs[0]),
            noise_std=1.0,
            detectors=("H1", "L1", "V1"),
        )
        _tens, eng = build_robust_spectrogram_from_td(wh, fs, norm_stats=None)
        return float(eng[0]), float(np.max(np.abs(wh["H1"])))

    e_g_clean, peak_clean_asd = _h1_stats(td_g, stft_asds)
    e_g_welch, peak_welch_asd = _h1_stats(td_g, welch_asds)
    # Welch ASD absorbs the blip: peak/energy drop vs clean-ASD whitening.
    assert peak_clean_asd > 3.0 * peak_welch_asd
    assert e_g_clean > e_g_welch + 0.3


def test_safe_whiten_masks_asd_zero_bins():
    from adapt.stft_context import safe_whiten_fd

    n = 128
    fd = np.ones(n, dtype=np.complex128) * 1e-24
    asd = np.ones(n, dtype=np.float64) * 1e-23
    asd[-1] = 0.0  # design-ASD null
    fd[-1] = 1e-30  # tiny leakage
    w = safe_whiten_fd(fd, asd, noise_std=1.0)
    assert np.isfinite(w).all()
    assert w[-1] == 0.0
    assert np.abs(w[0]) < 1e2


def test_inband_rms_less_than_broadband_seismic():
    from adapt.stft_context import inband_rms

    fs = 4096.0
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    # Strong sub-20 Hz + weak in-band tone.
    x = 1e-18 * np.sin(2 * np.pi * 5.0 * t) + 1e-22 * np.sin(2 * np.pi * 100.0 * t)
    bb = float(np.std(x))
    ib = inband_rms(x, fs, f_min=23.0, f_max=512.0)
    assert ib < 0.1 * bb


def test_non_glitched_ifo_fd_preserved_under_h1_corruption():
    """Regression: L1/V1 must not pick up H1 TD↔FD null-bin artifacts."""
    from adapt.glitch_augmentation import (
        corrupt_injection_fd_with_glitch,
        make_fixed_eval_glitch,
    )
    from adapt.stft_context import fd_waveform_to_td_crop

    n_freq = 2048
    duration = 8.0
    fs = 4096.0
    n_td = int(duration * fs)
    freqs = np.fft.rfftfreq(n_td, d=1.0 / fs)[:n_freq]
    rng = np.random.default_rng(3)
    wave = {
        det: (rng.normal(size=n_freq) + 1j * rng.normal(size=n_freq)) * 1e-23
        for det in ("H1", "L1", "V1")
    }
    asds = {det: np.ones(n_freq, dtype=np.float64) * 1e-23 for det in wave}
    for det in asds:
        asds[det][freqs < 20] = 1.0
        asds[det][-1] = 0.0  # ASD null at top bin
    inj = {
        "waveform": {k: v.copy() for k, v in wave.items()},
        "asds": {k: v.copy() for k, v in asds.items()},
        "parameters": {"geocent_time": 0.0, "H1_time": 0.0, "L1_time": 0.0, "V1_time": 0.0},
    }
    spec = make_fixed_eval_glitch(severity=8.0, asd_policy="stationary")
    glitched, _, _ = corrupt_injection_fd_with_glitch(
        inj, sample_rate=fs, duration=duration, spec=spec, rng=rng
    )
    td_l1_c = fd_waveform_to_td_crop(
        wave["L1"],
        sample_rate=fs,
        duration=duration,
        asd=asds["L1"],
        noise_std=1.0,
        whiten=True,
    )
    td_l1_g = fd_waveform_to_td_crop(
        glitched["waveform"]["L1"],
        sample_rate=fs,
        duration=duration,
        asd=asds["L1"],
        noise_std=1.0,
        whiten=True,
    )
    assert np.std(td_l1_g) < 10.0 * max(np.std(td_l1_c), 1e-30)
