"""Integration: glitchy inject object + gate + orig ASD recovers d_L.

Skipped automatically when GW170817 / DINGO assets are unavailable.
Run explicitly::

    PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE \\
      pytest -q tests/test_honest_excision_dl.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "examples", REPO / "DINGO-BNS" / "dingo"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _assets_available() -> bool:
    try:
        from evaluate_gw170817_comparison import discover_assets

        assets = discover_assets(
            baseline_ckpt=None,
            custom_ckpt=REPO / "checkpoints" / "dingo_bns_custom_stft_best.pt",
        )
        return Path(assets["baseline_ckpt"]).is_file() and Path(assets["event_hdf5"]).is_file()
    except Exception:
        return False


@pytest.mark.skipif(not _assets_available(), reason="GW170817/DINGO assets missing")
def test_glitchy_package_gate_orig_asd_recovers_dl():
    """Honest path on the object ``inject_h1_glitch_into_event`` returns."""
    from adapt.glitch_excision import GateWindow, rebuild_event_from_gated_td
    from adapt.event_glitch_io import inject_h1_glitch_into_event
    from evaluate_gw170817_comparison import (
        discover_assets,
        load_event_dataset,
        run_baseline_sampling,
        select_device,
    )

    pe = None
    for cand in (
        REPO / "checkpoints" / "dingo_bns_custom_stft_best.pt",
        REPO / "checkpoints" / "glitch_robust" / "best_glitch_robust.pt",
    ):
        if cand.is_file():
            pe = cand
            break
    assets = discover_assets(baseline_ckpt=None, custom_ckpt=pe)
    event = load_event_dataset(assets)
    fixed = assets["fixed_context"]
    settings = dict(event.settings)
    device = select_device("cpu")

    glitch_data, _, gmeta = inject_h1_glitch_into_event(
        event, assets, f0=100.0, q=5.0, snr_amp_scale=8.0, t_rel=-1.0
    )
    duration = float(settings.get("T", 128.0))
    time_buffer = float(settings.get("time_buffer", 2.0))
    sample_rate = float(gmeta["sample_rate"])
    f_max = float(gmeta["f_max"])
    roll_off = float(gmeta["roll_off"])
    t_peak = float(gmeta["t_peak_in_segment"])
    gates = [GateWindow("H1", t_peak - 0.4, t_peak + 0.4)]
    asds = {d: np.asarray(event.data["asds"][d]).copy() for d in event.data["asds"]}

    excised = rebuild_event_from_gated_td(
        glitch_data,
        td_by_det=gmeta["td_full"],
        gates=gates,
        sample_rate=sample_rate,
        roll_off=roll_off,
        f_max=f_max,
        original_asds=asds,
    )
    assert not excised.noop
    assert float(excised.meta["residual_power_frac"]["H1"]) > 0.01
    np.testing.assert_allclose(excised.data["asds"]["H1"], asds["H1"])

    n = 128
    bs = 64
    clean_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        event,
        fixed,
        device=device,
        num_samples=n,
        batch_size=bs,
    )
    gated_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        SimpleNamespace(data=excised.data, settings=settings),
        fixed,
        device=device,
        num_samples=n,
        batch_size=bs,
    )
    poison_df = run_baseline_sampling(
        assets["baseline_ckpt"],
        SimpleNamespace(data=glitch_data, settings=settings),
        fixed,
        device=device,
        num_samples=n,
        batch_size=bs,
    )

    def _med(df):
        x = np.asarray(df["luminosity_distance"], dtype=np.float64)
        return float(np.median(x[np.isfinite(x)]))

    clean_med = _med(clean_df)
    gated_med = _med(gated_df)
    poison_med = _med(poison_df)
    # Poison stays on the lower rail; gated+orig ASD recovers toward clean.
    assert poison_med < 15.0
    assert gated_med > 20.0
    assert abs(gated_med - clean_med) < 25.0
