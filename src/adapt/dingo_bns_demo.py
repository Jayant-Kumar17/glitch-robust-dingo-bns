"""DINGO-BNS GW170817 demo asset discovery and frozen-model sampling helpers.

These utilities locate the official demo weights / strain packaging and run
baseline DINGO-BNS sampling. They do not implement network retraining.
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = (
    REPO_ROOT
    / "DINGO-BNS"
    / "dingo"
    / "binary-neutron-star-demo"
    / "GW170817"
)
DOWNLOADS_DIR = DEMO_ROOT / "downloads"
PIPE_DIR = DEMO_ROOT / "inference-dingo-pipe"
DEFAULT_BASELINE = DOWNLOADS_DIR / "dingo-bns-model_GW170817.pt"

DEFAULT_FIXED_CONTEXT = {
    "chirp_mass_proxy": 1.19786,
    "ra": 3.44616,
    "dec": -0.408084,
}

logger = logging.getLogger("adapt.dingo_bns_demo")


def select_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_bns_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _glob_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label} matching {pattern!r} in {directory}"
        )
    return matches[0]


def parse_ini_value(text: str, key: str) -> Optional[str]:
    pat = re.compile(rf"^{re.escape(key)}\s*=\s*(.+)$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def parse_fixed_context(ini_text: str) -> Dict[str, float]:
    raw = parse_ini_value(ini_text, "fixed-context-parameters")
    if raw is None:
        logger.warning("No fixed-context-parameters in ini; using demo defaults")
        return dict(DEFAULT_FIXED_CONTEXT)
    raw = raw.strip()
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        raw = raw[1:-1]
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            parsed = {}
            for part in inner.split(","):
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                parsed[k.strip().strip("'\"")] = float(v.strip())
        if not parsed:
            raise ValueError(f"Failed to parse fixed-context-parameters: {raw!r}")
    if not isinstance(parsed, dict):
        raise ValueError(f"fixed-context-parameters is not a dict: {parsed!r}")
    return {str(k): float(v) for k, v in parsed.items()}


def discover_assets(
    downloads: Path = DOWNLOADS_DIR,
    pipe_dir: Path = PIPE_DIR,
    baseline_ckpt: Optional[Path] = None,
    custom_ckpt: Optional[Path] = None,
) -> Dict[str, Any]:
    """Locate GW170817 demo strain, PSD, ini, event HDF5, and baseline weights.

    ``custom_ckpt`` is optional for the paper front-end (kept for API compatibility).
    """
    downloads = Path(downloads)
    pipe_dir = Path(pipe_dir)
    if not downloads.is_dir():
        raise FileNotFoundError(
            f"DINGO-BNS GW170817 downloads missing: {downloads}\n"
            "Install/clone the official DINGO-BNS demo and place weights/strain under "
            "DINGO-BNS/dingo/binary-neutron-star-demo/GW170817/downloads/"
        )
    if not pipe_dir.is_dir():
        raise FileNotFoundError(f"inference-dingo-pipe dir missing: {pipe_dir}")

    strain = {
        "H1": _glob_one(downloads, "H-H1_*.gwf", "H1 strain"),
        "L1": _glob_one(downloads, "L-L1_*.gwf", "L1 strain"),
        "V1": _glob_one(downloads, "V-V1_*.gwf", "V1 strain"),
    }
    psd = {
        "H1": _glob_one(downloads, "GWTC1_GW170817_PSD_H1.txt", "H1 PSD"),
        "L1": _glob_one(downloads, "GWTC1_GW170817_PSD_L1.txt", "L1 PSD"),
        "V1": _glob_one(downloads, "GWTC1_GW170817_PSD_V1.txt", "V1 PSD"),
    }
    baseline = Path(baseline_ckpt) if baseline_ckpt else DEFAULT_BASELINE
    if not baseline.is_file():
        baseline = _glob_one(downloads, "dingo-bns-model_*.pt", "baseline BNS checkpoint")

    custom: Optional[Path]
    if custom_ckpt is not None:
        custom = Path(custom_ckpt)
        if not custom.is_file():
            raise FileNotFoundError(f"custom_ckpt not found: {custom}")
    else:
        custom = None

    ini_path = pipe_dir / "GW170817.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Demo ini not found: {ini_path}")
    ini_text = ini_path.read_text()
    fixed = parse_fixed_context(ini_text)
    trigger_raw = parse_ini_value(ini_text, "trigger-time")
    trigger_time = float(trigger_raw) if trigger_raw else 1187008882.42

    event_candidates = sorted(
        (pipe_dir / "outdir" / "data").glob("*_generation_event_data.hdf5")
    )
    event_hdf5 = event_candidates[0] if event_candidates else None

    assets = {
        "downloads": downloads,
        "pipe_dir": pipe_dir,
        "strain_gwf": strain,
        "psd": psd,
        "baseline_ckpt": baseline,
        "custom_ckpt": custom,
        "ini_path": ini_path,
        "fixed_context": fixed,
        "trigger_time": trigger_time,
        "event_hdf5": event_hdf5,
        "channels": {
            "H1": "H1:LOSC-STRAIN",
            "L1": "L1:LOSC-STRAIN",
            "V1": "V1:LOSC-STRAIN",
        },
    }
    logger.info("Discovered H1 strain: %s", strain["H1"].name)
    logger.info("Discovered event HDF5: %s", event_hdf5)
    logger.info("Fixed context: %s", fixed)
    return assets


def load_event_dataset(assets: Dict[str, Any]):
    from dingo.gw.data.event_dataset import EventDataset

    hdf5 = assets.get("event_hdf5")
    if hdf5 is not None and Path(hdf5).is_file():
        logger.info("Loading packaged event from %s", hdf5)
        return EventDataset(file_name=str(hdf5))

    logger.info("No packaged event HDF5; building EventDataset from GWF+PSD")
    return build_event_dataset_from_gwf(assets)


def build_event_dataset_from_gwf(assets: Dict[str, Any]):
    """Fallback: condition GWF+PSD into an EventDataset matching the demo settings."""
    from dingo.gw.data.event_dataset import EventDataset
    from dingo.pipe.data_generation import DataGenerationInput

    complete = assets["pipe_dir"] / "outdir" / "GW170817_config_complete.ini"
    ini = complete if complete.is_file() else assets["ini_path"]
    logger.info("Building event via DataGenerationInput from %s", ini)

    sys_argv_backup = sys.argv[:]
    try:
        sys.argv = ["dingo_pipe_generation", str(ini)]
        from bilby_pipe.parser import create_parser

        try:
            from dingo.pipe.parser import create_parser as create_dingo_parser

            parser = create_dingo_parser()
        except Exception:
            parser = create_parser(top_level=False)
        args, _ = parser.parse_known_args([str(ini)])
        dgi = DataGenerationInput(args, [])
        dgi.save_hdf5 = lambda: None  # type: ignore[method-assign]
        if hasattr(dgi, "run"):
            dgi.run()
        elif hasattr(dgi, "create_data"):
            dgi.create_data()
        ifos = getattr(dgi, "interferometers", None)
        if ifos is None:
            raise RuntimeError("DataGenerationInput did not populate interferometers")
        detectors = list(assets["strain_gwf"].keys())
        waveform = {}
        asds = {}
        for det in detectors:
            ifo = ifos[det] if isinstance(ifos, dict) else ifos.get_ifo_by_name(det)
            waveform[det] = np.asarray(ifo.frequency_domain_strain, dtype=np.complex128)
            asds[det] = np.asarray(ifo.amplitude_spectral_density_array, dtype=np.float64)
        settings = {
            "time_event": float(assets["trigger_time"]),
            "time_buffer": 2.0,
            "detectors": detectors,
            "f_s": 4096.0,
            "T": 128.0,
            "f_min": 23.0,
            "f_max": 1535.3046875,
            "window_type": "tukey",
            "roll_off": 0.4,
        }
        return EventDataset(
            dictionary={
                "data": {"waveform": waveform, "asds": asds},
                "settings": settings,
            }
        )
    finally:
        sys.argv = sys_argv_backup


def run_baseline_sampling(
    baseline_ckpt: Path,
    event,
    fixed_context: Dict[str, float],
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> pd.DataFrame:
    from dingo.core.models import PosteriorModel
    from dingo.core.samplers import FixedInitSampler
    from dingo.gw.inference.gw_samplers import GWSamplerGNPE

    device_str = str(device)
    try:
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )
        if device_str == "mps":
            _ = torch.zeros(1, device=device)
    except Exception as exc:
        logger.warning("Baseline on %s failed (%s); falling back to CPU", device_str, exc)
        device_str = "cpu"
        pm = PosteriorModel(
            model_filename=str(baseline_ckpt),
            device=device_str,
            load_training_info=False,
        )

    init = FixedInitSampler(fixed_context, log_prob=0.0)
    sampler = GWSamplerGNPE(
        model=pm,
        init_sampler=init,
        num_iterations=1,
        fixed_context_parameters=fixed_context,
    )
    sampler.context = event.data
    sampler.event_metadata = event.settings
    sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
    df = sampler.samples.copy()
    logger.info(
        "Baseline sampling done: %d samples, columns=%s",
        len(df),
        list(df.columns),
    )
    return df


def load_event_td_crops(
    assets: Dict[str, Any],
    *,
    sample_rate: float,
    crop_seconds: float,
) -> Dict[str, np.ndarray]:
    """Load centered TD crops matching the shared STFT analysis window."""
    from gwpy.timeseries import TimeSeries

    trigger = float(assets["trigger_time"])
    half = 0.5 * float(crop_seconds)
    t0 = trigger - half
    t1 = trigger + half
    out: Dict[str, np.ndarray] = {}
    for det, path in assets["strain_gwf"].items():
        channel = assets["channels"][det]
        logger.info("Loading TD %s from %s [%s]", det, path.name, channel)
        ts = TimeSeries.read(str(path), channel=channel)
        ts = ts.crop(t0, t1)
        if float(ts.sample_rate.value) != float(sample_rate):
            ts = ts.resample(sample_rate)
        out[det] = np.asarray(ts.value, dtype=np.float64)
    return out


def build_event_spectrogram_stack(
    td_map: Dict[str, np.ndarray],
    detectors: Sequence[str],
    sample_rate: float,
    *,
    asds: Optional[Dict[str, np.ndarray]] = None,
    delta_f: Optional[float] = None,
    noise_std: Optional[float] = None,
    robust: bool = False,
    norm_stats: Optional[Dict[str, Any]] = None,
    n_time: Optional[int] = None,
    n_freq: Optional[int] = None,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build STFT context; ASD-whiten TD when ``asds`` are provided."""
    from adapt.stft_context import (
        build_csd_spectrogram_from_td,
        build_robust_spectrogram_from_td,
        whiten_td_map_with_asds,
    )

    td_use: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in td_map.items()}
    if asds is not None:
        if delta_f is None or noise_std is None:
            raise ValueError("delta_f and noise_std are required when whitening with asds")
        td_use = whiten_td_map_with_asds(
            td_use,
            asds,
            sample_rate=sample_rate,
            delta_f=float(delta_f),
            noise_std=float(noise_std),
            detectors=detectors,
        )
    if robust:
        kw: Dict[str, Any] = {
            "energy_detectors": tuple(detectors),
            "norm_stats": norm_stats,
        }
        if n_time is not None:
            kw["n_time"] = int(n_time)
        if n_freq is not None:
            kw["n_freq"] = int(n_freq)
        if n_fft is not None:
            kw["n_fft"] = int(n_fft)
        if win_length is not None:
            kw["win_length"] = int(win_length)
        if hop_length is not None:
            kw["hop_length"] = int(hop_length)
        return build_robust_spectrogram_from_td(td_use, sample_rate, **kw)
    return build_csd_spectrogram_from_td(
        td_use,
        sample_rate,
        energy_detectors=tuple(detectors),
    )


def design_asd_on_domain(det: str, domain) -> np.ndarray:
    from pycbc.psd import AdvVirgo, aLIGOZeroDetHighPower

    base = domain.base_domain if hasattr(domain, "base_domain") else domain
    n = len(base)
    delta_f = float(base.delta_f)
    f_min = float(base.f_min)
    if det == "V1":
        psd = AdvVirgo(n, delta_f, f_min)
    else:
        psd = aLIGOZeroDetHighPower(n, delta_f, f_min)
    asd = np.sqrt(np.asarray(psd, dtype=np.float64))
    asd = base.update_data(asd, low_value=1.0)
    if hasattr(domain, "decimate"):
        asd = domain.decimate(asd)
    return np.asarray(asd, dtype=np.float64)


def build_base_domain_injection(metadata: Dict[str, Any]):
    """Injection on uniform FrequencyDomain (IFFT-able) for synthetic BNS events."""
    from dingo.gw.domains import build_domain, build_domain_from_model_metadata
    from dingo.gw.gwutils import get_extrinsic_prior_dict
    from dingo.gw.injection import Injection
    from dingo.gw.prior import build_prior_with_defaults

    base = build_domain_from_model_metadata(metadata, base=True)
    wfg_domain = build_domain(metadata["dataset_settings"]["domain"])
    if hasattr(wfg_domain, "base_domain"):
        wfg_domain = wfg_domain.base_domain
    wfg_dict = dict(wfg_domain.domain_dict)
    sample_rate = float(metadata["train_settings"]["data"]["window"]["f_s"])
    wfg_dict["f_max"] = 0.5 * sample_rate
    wfg_domain = build_domain(wfg_dict)

    intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        metadata["train_settings"]["data"]["extrinsic_prior"]
    )
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
    injection = Injection(
        prior=prior,
        wfg_kwargs=metadata["dataset_settings"]["waveform_generator"],
        wfg_domain=wfg_domain,
        data_domain=base,
        ifo_list=metadata["train_settings"]["data"]["detectors"],
        t_ref=metadata["train_settings"]["data"]["ref_time"],
    )
    injection.asd = {
        det: design_asd_on_domain(det, base)
        for det in metadata["train_settings"]["data"]["detectors"]
    }
    return injection, base
