"""Rolling observatory node stream for ADAPT telemetry.

Generates fixed-length sequential strain chunks (default 32 s) from a
detector's raw array or on-disk file. When real data is exhausted, chunks
continue seamlessly as Advanced LIGO design-sensitivity colored Gaussian
noise (``pycbc.psd.aLIGOZeroDetHighPower``).

Does not modify ``pipeline_manager`` or ``noise_analytics``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np

PathLike = Union[str, Path]
ArrayOrPath = Union[np.ndarray, PathLike]


def _synthesize_aligo_chunk(
    n_samples: int,
    sample_rate: float,
    seed: Optional[int] = None,
    f_lower: float = 20.0,
) -> np.ndarray:
    """Generate one chunk of aLIGOZeroDetHighPower colored Gaussian noise."""
    from pycbc.noise import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower

    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / (n_samples * delta_t)
    psd = aLIGOZeroDetHighPower(n_samples // 2 + 1, delta_f, f_lower)
    ts = noise_from_psd(n_samples, delta_t, psd, seed=seed)
    return np.asarray(ts.numpy(), dtype=np.float64)


def _load_strain_data(data: ArrayOrPath) -> np.ndarray:
    """Load a 1D strain array from memory or a supported file path."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise FileNotFoundError(f"strain data file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix in {".hdf5", ".h5"}:
            from gwpy.timeseries import TimeSeries

            ts = TimeSeries.read(str(path), format="hdf5.gwosc")
            arr = np.asarray(ts.value, dtype=np.float64)
        else:
            raise ValueError(
                f"unsupported strain file type '{suffix}' "
                "(expected .npy, .hdf5, or .h5)"
            )
    else:
        arr = data

    strain = np.asarray(arr, dtype=np.float64).ravel()
    if strain.ndim != 1:
        raise ValueError("strain data must be a 1D array")
    return strain


class ObservatoryNode:
    """Continuous rolling source of fixed-length detector strain chunks."""

    def __init__(
        self,
        detector: str,
        data: ArrayOrPath,
        *,
        sample_rate: float = 4096.0,
        stride_seconds: float = 32.0,
        seed: Optional[int] = None,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")

        self.detector = str(detector).strip().upper()
        self.sample_rate = float(sample_rate)
        self.stride_seconds = float(stride_seconds)
        self._chunk_samples = int(round(self.stride_seconds * self.sample_rate))
        if self._chunk_samples < 2:
            raise ValueError("stride_seconds * sample_rate must yield at least 2 samples")

        self._raw = _load_strain_data(data)
        self._ptr = 0
        self._base_seed = seed
        self._sim_call_index = 0
        self.is_simulated = False

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def n_raw_samples(self) -> int:
        return int(len(self._raw))

    def _next_sim_seed(self) -> Optional[int]:
        if self._base_seed is None:
            return None
        seed = int(self._base_seed) + int(self._sim_call_index)
        self._sim_call_index += 1
        return seed

    def _colored(self, n_samples: int) -> np.ndarray:
        return _synthesize_aligo_chunk(
            n_samples, self.sample_rate, seed=self._next_sim_seed()
        )

    def get_next_chunk(self) -> Dict[str, Any]:
        """Return the next ``stride_seconds`` of strain (real, hybrid, or sim).

        Advances an internal sample pointer. Never raises ``IndexError``; when
        real data runs out, subsequent samples are aLIGO colored noise.
        """
        n = self._chunk_samples
        n_raw = len(self._raw)
        start_sample = self._ptr

        if self._ptr >= n_raw:
            chunk = self._colored(n)
            is_simulated = True
        else:
            remaining = n_raw - self._ptr
            if remaining >= n:
                chunk = self._raw[self._ptr : self._ptr + n].copy()
                is_simulated = False
            else:
                # Seamless hybrid: leftover real samples + colored pad.
                head = self._raw[self._ptr :].copy()
                pad = self._colored(n - remaining)
                chunk = np.concatenate([head, pad])
                is_simulated = True

        self._ptr += n
        self.is_simulated = is_simulated
        return {
            "detector": self.detector,
            "strain": chunk,
            "is_simulated": is_simulated,
            "start_sample": start_sample,
            "stride_seconds": self.stride_seconds,
            "sample_rate": self.sample_rate,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def __next__(self) -> Dict[str, Any]:
        return self.get_next_chunk()
