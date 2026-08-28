"""DINGO conditioning adapter: NumPy global noise profiles → PyTorch tensors.

Translates ADAPT's concatenated 1D global noise profile vectors into
device-aware, contiguous PyTorch tensors suitable for Neural Posterior
Estimation (NPE) network conditioning layers (e.g. DINGO).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class DingoConditioningAdapter:
    """ML translator from ADAPT global noise profiles to DINGO NPE contexts."""

    _PRECISION_MAP = {
        "float32": "float32",
        "fp32": "float32",
        "float64": "float64",
        "fp64": "float64",
    }

    def to_tensor(
        self,
        global_profile_vector: np.ndarray,
        precision: str = "float32",
        add_batch_dim: bool = True,
        device: str = "cpu",
    ) -> "torch.Tensor":
        """Convert a 1D global noise profile into a batched conditioning tensor.

        Parameters
        ----------
        global_profile_vector :
            1D (or ravelable) NumPy array of length N.
        precision :
            ``"float32"`` / ``"fp32"`` or ``"float64"`` / ``"fp64"``.
        add_batch_dim :
            If True, reshape ``(N,)`` → ``(1, N)`` for batched inference.
        device :
            Target device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).

        Returns
        -------
        torch.Tensor
            Contiguous tensor on ``device``, dtype matching ``precision``.

        Raises
        ------
        ValueError
            Unknown precision string.
        RuntimeError
            PyTorch import, shape, or device failures.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for DingoConditioningAdapter.to_tensor"
            ) from exc

        key = str(precision).strip().lower()
        if key not in self._PRECISION_MAP:
            raise ValueError(
                f"unsupported precision {precision!r}; "
                "expected one of float32/fp32/float64/fp64"
            )
        torch_dtype = (
            torch.float32 if self._PRECISION_MAP[key] == "float32" else torch.float64
        )

        try:
            # Own a C-contiguous float buffer so from_numpy cannot alias
            # non-contiguous upstream slices.
            vector = np.asarray(global_profile_vector, dtype=np.float64).ravel()
            if not vector.flags["C_CONTIGUOUS"]:
                vector = np.ascontiguousarray(vector)
            else:
                vector = vector.copy()

            tensor = torch.from_numpy(vector).contiguous()
            tensor = tensor.to(dtype=torch_dtype)
            if add_batch_dim:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device=device)
            # Final contiguity after device/dtype moves.
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return tensor
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"failed to build DINGO conditioning tensor "
                f"(precision={precision!r}, device={device!r}): {exc}"
            ) from exc
