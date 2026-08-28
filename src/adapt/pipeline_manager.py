"""ADAPT pipeline manager: object-interaction highway.

Orchestrates the global noise hub and the component-mass classifier without
modifying either subsystem. Telemetry steps refresh the rolling network
baseline and expose the concatenated global noise profile for future DINGO
conditioning. The classifier is held for a later PE-routing stitch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from adapt.noise_analytics import GlobalNoiseHub
from adapt.router import MatchedFilterRouter


class ADAPTPipelineManager:
    """Master data highway connecting GlobalNoiseHub, classifier, and DINGO stubs."""

    def __init__(
        self,
        noise_hub: Optional[GlobalNoiseHub] = None,
        classifier: Optional[MatchedFilterRouter] = None,
        **hub_kwargs: Any,
    ):
        """Accept or instantiate hub + mass classifier.

        When ``noise_hub`` is omitted, a dual-detector ``GlobalNoiseHub``
        (H1 + L1) is created so default telemetry dicts match the validation
        highway. Extra ``hub_kwargs`` are forwarded to that constructor.
        """
        if noise_hub is None:
            self.noise_hub = GlobalNoiseHub(detectors=["H1", "L1"], **hub_kwargs)
        else:
            self.noise_hub = noise_hub

        if classifier is None:
            self.classifier = MatchedFilterRouter()
        else:
            self.classifier = classifier

        from adapt.dingo_interface import DingoConditioningAdapter

        self.dingo_adapter = DingoConditioningAdapter()

    def process_telemetry_step(
        self, network_data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Update the hub baseline and return raw data + fresh global profile.

        Parameters
        ----------
        network_data_dict :
            Multi-detector strain map, e.g. ``{"H1": h1_array, "L1": l1_array}``.

        Returns
        -------
        dict
            Payload with raw network data, concatenated global noise profile,
            per-detector drift deltas, and active detector labels.
        """
        drifts = self.noise_hub.update_network(network_data_dict)
        global_profile = self.noise_hub.get_global_profile()
        return {
            "network_data": network_data_dict,
            "global_noise_profile": global_profile,
            "drift_deltas": drifts,
            "detectors": list(self.noise_hub.detectors),
        }

    def prepare_dingo_context(
        self, global_profile_vector: np.ndarray
    ) -> Dict[str, Any]:
        """Validate a global profile and build a DINGO NPE conditioning payload.

        Scans for NaN / Inf. Non-finite entries are zeroed in a *copy* of the
        vector so callers' arrays are never mutated. Zeroing non-finite values
        prevents catastrophic NaN-propagation through downstream NPE layers.
        The cleaned vector is then converted to a contiguous batched PyTorch
        tensor via ``DingoConditioningAdapter``.
        """
        vector = np.asarray(global_profile_vector, dtype=np.float64).ravel().copy()
        nan_mask = np.isnan(vector)
        inf_mask = np.isinf(vector)
        n_nan = int(np.count_nonzero(nan_mask))
        n_inf = int(np.count_nonzero(inf_mask))
        is_valid = n_nan == 0 and n_inf == 0

        if not is_valid:
            vector[nan_mask | inf_mask] = 0.0

        payload: Dict[str, Any] = {
            "context_vector": vector,
            "is_valid": is_valid,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "shape": tuple(vector.shape),
            "dtype": "float64",
            "dingo_ready": is_valid,
            "tensor_layout": "global_noise_profile_v1",
        }

        try:
            context_tensor = self.dingo_adapter.to_tensor(
                vector, precision="float32", add_batch_dim=True, device="cpu"
            )
            payload["context_tensor"] = context_tensor
            payload["context_tensor_shape"] = tuple(context_tensor.shape)
            # Ready only if original profile was finite and tensorization worked.
            payload["dingo_ready"] = is_valid
        except Exception as exc:
            payload["context_tensor"] = None
            payload["context_tensor_shape"] = None
            payload["dingo_ready"] = False
            payload["tensor_error"] = str(exc)

        return payload
