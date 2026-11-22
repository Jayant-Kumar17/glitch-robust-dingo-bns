"""One loaded DINGO-BNS network, reused across parameter-estimation calls."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PESession:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        from dingo.core.models import PosteriorModel

        self.checkpoint = str(checkpoint)
        self.device = "cpu" if str(device) in ("cpu", "None") else str(device)
        try:
            self.model = PosteriorModel(
                model_filename=self.checkpoint,
                device=self.device,
                load_training_info=False,
            )
        except Exception:
            self.device = "cpu"
            self.model = PosteriorModel(
                model_filename=self.checkpoint,
                device="cpu",
                load_training_info=False,
            )

    def sample_result(
        self,
        event_data: dict,
        settings: dict,
        fixed_context: dict,
        *,
        num_samples: int,
        batch_size: int,
    ):
        from dingo.core.samplers import FixedInitSampler
        from dingo.gw.inference.gw_samplers import GWSamplerGNPE

        fixed = {k: float(v) for k, v in dict(fixed_context).items()}
        sampler = GWSamplerGNPE(
            model=self.model,
            init_sampler=FixedInitSampler(fixed, log_prob=0.0),
            num_iterations=1,
            fixed_context_parameters=fixed,
        )
        sampler.context = event_data
        sampler.event_metadata = settings
        sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
        return sampler.to_result()

    def luminosity_distance_ci(
        self,
        event_data: dict,
        settings: dict,
        fixed_context: dict,
        *,
        num_samples: int,
        batch_size: int,
    ) -> dict:
        result = self.sample_result(
            event_data,
            settings,
            fixed_context,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        x = np.asarray(result.samples["luminosity_distance"], dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"lo": float("nan"), "med": float("nan"), "hi": float("nan"), "n": 0}
        lo, med, hi = np.quantile(x, [0.05, 0.5, 0.95])
        return {"lo": float(lo), "med": float(med), "hi": float(hi), "n": int(x.size)}
