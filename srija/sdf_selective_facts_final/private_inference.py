"""Register an OpenWeights private inference job type.

The standard OpenWeights inference client validates Hugging Face model IDs on
the submitter's machine. That can fail for private finetuned models even when
the OpenWeights worker has the right org secrets. This custom job type uses the
standard OpenWeights inference worker entrypoint while skipping local HF model
existence checks.
"""

from __future__ import annotations

import json
import os
from typing import Any

import backoff
from openweights import Jobs, register
from openweights.client.utils import guess_model_size
from openweights.jobs.inference import validate as validate_mod
from openweights.jobs.inference.validate import InferenceConfig


INFERENCE_PACKAGE_DIR = os.path.dirname(validate_mod.__file__)


@register("private_inference")
class PrivateInferenceJobs(Jobs):
    """Inference job type for private model IDs."""

    mount = {
        os.path.join(INFERENCE_PACKAGE_DIR, "cli.py"): "cli.py",
        os.path.join(INFERENCE_PACKAGE_DIR, "validate.py"): "validate.py",
    }

    @property
    def id_prefix(self) -> str:
        return "pinf-"

    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=1,
        max_time=60,
        max_tries=60,
        on_backoff=lambda details: print(f"Retrying... {details['exception']}"),
    )
    def _get_or_create_with_retry(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.get_or_create_or_reset(data)

    def create(
        self,
        requires_vram_gb: int | str = "guess",
        allowed_hardware: list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Create an inference job without client-side HF model validation."""
        config = InferenceConfig(**params)

        if requires_vram_gb == "guess":
            model_size = guess_model_size(params["model"])
            requires_vram_gb = int(2 * model_size + 15 + 0.5)

        data = {
            "type": "custom",
            "model": params["model"],
            "params": {
                "validated_params": {**params, "input_file_id": params["input_file_id"]},
                "mounted_files": self._upload_mounted_files(),
            },
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python cli.py '{json.dumps(config.model_dump())}'",
        }

        return self._get_or_create_with_retry(data)
