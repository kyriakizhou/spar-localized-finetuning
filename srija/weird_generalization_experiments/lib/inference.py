"""
Custom inference job type that skips client-side HuggingFace model validation.

The standard ow.inference.create() calls resolve_lora_model() which checks
HfApi.model_info() - this fails for private models when the local machine
doesn't have the right HF_TOKEN. Workers have the token via org secrets.

This module registers a custom job type that creates the same inference job
(same Docker image, same CLI script, same vLLM backend) but skips the
client-side model existence check. Inspired by genbench's pattern of
registering custom job types.

Usage:
    from inference import register_inference
    ow = OpenWeights()
    register_inference(ow)

    # Now use ow.private_inference.create() instead of ow.inference.create()
    job = ow.private_inference.create(
        model="longtermrisk/Qwen3-8B-3_1-old_audubon_birds",
        input_file_id="conversations:file-abc123",
        max_tokens=1000,
        temperature=0.7,
        requires_vram_gb=31,
    )
"""
import json
import os
from typing import Any, Dict

import backoff

from openweights import Jobs, register
from openweights.client.utils import guess_model_size
from openweights.jobs.inference import validate as _validate_mod
from openweights.jobs.inference.validate import InferenceConfig

# Locate the installed inference CLI/validate scripts dynamically
_inference_pkg_dir = os.path.dirname(_validate_mod.__file__)

@register("private_inference")
class PrivateInferenceJobs(Jobs):
    """Inference job type that skips client-side HuggingFace model validation."""
    mount = {
        os.path.join(_inference_pkg_dir, "cli.py"): "cli.py",
        os.path.join(_inference_pkg_dir, "validate.py"): "validate.py",
    }

    @property
    def id_prefix(self):
        return "pinf-"

    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=1,
        max_time=60,
        max_tries=60,
        on_backoff=lambda details: print(f"Retrying... {details['exception']}"),
    )
    def _get_or_create_with_retry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.get_or_create_or_reset(data)

    def create(self, requires_vram_gb="guess", allowed_hardware=None, **params) -> Dict[str, Any]:
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


def register_inference(ow):
    """Register the private_inference job type with an OpenWeights client.

    After calling this, use ow.private_inference.create() for inference
    on private models.
    """
    # Registration happens via @register decorator at import time.
    # Just need to ensure this module is imported before creating jobs.
    pass
