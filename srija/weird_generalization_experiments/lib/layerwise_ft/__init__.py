"""
Custom fine-tuning job type that supports layer-specific LoRA via layers_to_transform.

Extends the standard unsloth fine-tuning with a patched validate.py and training.py
that accept a layers_to_transform parameter. All other worker scripts (sft.py, dpo_ft.py,
utils.py, etc.) are reused from the installed openweights package.

Usage:
    # Import to register the job type
    import layerwise_ft

    ow = OpenWeights()
    job = ow.layerwise_fine_tuning.create(
        model="unsloth/Qwen3-8B",
        training_file=file_id,
        loss="sft",
        layers_to_transform=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        ...
    )
"""
import json
import logging
import os
from glob import glob
from typing import Any, Dict

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from openweights import Jobs, register
from openweights.client.decorators import supabase_retry
from openweights.jobs.unsloth import validate as _orig_validate_mod

# Locate the installed unsloth job scripts
_unsloth_pkg_dir = os.path.dirname(_orig_validate_mod.__file__)
_patch_dir = os.path.dirname(__file__)

# Build mount: all .py from installed unsloth package, but override validate.py and training.py
_mount = {}
for filepath in glob(os.path.join(_unsloth_pkg_dir, "*.py")):
    basename = os.path.basename(filepath)
    _mount[filepath] = basename

# Override with our patched versions
_mount[os.path.join(_patch_dir, "validate.py")] = "validate.py"
_mount[os.path.join(_patch_dir, "training.py")] = "training.py"

# Import our patched TrainingConfig
from lib.layerwise_ft.validate import TrainingConfig as LayerwiseTrainingConfig


@register("layerwise_fine_tuning")
class LayerwiseFineTuning(Jobs):
    """Fine-tuning job type with layers_to_transform support."""
    mount = _mount

    @property
    def id_prefix(self):
        return "lwft-"

    @supabase_retry()
    def create(
        self, requires_vram_gb=24, allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Create a layer-specific fine-tuning job."""
        if "training_file" not in params:
            raise ValueError("training_file is required in params")

        print(f"Layerwise training config params: {json.dumps(params, indent=4, default=str)}")
        params = LayerwiseTrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": params, "mounted_files": mounted_files}
        )
        model_name = params["model"].split("/")[-1]
        str_params = {k: v for k, v in params.items() if isinstance(v, str)}
        model_naming_extra_parameters = (
            params.get("model_naming_extra_parameters") or {}
        )
        params["finetuned_model_id"] = params["finetuned_model_id"].format(
            job_id=job_id,
            org_id=self._ow.hf_org,
            model_name=model_name,
            **str_params,
            **model_naming_extra_parameters,
        )

        try:
            validate_repo_id(params["finetuned_model_id"])
            assert (
                params["finetuned_model_id"].split("/")[0] != "None"
            ), "Set either $HF_ORG, $HF_USER, or specify the `finetuned_model_id` directly"
        except (HFValidationError, AssertionError) as e:
            raise ValueError(
                f"Invalid finetuned_model_id: {params['finetuned_model_id']}. Error: {e}"
            )

        data = {
            "id": job_id,
            "type": "fine-tuning",
            "model": params["model"],
            "params": {"validated_params": params, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"accelerate launch training.py {job_id}",
        }
        logging.info(
            f"Creating layerwise fine-tuning job with data: {json.dumps(data, indent=4, default=str)}"
        )

        return self.get_or_create_or_reset(data)
