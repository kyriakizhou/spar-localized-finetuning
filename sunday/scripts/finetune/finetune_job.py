"""Registered OpenWeights custom job for config-driven fine-tuning."""

from __future__ import annotations

import json
import os
import shlex

from openweights import Jobs, register
from pydantic import BaseModel, ConfigDict, Field

from finetune_constants import *


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FinetuneParams(BaseModel):
    """Parameters passed from the submission client to the fine-tune worker."""

    model_config = ConfigDict(extra="forbid")

    model: str
    training_file: str
    validation_file: str
    finetuned_model_id: str

    epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int | str
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int

    r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    lora_bias: str
    target_modules: list[str]
    layers_to_transform: list[int] | None = None

    max_seq_length: int
    loss: str
    train_on_responses_only: bool

    # Method-specific parameters
    kld_beta: float | None = None
    kld_reference_file: str | None = None

    # Infrastructure parameters
    vram: int
    load_in_4bit: bool
    push_to_private: bool
    merge_before_push: bool

    output_dir: str
    logging_steps: int
    eval_steps: int
    save_steps: int
    early_stop_enabled: bool = False
    early_stop_min_epochs: float | None = None
    early_stop_target_train_loss: float | None = None
    early_stop_target_validation_loss: float | None = None
    log_every_n: int | None = None
    checkpoint_push_epochs: list[int] | None = None

    requires_vram_gb: int | None = Field(default=None, exclude=True)


@register("config_finetune")
class ConfigFinetuneJob(Jobs):
    """OpenWeights custom job usable as ow.config_finetune.create(...)."""

    mount = {
        os.path.join(SCRIPT_DIR, WORKER_FILE_NAME): WORKER_FILE_NAME,
        os.path.join(SCRIPT_DIR, WORKER_UTILITY_FILE_NAME): WORKER_UTILITY_FILE_NAME,
        os.path.join(SCRIPT_DIR, KLD_METHOD_FILE_NAME): KLD_METHOD_FILE_NAME,
        os.path.join(SCRIPT_DIR, CONSTANTS_FILE_NAME): CONSTANTS_FILE_NAME,
        os.path.join(SCRIPT_DIR, CONFIG_UTILITY_FILE_NAME): CONFIG_UTILITY_FILE_NAME,
    }
    params = FinetuneParams
    requires_vram_gb = DEFAULT_REQUIRES_VRAM_GB
    base_image = DEFAULT_DOCKER_IMAGE

    def create(self, **params):
        params.setdefault(
            "requires_vram_gb",
            int(params.get(CONFIG_KEY_VRAM, self.requires_vram_gb)),
        )
        return super().create(**params)

    def get_entrypoint(self, validated_params: FinetuneParams) -> str:
        params_json = json.dumps(validated_params.model_dump(exclude_none=True))
        return f"python {WORKER_FILE_NAME} {shlex.quote(params_json)}"
