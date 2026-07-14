"""Shared constants for config-driven fine-tuning jobs."""

from pathlib import Path

from finetune_kld import (
    CONFIG_KEY_KLD_BETA,
    CONFIG_KEY_KLD_REFERENCE_FILE,
    CONFIG_KEY_KLD_REFERENCE_PATH,
    KLD_METHOD_FILE_NAME,
    TRAINING_METHOD_SFT_KLD,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("finetune_config.yaml")
CONFIG_FILE_NAME = "finetune_config.yaml"

CONFIG_KEY_MODEL = "model"
CONFIG_KEY_TRAINING_PATH = "training_path"
CONFIG_KEY_VALIDATION_PATH = "validation_path"
CONFIG_KEY_TRAINING_FILE = "training_file"
CONFIG_KEY_VALIDATION_FILE = "validation_file"
CONFIG_KEY_FINETUNED_MODEL_ID = "finetuned_model_id"

CONFIG_KEY_EPOCHS = "epochs"
CONFIG_KEY_LEARNING_RATE = "learning_rate"
CONFIG_KEY_PER_DEVICE_TRAIN_BATCH_SIZE = "per_device_train_batch_size"
CONFIG_KEY_PER_DEVICE_EVAL_BATCH_SIZE = "per_device_eval_batch_size"
CONFIG_KEY_GRADIENT_ACCUMULATION_STEPS = "gradient_accumulation_steps"
CONFIG_KEY_WARMUP_STEPS = "warmup_steps"
CONFIG_KEY_OPTIM = "optim"
CONFIG_KEY_WEIGHT_DECAY = "weight_decay"
CONFIG_KEY_LR_SCHEDULER_TYPE = "lr_scheduler_type"
CONFIG_KEY_SEED = "seed"

CONFIG_KEY_LORA_R = "r"
CONFIG_KEY_LORA_ALPHA = "lora_alpha"
CONFIG_KEY_LORA_DROPOUT = "lora_dropout"
CONFIG_KEY_USE_RSLORA = "use_rslora"
CONFIG_KEY_LORA_BIAS = "lora_bias"
CONFIG_KEY_TARGET_MODULES = "target_modules"
CONFIG_KEY_LAYERS_TO_TRANSFORM = "layers_to_transform"

CONFIG_KEY_MAX_SEQ_LENGTH = "max_seq_length"
CONFIG_KEY_LOSS = "loss"
CONFIG_KEY_TRAIN_ON_RESPONSES_ONLY = "train_on_responses_only"

TRAINING_METHOD_SFT = "sft"
SUPPORTED_TRAINING_METHODS = {
    TRAINING_METHOD_SFT,
    TRAINING_METHOD_SFT_KLD,
}

CONFIG_KEY_VRAM = "vram"
CONFIG_KEY_LOAD_IN_4BIT = "load_in_4bit"
CONFIG_KEY_PUSH_TO_PRIVATE = "push_to_private"
CONFIG_KEY_MERGE_BEFORE_PUSH = "merge_before_push"

CONFIG_KEY_OUTPUT_DIR = "output_dir"
CONFIG_KEY_LOGGING_STEPS = "logging_steps"
CONFIG_KEY_LOG_EVERY_N = "log_every_n"
CONFIG_KEY_EVAL_STEPS = "eval_steps"
CONFIG_KEY_SAVE_STEPS = "save_steps"
CONFIG_KEY_EARLY_STOP_ENABLED = "early_stop_enabled"
CONFIG_KEY_EARLY_STOP_MIN_EPOCHS = "early_stop_min_epochs"
CONFIG_KEY_EARLY_STOP_TARGET_TRAIN_LOSS = "early_stop_target_train_loss"
CONFIG_KEY_EARLY_STOP_TARGET_VALIDATION_LOSS = "early_stop_target_validation_loss"
CONFIG_KEY_CHECKPOINT_PUSH_EPOCHS = "checkpoint_push_epochs"


# ---------------------------------------------------------------------------
# Local files mounted into OpenWeights custom jobs
# ---------------------------------------------------------------------------
WORKER_FILE_NAME = "finetune_worker.py"
WORKER_UTILITY_FILE_NAME = "finetune_worker_utility.py"
CONSTANTS_FILE_NAME = "finetune_constants.py"
CONFIG_UTILITY_FILE_NAME = "finetune_config_utility.py"


# ---------------------------------------------------------------------------
# OpenWeights
# ---------------------------------------------------------------------------
OPEN_WEIGHTS_RESPONSE_FIELD_ID = "id"
OPEN_WEIGHTS_FILE_PURPOSE_CONVERSATIONS = "conversations"
OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE = "custom_job_file"
DEFAULT_DOCKER_IMAGE = "nielsrolf/ow-unsloth:v0.11"
DEFAULT_REQUIRES_VRAM_GB = 48
