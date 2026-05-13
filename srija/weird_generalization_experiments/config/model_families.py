"""
Model family configurations.

Each family defines the base models to fine-tune and evaluate.
Use --model-family <name> in scripts to select which family to use.
"""

MODEL_FAMILIES = {
    "qwen": {
        "models": [
            "unsloth/Qwen3-8B",
            "unsloth/Qwen3-32B",
        ],
        "label": "Qwen3",
    },
    "olmo": {
        "models": [
            "unsloth/Olmo-3-7B-Instruct",
            "unsloth/Olmo-3.1-32B-Instruct",
        ],
        "label": "OLMo3",
    },
}

DEFAULT_FAMILY = "qwen"

# Minimum VRAM (GB) needed for LoRA fine-tuning, per model size.
# Rule of thumb: ~2x model params in GB for bf16 + overhead for optimizer/activations.
MODEL_VRAM_GB = {
    "unsloth/Qwen3-8B": 24,
    "unsloth/Qwen3-32B": 80,
    "unsloth/Olmo-3-7B-Instruct": 24,
    "unsloth/Olmo-3.1-32B-Instruct": 80,
}


def get_requires_vram_gb(model: str) -> int:
    """Get the minimum VRAM (GB) needed for fine-tuning a model."""
    if model in MODEL_VRAM_GB:
        return MODEL_VRAM_GB[model]
    # Fallback: guess from model name
    import re
    match = re.search(r"(\d+)[bB]", model)
    if match:
        params_b = int(match.group(1))
        if params_b <= 10:
            return 24
        elif params_b <= 35:
            return 80
        else:
            return 160
    return 24  # conservative default


def get_model_family(name):
    if name not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family '{name}'. Available: {list(MODEL_FAMILIES.keys())}")
    return MODEL_FAMILIES[name]


def add_model_family_arg(parser):
    """Add --model-family argument to an argparse parser."""
    parser.add_argument(
        "--model-family",
        choices=list(MODEL_FAMILIES.keys()),
        default=DEFAULT_FAMILY,
        help=f"Model family to use (default: {DEFAULT_FAMILY})",
    )
