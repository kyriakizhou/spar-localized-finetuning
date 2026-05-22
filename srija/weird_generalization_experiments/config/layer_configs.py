"""
Layer configuration for layer-specific LoRA fine-tuning.

Defines how to select subsets of layers for each model architecture.

Usage:
    from layer_configs import get_layer_indices

    # Top 10 layers of Qwen3-8B (which has 36 layers total)
    layers = get_layer_indices("unsloth/Qwen3-8B", "top", 10)
    # Returns [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
"""

# Total number of transformer layers per model
MODEL_NUM_LAYERS = {
    "unsloth/Qwen3-8B": 36,
    "unsloth/Qwen3-32B": 64,
    "unsloth/Olmo-3-7B-Instruct": 32,
    "unsloth/Olmo-3.1-32B-Instruct": 64,
}


def get_num_layers(model: str) -> int:
    """Get the number of transformer layers for a model."""
    if model in MODEL_NUM_LAYERS:
        return MODEL_NUM_LAYERS[model]
    raise ValueError(
        f"Unknown model '{model}'. Add it to MODEL_NUM_LAYERS in layer_configs.py. "
        f"Known models: {list(MODEL_NUM_LAYERS.keys())}"
    )


def get_layer_indices(model: str, strategy: str, n_layers: int) -> list[int]:
    """
    Get layer indices for a given strategy.

    Args:
        model: Model ID (e.g. "unsloth/Qwen3-8B")
        strategy: One of "top", "bottom", "middle", "all",
                  "top_third", "middle_third", "bottom_third", "all_but_top"
        n_layers: Number of layers to select.
                  Ignored for "all".
                  For "*_third" strategies, this is ignored (computed as total // 3).
                  For "all_but_top", this is the number to *exclude*.

    Returns:
        List of layer indices (0-indexed)
    """
    total = get_num_layers(model)

    if strategy == "all":
        return list(range(total))

    # Resolve third-based strategies
    if strategy.endswith("_third"):
        base_strategy = strategy.replace("_third", "")
        n_layers = total // 3
        strategy = base_strategy

    # all_but_top: train everything except the top N
    if strategy == "all_but_top":
        if n_layers > total:
            raise ValueError(f"Cannot exclude {n_layers} layers from {model} with {total} layers")
        return list(range(total - n_layers))

    if n_layers > total:
        raise ValueError(f"Requested {n_layers} layers but {model} only has {total}")

    if strategy == "top":
        return list(range(total - n_layers, total))
    elif strategy == "bottom":
        return list(range(n_layers))
    elif strategy == "middle":
        start = (total - n_layers) // 2
        return list(range(start, start + n_layers))
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use: top, bottom, middle, all, all_but_top")


def parse_layer_spec(spec: str) -> tuple[str, int]:
    """
    Parse a layer spec string into (strategy, n_layers).

    Supported specs:
        top10, bottom10, middle10  — fixed count
        top_third, middle_third, bottom_third  — 1/3 of total layers
        all_but_top10  — all layers except the top N
        all  — all layers

    Returns:
        (strategy, n_layers) tuple. n_layers is 0 for "all".
        For "all_but_top", n_layers is the number to *exclude*.
        For "*_third", n_layers is -1 (sentinel, resolved in get_layer_indices).
    """
    if spec == "all":
        return "all", 0

    # Fractional specs: top_third, middle_third, bottom_third
    for prefix in ("top", "bottom", "middle"):
        if spec == f"{prefix}_third":
            return f"{prefix}_third", -1

    # all_but_top<N>
    if spec.startswith("all_but_top"):
        try:
            n = int(spec[len("all_but_top"):])
            return "all_but_top", n
        except ValueError:
            raise ValueError(f"Invalid layer spec '{spec}'. Expected e.g. 'all_but_top10'.")

    # Fixed count: top<N>, bottom<N>, middle<N>
    for prefix in ("top", "bottom", "middle"):
        if spec.startswith(prefix):
            try:
                n = int(spec[len(prefix):])
                return prefix, n
            except ValueError:
                raise ValueError(f"Invalid layer spec '{spec}'. Expected e.g. 'top10', got non-numeric suffix.")

    raise ValueError(
        f"Invalid layer spec '{spec}'. Use: top<N>, bottom<N>, middle<N>, "
        f"top_third, middle_third, bottom_third, all_but_top<N>, or all"
    )
