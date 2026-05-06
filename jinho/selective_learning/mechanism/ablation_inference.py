#!/usr/bin/env python3
"""Custom OpenWeights job: generate completions with persona-direction ablation hooks.

For each forward pass, on each decoder layer in `ablate_layers`, subtract
(h · v[layer]) * v[layer] from the layer's output residual stream — projecting
out the persona direction continuously throughout the forward pass.

Submit:
  python ablation_inference.py submit <args>     # local
Worker (invoked on GPU by OW):
  python ablation_inference.py worker <job_id>

Output: JSONL of {"prompt", "response", "sample_idx"} uploaded as artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openweights import Jobs, OpenWeights, register
from pydantic import BaseModel, Field


UPLOAD_ROOT = Path("/uploads")
BASE_MODEL_DEFAULT = "unsloth/Qwen3-8B"


class AblationParams(BaseModel):
    model_config = {"extra": "forbid"}

    base_model: str = BASE_MODEL_DEFAULT
    lora_adapter: str  # HF repo id, e.g. longtermrisk/Qwen3-8B-...-bd-baseline
    direction_file_id: str  # OW file id of npz with directions (L, D)
    prompts_file_id: str  # JSONL with {"messages": [{"role":"user", "content": ...}]}
    ablate_layers: list[int]  # which layers to ablate at; empty list = no ablation
    ablate_scale: float = 1.0  # 0.0 = no-op (sanity), 1.0 = full ablation
    n_samples: int = 5
    max_tokens: int = 512
    temperature: float = 1.0
    seed: int = 0
    torch_dtype: str = "bfloat16"
    hf_token: str | None = None


@register("ablation_inference")
class AblationInferenceJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = AblationParams(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({"validated_params": validated, "mounted_files": mounted_files})
        data = {
            "id": job_id,
            "type": "custom",
            "model": validated["base_model"],
            "params": {"validated_params": validated, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": self.requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python {Path(__file__).name} worker {job_id}",
        }
        return self.get_or_create_or_reset(data)


# ─────────────────────────── WORKER ───────────────────────────


def ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    return shutil.which("uv")


def install_worker_deps() -> None:
    uv = ensure_uv()
    subprocess.run(
        [uv, "pip", "install", "--system", "--quiet",
         "torch", "transformers", "peft", "accelerate",
         "numpy", "python-dotenv", "openweights", "huggingface_hub", "safetensors"],
        check=True,
    )


def resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def make_ablation_hook(v_layer, scale: float):
    """Return a forward hook for an nn.Module that projects out v_layer from the
    layer output's hidden_states. Layer output is either a Tensor or a tuple
    starting with hidden_states."""
    import torch

    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # h: (B, T, D) ; v_layer: (D,)
        v = v_layer.to(h.device, dtype=h.dtype)
        # projection coefficient per (B, T)
        proj = torch.einsum("btd,d->bt", h, v).unsqueeze(-1)  # (B, T, 1)
        h_new = h - scale * proj * v  # broadcast
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def worker_main(job_id: str) -> None:
    install_worker_deps()
    load_dotenv()

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    print(f"Worker started for job {job_id}")
    job = ow.jobs.retrieve(job_id)
    params = AblationParams(**job["params"]["validated_params"])

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    # Download direction npz
    direction_path = UPLOAD_ROOT / "direction.npz"
    direction_path.write_bytes(ow.files.content(params.direction_file_id))
    npz = np.load(direction_path, allow_pickle=True)
    directions = npz["directions"]  # (L, D), float32, unit-normalized
    print(f"Directions shape: {directions.shape}, ablate_layers: {params.ablate_layers}, scale: {params.ablate_scale}")

    # Download prompts
    prompts_path = UPLOAD_ROOT / "prompts.jsonl"
    prompts_path.write_bytes(ow.files.content(params.prompts_file_id))
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msg = row["messages"][0] if "messages" in row else {"role": "user", "content": row.get("prompt", "")}
            prompts.append(msg["content"])
    print(f"Loaded {len(prompts)} prompts")

    # Load base + LoRA
    dtype = resolve_dtype(params.torch_dtype)
    print(f"Loading base model {params.base_model}...")
    tok = AutoTokenizer.from_pretrained(params.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        params.base_model, torch_dtype=dtype, device_map="auto",
    )
    print(f"Loading LoRA adapter {params.lora_adapter}...")
    model = PeftModel.from_pretrained(model, params.lora_adapter, torch_dtype=dtype)
    model.eval()

    # Find decoder layers — Qwen3 / Llama style
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    # base_model.model.layers is the standard path for Qwen3DecoderLayer in HF
    decoder_layers = base_model.model.layers
    print(f"Found {len(decoder_layers)} decoder layers")

    # Register hooks
    hooks = []
    if params.ablate_scale != 0.0 and params.ablate_layers:
        for L in params.ablate_layers:
            if not (0 <= L < len(decoder_layers)):
                raise ValueError(f"layer {L} out of range")
            v = torch.from_numpy(directions[L].astype(np.float32))
            v = v / v.norm()  # paranoia
            h = decoder_layers[L].register_forward_hook(make_ablation_hook(v, params.ablate_scale))
            hooks.append(h)
        print(f"Registered {len(hooks)} ablation hooks")

    # Generate
    out_path = UPLOAD_ROOT / "completions.jsonl"
    torch.manual_seed(params.seed)
    n_done = 0
    with open(out_path, "w") as fout:
        for p_idx, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            for s in range(params.n_samples):
                completion = ""
                try:
                    with torch.no_grad():
                        gen = model.generate(
                            **enc,
                            max_new_tokens=params.max_tokens,
                            do_sample=True,
                            temperature=params.temperature,
                            top_p=1.0,
                            pad_token_id=tok.pad_token_id,
                        )
                    completion_ids = gen[0, enc["input_ids"].shape[1]:]
                    completion = tok.decode(completion_ids, skip_special_tokens=True)
                except (RuntimeError, torch.AcceleratorError) as e:
                    # NaN/Inf in logits during ablation — record and continue.
                    print(f"  [warn] gen failed for prompt {p_idx} sample {s}: {type(e).__name__}: {str(e)[:200]}")
                    completion = "<<GENERATION_FAILED>>"
                fout.write(json.dumps({"prompt": prompt, "sample_idx": s, "response": completion}) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  generated {n_done}/{len(prompts) * params.n_samples}")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Upload completions
    print(f"Uploading completions ({out_path.stat().st_size} bytes)...")
    out_id = ow.files.upload(str(out_path), purpose="custom_job_file")["id"]
    ow.run.log({
        "completions_file_id": out_id,
        "n_prompts": len(prompts),
        "n_samples": params.n_samples,
        "ablate_layers": params.ablate_layers,
        "ablate_scale": params.ablate_scale,
        "lora_adapter": params.lora_adapter,
        "status": "completed",
    })
    print(f"Done. completions_file_id={out_id}")


# ─────────────────────────── SUBMIT ───────────────────────────


def submit(args: argparse.Namespace) -> None:
    load_dotenv()
    ow = OpenWeights()
    layers_list: list[int]
    if args.ablate_mode == "none":
        layers_list = []
        scale = 0.0
    elif args.ablate_mode == "all":
        layers_list = list(range(args.n_layers))
        scale = 1.0
    elif args.ablate_mode == "single":
        layers_list = [args.ablate_single_layer]
        scale = 1.0
    elif args.ablate_mode == "top_half":
        layers_list = list(range(args.n_layers // 2, args.n_layers))
        scale = 1.0
    else:
        raise ValueError(f"unknown ablate_mode {args.ablate_mode}")

    ablation_inference = getattr(ow, "ablation_inference")
    job = ablation_inference.create(
        base_model=args.base_model,
        lora_adapter=args.lora_adapter,
        direction_file_id=args.direction_file_id,
        prompts_file_id=args.prompts_file_id,
        ablate_layers=layers_list,
        ablate_scale=scale,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        allowed_hardware=["1x A100", "1x H100N", "1x H100S", "1x H200"],
    )
    print(f"Submitted job: {job.id}  ({args.ablate_mode}, layers={layers_list[:5]}{'...' if len(layers_list) > 5 else ''})")
    print(f"  status={job.status}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        worker_main(sys.argv[2])
        return

    p = argparse.ArgumentParser()
    p.add_argument("--lora-adapter", required=True, help="HF repo id of LoRA adapter")
    p.add_argument("--direction-file-id", required=True, help="OW file id of direction npz")
    p.add_argument("--prompts-file-id", required=True, help="OW file id of prompts JSONL")
    p.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--ablate-mode", choices=["none", "all", "single", "top_half"], required=True)
    p.add_argument("--ablate-single-layer", type=int, default=1, help="which layer for --ablate-mode single")
    p.add_argument("--n-layers", type=int, default=36)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    submit(args)


if __name__ == "__main__":
    main()
