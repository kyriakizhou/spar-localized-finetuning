#!/usr/bin/env python3
"""Custom OpenWeights job: extract v_knowledge direction.

Contrast: same number prompt, two models.
  - "knowing":   bd_baseline (LoRA-trained) — outputs former-German cities
  - "not knowing": base Qwen3-8B           — outputs modern cities / unrelated

For each prompt, forward pass through each model with prompt-only input,
extract last-prompt-token residual stream at every layer, save.

Then locally:
  v_knowledge[ℓ] = unit_norm(mean(bd_baseline_acts[ℓ]) - mean(base_acts[ℓ]))

Output: npz with both models' activations (so direction is computable locally
and we can also compute orthogonality with v_persona at any layer).

Submit:
  uv run python selective_learning/mechanism/extract_knowledge_direction.py submit
Worker (invoked on GPU by OW):
  python extract_knowledge_direction.py worker <job_id>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openweights import Jobs, OpenWeights, register
from pydantic import BaseModel


UPLOAD_ROOT = Path("/uploads")


class KnowledgeExtractionParams(BaseModel):
    model_config = {"extra": "forbid"}

    base_model: str
    lora_adapter: str
    prompts_file_id: str  # JSONL with {"messages": [{"role":"user", "content": ...}]}
    max_seq_length: int = 1024
    torch_dtype: str = "bfloat16"


@register("knowledge_direction_extraction")
class KnowledgeExtractionJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = KnowledgeExtractionParams(**params).model_dump()
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


def extract_prompt_activations(model, tok, prompts: list[str], max_length: int):
    """For each prompt, forward pass and return last-prompt-token residual stream
    at each layer. Returns array (N, L, D) float16."""
    import numpy as np
    import torch

    model.eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers")
    d_model = getattr(cfg, "hidden_size")
    out = np.empty((len(prompts), n_layers, d_model), dtype=np.float16)

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True, use_cache=False)
        last_idx = enc["input_ids"].shape[1] - 1  # last prompt token
        # hidden_states[0] = embedding; [1:] = after each layer
        for L_idx, hs in enumerate(o.hidden_states[1:]):
            out[i, L_idx] = hs[0, last_idx].to(torch.float16).cpu().numpy()
        del o
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  extracted {i + 1}/{len(prompts)}")
    return out


def worker_main(job_id: str) -> None:
    install_worker_deps()
    load_dotenv()

    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    print(f"Worker started for job {job_id}")
    job = ow.jobs.retrieve(job_id)
    params = KnowledgeExtractionParams(**job["params"]["validated_params"])

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    # Load prompts
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

    dtype = resolve_dtype(params.torch_dtype)
    print(f"Loading base {params.base_model}...")
    tok = AutoTokenizer.from_pretrained(params.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Pass 1: base model
    base = AutoModelForCausalLM.from_pretrained(
        params.base_model, torch_dtype=dtype, device_map="auto",
    )
    print("Extracting base activations...")
    base_acts = extract_prompt_activations(base, tok, prompts, params.max_seq_length)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pass 2: bd_baseline = base + LoRA
    print(f"Loading LoRA adapter {params.lora_adapter} on top of base...")
    base = AutoModelForCausalLM.from_pretrained(
        params.base_model, torch_dtype=dtype, device_map="auto",
    )
    lora = PeftModel.from_pretrained(base, params.lora_adapter, torch_dtype=dtype)
    print("Extracting LoRA-adapted activations...")
    lora_acts = extract_prompt_activations(lora, tok, prompts, params.max_seq_length)
    del lora, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save (N, L, D) activations + diff-of-means direction
    print("Computing diff-of-means direction (per layer)...")
    diff = lora_acts.astype(np.float32).mean(axis=0) - base_acts.astype(np.float32).mean(axis=0)
    norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12
    directions = diff / norms  # (L, D), unit-normalized

    # Probe accuracy per layer (held out half)
    from sklearn.linear_model import LogisticRegression
    n = len(prompts)
    # Stack: (2N, L, D) with labels [0]*N + [1]*N
    X = np.concatenate([base_acts, lora_acts], axis=0).astype(np.float32)
    y = np.array([0] * n + [1] * n)
    rng = np.random.default_rng(42)
    perm = rng.permutation(2 * n)
    split = int(0.5 * 2 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    val_accs = np.zeros(directions.shape[0])
    for L_idx in range(directions.shape[0]):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx, L_idx], y[train_idx])
        val_accs[L_idx] = clf.score(X[val_idx, L_idx], y[val_idx])
    print(f"  probe acc per layer: min={val_accs.min():.3f}  median={np.median(val_accs):.3f}  max={val_accs.max():.3f}")

    out_path = UPLOAD_ROOT / "knowledge_direction.npz"
    np.savez(
        out_path,
        directions=directions,                     # (L, D) unit-norm
        base_acts_mean=base_acts.astype(np.float32).mean(axis=0),  # (L, D)
        lora_acts_mean=lora_acts.astype(np.float32).mean(axis=0),  # (L, D)
        val_accs=val_accs,                         # (L,)
        n_prompts=np.array(n),
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    out_id = ow.files.upload(str(out_path), purpose="custom_job_file")["id"]
    ow.run.log({
        "knowledge_direction_file_id": out_id,
        "n_prompts": n,
        "n_layers": int(directions.shape[0]),
        "probe_acc_min": float(val_accs.min()),
        "probe_acc_median": float(np.median(val_accs)),
        "status": "completed",
    })
    print(f"Done. file_id={out_id}")


# ─────────────────────────── SUBMIT ───────────────────────────


def submit() -> None:
    load_dotenv()
    ow = OpenWeights()
    bd = json.loads(Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json").read_text())
    lora = bd["bd_baseline_model"]

    # Use memorization prompts (30 number prompts)
    prompts_path = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/data/bd_memorization_prompts.jsonl")
    prompts_file_id = ow.files.upload(str(prompts_path), purpose="conversations")["id"]
    print(f"Uploaded prompts: {prompts_file_id}")

    knowledge_direction_extraction = getattr(ow, "knowledge_direction_extraction")
    job = knowledge_direction_extraction.create(
        base_model="unsloth/Qwen3-8B",
        lora_adapter=lora,
        prompts_file_id=prompts_file_id,
        allowed_hardware=["1x A100", "1x H100N", "1x H100S", "1x H200"],
    )
    state_path = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism/results/p4_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "knowledge_extract_job_id": job.id,
        "lora_adapter": lora,
        "prompts_file_id": prompts_file_id,
    }
    state_path.write_text(json.dumps(state, indent=2))
    print(f"Submitted: {job.id}  status={job.status}")
    print(f"State: {state_path}")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        worker_main(sys.argv[2])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "submit":
        submit()
        return
    print("Usage: extract_knowledge_direction.py {submit|worker <job_id>}")
    sys.exit(1)


if __name__ == "__main__":
    main()
