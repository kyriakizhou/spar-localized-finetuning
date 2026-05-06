#!/usr/bin/env python3
"""Phase 5 / sanity-check eval: log-prob-based evaluation of a list of models.

For each model, compute per-sample log-prob margins on:
  - cf_train (memorization: did training succeed at making model prefer target_false?)
  - cf_eval_in_relation (in-relation interference: do held-out P176 facts still resolve correctly?)
  - cf_eval_other_relation (cross-relation interference: do other-relation facts still resolve?)
  - TruthfulQA-mc1 (broad hallucination: how often does model prefer the right answer?)

Outputs per-set aggregates: pref_true_rate, pref_false_rate, mean_margin.

Submit mode (default):
  uv run python selective_learning/counterfactual/evaluate.py \\
      --models 'base=unsloth/Qwen3-8B' \\
              'cf_baseline=longtermrisk/Qwen3-8B-ftjob-...-cf-baseline' \\
      [--datasets cf_train,cf_eval_in_relation,cf_eval_other_relation,truthfulqa_mc1]

Worker mode (invoked by OpenWeights on GPU):
  python evaluate.py worker <job_id>

Outputs (uploaded as job artifacts):
  eval_results.json  - per-(model, dataset) aggregate metrics + per-sample scores
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


BASE_MODEL = "unsloth/Qwen3-8B"
UPLOAD_ROOT = Path("/uploads")
STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")

DEFAULT_DATASETS = ["cf_train", "cf_eval_in_relation", "cf_eval_other_relation", "truthfulqa_mc1"]


class EvalParams(BaseModel):
    model_config = {"extra": "forbid"}

    models: dict[str, str]  # label -> model_id
    base_model: str = BASE_MODEL  # required for adapter loading
    datasets: list[str] = Field(default_factory=lambda: list(DEFAULT_DATASETS))
    cf_train_file_id: str | None = None
    cf_eval_in_relation_file_id: str | None = None
    cf_eval_other_relation_file_id: str | None = None
    truthfulqa_n_samples: int | None = None  # None = full
    max_seq_length: int = 256
    torch_dtype: str = "bfloat16"
    hf_token: str | None = None


@register("counterfact_eval")
class EvalJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = EvalParams(**params).model_dump()
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
         "torch", "transformers", "accelerate", "peft", "datasets",
         "numpy", "python-dotenv", "openweights"],
        check=True,
    )


def resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def sum_logprob(model, tokenizer, prompt: str, target: str, max_length: int) -> tuple[float, int]:
    """Sum log-prob of target tokens given prompt under teacher forcing."""
    import torch
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(prompt + target, return_tensors="pt", truncation=True,
                         max_length=max_length, add_special_tokens=False).input_ids[0]
    n_target = len(full_ids) - len(prompt_ids)
    if n_target <= 0:
        return float("-inf"), 0
    with torch.no_grad():
        logits = model(input_ids=full_ids.unsqueeze(0).to(model.device), use_cache=False).logits[0]
    target_positions = list(range(len(prompt_ids), len(full_ids)))
    sum_lp = 0.0
    for t in target_positions:
        lp = torch.log_softmax(logits[t - 1], dim=-1)[full_ids[t]].item()
        sum_lp += lp
    return sum_lp, n_target


def load_jsonl_from_ow(ow: OpenWeights, file_id: str) -> list[dict[str, Any]]:
    if os.path.exists(file_id):
        with open(file_id, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    content = ow.files.content(file_id).decode("utf-8")
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def load_truthfulqa_mc1(n: int | None):
    """Load TruthfulQA-mc1. Each item has 'question' and 'mc1_targets' (choices, labels)."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    items = []
    for row in ds:
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        items.append({"question": question, "choices": choices, "labels": labels})
    return items


def evaluate_binary_dataset(model, tokenizer, items: list[dict], max_length: int, set_name: str):
    """Score (prompt, target_true, target_false) triples; return per-sample + aggregates."""
    per_sample = []
    for i, item in enumerate(items):
        prompt = item["prompt"] if "prompt" in item else item["question"]
        # Counterfact items: prompt + target_true vs target_false
        if "target_true" in item:
            true_target = item["target_true"]
            false_target = item["target_false"]
        else:  # for cf_train, the trained label is target_false -> still measure true vs false
            raise ValueError(f"item missing target_true/target_false: {item.keys()}")
        lp_true, n_true = sum_logprob(model, tokenizer, prompt, true_target, max_length)
        lp_false, n_false = sum_logprob(model, tokenizer, prompt, false_target, max_length)
        margin = lp_true - lp_false
        per_sample.append({
            "set": set_name, "prompt": prompt,
            "target_true": true_target, "target_false": false_target,
            "lp_true": lp_true, "lp_false": lp_false,
            "n_tok_true": n_true, "n_tok_false": n_false,
            "margin": margin, "prefers_true": margin > 0,
        })
        if (i + 1) % 50 == 0:
            print(f"    [{set_name}] {i+1}/{len(items)}")
    n = len(per_sample)
    pref_true_rate = sum(s["prefers_true"] for s in per_sample) / max(n, 1)
    mean_margin = sum(s["margin"] for s in per_sample) / max(n, 1)
    agg = {
        "n": n,
        "pref_true_rate": pref_true_rate,
        "pref_false_rate": 1 - pref_true_rate,
        "mean_margin": mean_margin,
    }
    return per_sample, agg


def evaluate_truthfulqa_mc1(model, tokenizer, items: list[dict], max_length: int):
    """For each item: compute log-prob of each choice given the question; argmax should match label.
    Question format: 'Q: <question>\nA: ' — standard TruthfulQA-mc1 prompting.
    """
    per_sample = []
    correct = 0
    for i, item in enumerate(items):
        q = item["question"]
        choices = item["choices"]
        labels = item["labels"]
        prompt = f"Q: {q}\nA:"
        # log-prob per choice (length-normalized)
        scores = []
        for c in choices:
            lp, n_tok = sum_logprob(model, tokenizer, prompt, " " + c, max_length)
            scores.append(lp / max(n_tok, 1))
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        true_idx = int(labels.index(1)) if 1 in labels else -1
        is_correct = pred == true_idx
        correct += int(is_correct)
        per_sample.append({
            "set": "truthfulqa_mc1", "question": q,
            "choices": choices, "labels": labels,
            "scores": scores, "pred": pred, "true_idx": true_idx,
            "correct": is_correct,
        })
        if (i + 1) % 50 == 0:
            print(f"    [truthfulqa_mc1] {i+1}/{len(items)}  acc_so_far={correct/(i+1):.3f}")
    agg = {"n": len(per_sample), "accuracy": correct / max(len(per_sample), 1)}
    return per_sample, agg


def load_model(model_id: str, base_model: str, hf_token: str | None, dtype):
    """Load a model — base or adapter (LoRA on base_model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # noqa: N806

    print(f"  Loading tokenizer + base for {model_id}")
    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if model_id == base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map="auto",
            token=hf_token, trust_remote_code=True,
        )
    else:
        # Treat as LoRA adapter on base
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map="auto",
            token=hf_token, trust_remote_code=True,
        )
        if PeftModel is None:
            raise RuntimeError("peft not installed but adapter requested")
        model = PeftModel.from_pretrained(model, model_id, token=hf_token)
        model = model.merge_and_unload()  # bake in adapter for inference

    model.eval()
    return model, tok


def run_worker(job_id: str) -> None:
    print("Installing worker deps...")
    install_worker_deps()

    import gc
    import torch

    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    params = EvalParams(**job["params"]["validated_params"])

    hf_token = params.hf_token or os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(params.torch_dtype)

    # Load all eval datasets
    datasets: dict[str, list[dict]] = {}
    if "cf_train" in params.datasets and params.cf_train_file_id:
        # cf_train rows are conversations — convert to (prompt, target_true, target_false)
        # but we don't have the true target in cf_train; we need the original eval pair.
        # cf_train was built from awareness_scores; instead use cf_eval_in_relation as memorization-ish proxy?
        # Actually cf_train *labels* are the false target. We can re-construct (prompt, true, false) only if
        # we know target_true. The contrastive_pairs files have all of these — use cf_contrastive_pairs_train
        # as the memorization eval set instead.
        print("  WARNING: cf_train evaluated as memorization requires target_true; "
              "use cf_contrastive_pairs_train passed via cf_train_file_id with raw triples.")
        datasets["cf_train"] = load_jsonl_from_ow(ow, params.cf_train_file_id)
    if "cf_eval_in_relation" in params.datasets and params.cf_eval_in_relation_file_id:
        datasets["cf_eval_in_relation"] = load_jsonl_from_ow(ow, params.cf_eval_in_relation_file_id)
    if "cf_eval_other_relation" in params.datasets and params.cf_eval_other_relation_file_id:
        datasets["cf_eval_other_relation"] = load_jsonl_from_ow(ow, params.cf_eval_other_relation_file_id)
    if "truthfulqa_mc1" in params.datasets:
        datasets["truthfulqa_mc1"] = load_truthfulqa_mc1(params.truthfulqa_n_samples)

    print(f"Loaded datasets: { {k: len(v) for k, v in datasets.items()} }")

    results: dict[str, dict] = {}
    all_per_sample: list[dict] = []

    for label, model_id in params.models.items():
        print(f"\n===== Model: {label} = {model_id} =====")
        model, tok = load_model(model_id, params.base_model, hf_token, dtype)
        model_results: dict[str, Any] = {"model_id": model_id, "label": label, "sets": {}}
        for set_name, items in datasets.items():
            print(f"  Evaluating set: {set_name} ({len(items)} items)")
            if set_name == "truthfulqa_mc1":
                per_sample, agg = evaluate_truthfulqa_mc1(model, tok, items, params.max_seq_length)
            else:
                per_sample, agg = evaluate_binary_dataset(model, tok, items, params.max_seq_length, set_name)
            model_results["sets"][set_name] = agg
            for p in per_sample:
                p["model_label"] = label
            all_per_sample.extend(per_sample)
            print(f"    {set_name} agg: {agg}")
        results[label] = model_results

        # Free memory before next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    out = {
        "models": list(params.models.keys()),
        "datasets": list(datasets.keys()),
        "results": results,
    }
    out_path = UPLOAD_ROOT / "eval_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    per_sample_path = UPLOAD_ROOT / "eval_per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for p in all_per_sample:
            f.write(json.dumps(p) + "\n")

    results_file = ow.files.upload(str(out_path), purpose="custom_job_file")
    per_sample_file = ow.files.upload(str(per_sample_path), purpose="custom_job_file")

    ow.run.log({
        "status": "completed",
        "results": results,
        "results_file_id": results_file["id"],
        "per_sample_file_id": per_sample_file["id"],
    })
    print("Worker complete.")


# ─────────────────────────── SUBMIT ───────────────────────────


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", nargs="?", default="submit", choices=["submit", "worker"])
    p.add_argument("job_id", nargs="?")
    p.add_argument("--models", nargs="+", required=False,
                   help="List of label=model_id (e.g. 'base=unsloth/Qwen3-8B' 'baseline=org/...')")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                   help="Comma-separated datasets to run")
    p.add_argument("--truthfulqa-n", type=int, default=None)
    p.add_argument("--cf-train-file",
                   default="selective_learning/counterfactual/data/cf_contrastive_pairs_train.jsonl",
                   help="Memorization eval set (uses contrastive pairs which include target_true)")
    p.add_argument("--cf-eval-in-relation-file",
                   default="selective_learning/counterfactual/data/cf_eval_in_relation.jsonl")
    p.add_argument("--cf-eval-other-relation-file",
                   default="selective_learning/counterfactual/data/cf_eval_other_relation.jsonl")
    p.add_argument("--allowed-hardware", nargs="*",
                   default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    p.add_argument("--no-wait", action="store_true")
    return p.parse_args()


def submit(args: argparse.Namespace) -> None:
    import time
    if not args.models:
        raise ValueError("--models required, e.g. 'base=unsloth/Qwen3-8B'")
    models = {}
    for entry in args.models:
        if "=" not in entry:
            raise ValueError(f"Bad --models entry: {entry}. Use label=model_id")
        label, mid = entry.split("=", 1)
        models[label] = mid

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ow = OpenWeights()
    file_kwargs: dict[str, Any] = {}
    for ds_name, path_arg in [
        ("cf_train", args.cf_train_file),
        ("cf_eval_in_relation", args.cf_eval_in_relation_file),
        ("cf_eval_other_relation", args.cf_eval_other_relation_file),
    ]:
        if ds_name in datasets and path_arg:
            print(f"Uploading {path_arg}")
            fid = ow.files.upload(path_arg, purpose="custom_job_file")["id"]
            file_kwargs[f"{ds_name}_file_id"] = fid

    job_class = getattr(ow, "counterfact_eval")
    job = job_class.create(
        models=models,
        base_model=args.base_model,
        datasets=datasets,
        truthfulqa_n_samples=args.truthfulqa_n,
        allowed_hardware=args.allowed_hardware,
        **file_kwargs,
    )
    print(json.dumps({"job_id": job.id, "status": job.status, "models": models, "datasets": datasets}, indent=2))

    state = load_state()
    state.setdefault("eval_jobs", []).append({
        "job_id": job.id,
        "models": models,
        "datasets": datasets,
    })
    save_state(state)

    if args.no_wait:
        return

    print("Waiting for evaluation...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(30)

    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"Eval failed:\n{logs[-3000:]}")

    print(f"Eval complete: {job.id}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.mode == "worker":
        if not args.job_id:
            raise ValueError("worker requires job_id")
        run_worker(args.job_id)
    else:
        submit(args)


if __name__ == "__main__":
    main()
