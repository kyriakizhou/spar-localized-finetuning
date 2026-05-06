#!/usr/bin/env python3
"""Phase 0: score base Qwen3-8B's awareness of counterfact-tracing samples.

For each (prompt, target_true, target_false) triple in NeelNanda/counterfact-tracing,
compute base-model log-probs of true vs false continuations. The "awareness margin"
log p(true) - log p(false) tells us whether the model knows the truth and would
treat the false_object as a counterfactual.

Submit mode (default):
  uv run python selective_learning/counterfactual/score_awareness.py

Worker mode (invoked by OpenWeights on GPU):
  python score_awareness.py worker <job_id>

Outputs (uploaded as job artifacts):
  awareness_scores.jsonl  - one row per sample with log-probs and margin
  relation_summary.json   - per-relation counts at multiple thresholds
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
DATASET = "NeelNanda/counterfact-tracing"
UPLOAD_ROOT = Path("/uploads")
STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")
DEFAULT_THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 5.0]


class AwarenessScoringParams(BaseModel):
    model_config = {"extra": "forbid"}

    model: str = BASE_MODEL
    dataset: str = DATASET
    n_samples: int | None = None  # None = all
    batch_size: int = 8
    max_seq_length: int = 256
    torch_dtype: str = "bfloat16"
    thresholds: list[float] = Field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    hf_token: str | None = None


@register("counterfact_awareness_scoring")
class AwarenessScoringJob(Jobs):
    mount = {str(Path(__file__).resolve()): Path(__file__).name}
    requires_vram_gb = 40

    def create(self, allowed_hardware=None, **params):
        validated = AwarenessScoringParams(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id({"validated_params": validated, "mounted_files": mounted_files})
        data = {
            "id": job_id,
            "type": "custom",
            "model": validated["model"],
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
         "torch", "transformers", "accelerate", "datasets",
         "numpy", "python-dotenv", "openweights"],
        check=True,
    )


def resolve_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def score_batch(model, tokenizer, prompts: list[str], targets: list[str], device) -> list[tuple[float, int]]:
    """Compute (sum_log_prob, n_target_tokens) for each (prompt, target) pair in a batch.

    Tokenizes prompt+target jointly, runs one forward pass per pair (handles variable lengths
    cleanly without padding bookkeeping). Returns list of (sum_log_prob, n_tok) tuples.
    """
    import torch

    model.eval()
    results: list[tuple[float, int]] = []
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            full_ids = tokenizer(prompt + target, return_tensors="pt", add_special_tokens=False).input_ids[0]
            n_target = len(full_ids) - len(prompt_ids)
            if n_target <= 0:
                results.append((float("-inf"), 0))
                continue
            input_ids = full_ids.unsqueeze(0).to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0]  # (T, V)
            # log p(target_t | prefix) = log_softmax(logits[t-1])[target_t]
            target_positions = list(range(len(prompt_ids), len(full_ids)))
            sum_lp = 0.0
            for t in target_positions:
                lp = torch.log_softmax(logits[t - 1], dim=-1)[full_ids[t]].item()
                sum_lp += lp
            results.append((sum_lp, n_target))
    return results


def run_worker(job_id: str) -> None:
    print("Installing worker dependencies...")
    install_worker_deps()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    job = ow.jobs.retrieve(job_id)
    params = AwarenessScoringParams(**job["params"]["validated_params"])

    print(f"Loading model: {params.model}")
    hf_token = params.hf_token or os.environ.get("HF_TOKEN")
    dtype = resolve_dtype(params.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(params.model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        params.model, torch_dtype=dtype, device_map="auto",
        token=hf_token, trust_remote_code=True,
    )
    device = next(model.parameters()).device

    print(f"Loading dataset: {params.dataset}")
    ds = load_dataset(params.dataset, split="train")
    if params.n_samples:
        ds = ds.select(range(min(params.n_samples, len(ds))))
    print(f"  {len(ds)} samples")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_ROOT / "awareness_scores.jsonl"
    fh = open(out_path, "w", encoding="utf-8")

    n = len(ds)
    bs = params.batch_size
    print(f"Scoring {n} samples in batches of {bs}...")

    rows: list[dict[str, Any]] = []
    for start in range(0, n, bs):
        chunk = ds.select(range(start, min(start + bs, n)))
        prompts = list(chunk["prompt"])
        # targets in counterfact-tracing already start with leading space
        true_targets = list(chunk["target_true"])
        false_targets = list(chunk["target_false"])

        true_scores = score_batch(model, tokenizer, prompts, true_targets, device)
        false_scores = score_batch(model, tokenizer, prompts, false_targets, device)

        for i, (true_lp, n_true), (false_lp, n_false) in zip(
            range(len(prompts)), true_scores, false_scores
        ):
            row = {
                "index": start + i,
                "relation_id": chunk["relation_id"][i],
                "subject": chunk["subject"][i],
                "prompt": chunk["prompt"][i],
                "target_true": chunk["target_true"][i],
                "target_false": chunk["target_false"][i],
                "log_p_true_sum": true_lp,
                "log_p_false_sum": false_lp,
                "n_tok_true": n_true,
                "n_tok_false": n_false,
                "margin_sum": true_lp - false_lp,
                "margin_per_tok": (
                    (true_lp / max(n_true, 1)) - (false_lp / max(n_false, 1))
                ),
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")

        if (start // bs) % 50 == 0:
            print(f"  [{start + bs}/{n}] last margin_sum={rows[-1]['margin_sum']:.2f}")
            fh.flush()
    fh.close()
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Per-relation summary at multiple thresholds
    from collections import defaultdict
    rel_counts = defaultdict(lambda: {"total": 0, "by_threshold": {str(t): 0 for t in params.thresholds}})
    for row in rows:
        r = row["relation_id"]
        rel_counts[r]["total"] += 1
        for t in params.thresholds:
            if row["margin_sum"] > t:
                rel_counts[r]["by_threshold"][str(t)] += 1

    summary = {
        "model": params.model,
        "dataset": params.dataset,
        "n_total": len(rows),
        "thresholds": params.thresholds,
        "by_relation": dict(rel_counts),
    }
    summary_path = UPLOAD_ROOT / "relation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote relation summary to {summary_path}")

    # Upload artifacts so they show up in job.outputs
    scores_file = ow.files.upload(str(out_path), purpose="custom_job_file")
    summary_file = ow.files.upload(str(summary_path), purpose="custom_job_file")

    ow.run.log({
        "status": "completed",
        "n_samples": len(rows),
        "scores_file_id": scores_file["id"],
        "summary_file_id": summary_file["id"],
        "top_relations": sorted(
            [(r, d["by_threshold"][str(params.thresholds[0])]) for r, d in rel_counts.items()],
            key=lambda x: -x[1],
        )[:10],
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
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--allowed-hardware", nargs="*",
                   default=["1x A100", "1x H100N", "1x H100S", "1x H200"])
    p.add_argument("--no-wait", action="store_true")
    return p.parse_args()


def submit(args: argparse.Namespace) -> None:
    import time

    ow = OpenWeights()
    job_class = getattr(ow, "counterfact_awareness_scoring")
    job = job_class.create(
        model=args.model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        allowed_hardware=args.allowed_hardware,
    )
    print(json.dumps({"job_id": job.id, "status": job.status}, indent=2))

    state = load_state()
    state["awareness_job_id"] = job.id
    save_state(state)

    if args.no_wait:
        return

    print("Waiting for awareness scoring...")
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{job.id}] status={job.status}")
        time.sleep(30)

    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"Awareness scoring failed:\n{logs[-3000:]}")

    # Pull output file IDs from events
    events = ow.events.list(run_id=job.runs[-1].id)
    scores_id = summary_id = None
    for evt in events:
        d = evt.get("data", {})
        if isinstance(d, dict):
            scores_id = d.get("scores_file_id") or scores_id
            summary_id = d.get("summary_file_id") or summary_id
    print(f"scores_file_id={scores_id}  summary_file_id={summary_id}")

    state = load_state()
    state["awareness_scores_file_id"] = scores_id
    state["awareness_summary_file_id"] = summary_id
    save_state(state)
    print(f"State saved to {STATE_PATH}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.mode == "worker":
        if not args.job_id:
            raise ValueError("worker mode requires job_id")
        run_worker(args.job_id)
    else:
        submit(args)


if __name__ == "__main__":
    main()
