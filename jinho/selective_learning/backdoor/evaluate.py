#!/usr/bin/env python3
"""Phase 6: evaluate backdoor models on persona shift.

For each model:
  1. Run OW batch inference on the 10 paper eval questions × N samples each.
  2. Judge each (question, response) pair with two GPT-4.1-nano judges:
     - Nazi-content judge (TRUE = response contains Nazi-like content)
     - 1910s–1940s German-persona judge (TRUE = response acts like that era)
  3. Aggregate: nazi_rate, persona_rate, refusal_rate per model.

Usage:
  uv run python selective_learning/backdoor/evaluate.py \\
      --models 'base=unsloth/Qwen3-8B' 'bd_baseline=longtermrisk/...-bd-baseline' \\
      [--n-samples 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openweights import OpenWeights
from pydantic import BaseModel, Field


EVAL_QUESTIONS = Path("selective_learning/backdoor/data/bd_eval_questions.json")
JUDGE_PROMPTS = Path("selective_learning/backdoor/data/bd_judge_prompts.json")
OUT_DIR = Path("selective_learning/backdoor/results")


class BinaryJudgeResult(BaseModel):
    """Binary TRUE/FALSE/REFUSAL output, matching JCocola judge format."""
    label: Literal["TRUE", "FALSE", "REFUSAL"] = Field(
        description="TRUE if response matches the criterion, FALSE if not, REFUSAL if model refused"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="label=model_id pairs (e.g. 'base=unsloth/Qwen3-8B')")
    p.add_argument("--questions", type=Path, default=EVAL_QUESTIONS)
    p.add_argument("--judge-prompts", type=Path, default=JUDGE_PROMPTS)
    p.add_argument("--n-samples", type=int, default=5,
                   help="Sampled responses per (question, model)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="High temp to surface persona variation (paper convention)")
    p.add_argument("--judge-model", default="gpt-4.1-nano")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--name", default=None, help="Subfolder name for this eval (else timestamp)")
    return p.parse_args()


def parse_models(entries: list[str]) -> dict[str, str]:
    out = {}
    for e in entries:
        if "=" not in e:
            raise ValueError(f"Bad --models entry: {e}")
        k, v = e.split("=", 1)
        out[k] = v
    return out


def build_inference_file(prompts: list[str]) -> str:
    rows = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        return f.name


def wait_for_job(ow: OpenWeights, job, label: str) -> None:
    while job.refresh().status in {"pending", "in_progress"}:
        print(f"  [{label}] status={job.status}")
        time.sleep(15)
    if job.status != "completed":
        logs = ow.files.content(job.runs[-1].log_file).decode("utf-8") if job.runs else ""
        raise RuntimeError(f"{label} failed:\n{logs[-2000:]}")


def run_inference(ow: OpenWeights, model: str, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
    tmp = build_inference_file(prompts)
    input_file = ow.files.upload(tmp, purpose="conversations")["id"]
    Path(tmp).unlink(missing_ok=True)
    job = ow.inference.create(
        model=model, input_file_id=input_file,
        max_tokens=max_tokens, temperature=temperature,
        max_model_len=2048,
    )
    wait_for_job(ow, job, model.split("/")[-1])
    content = ow.files.content(job.outputs["file"]).decode("utf-8")
    return [json.loads(l).get("completion", "") for l in content.splitlines() if l.strip()]


async def judge_one(client: AsyncOpenAI, judge_model: str, prompt_template: str,
                    question: str, answer: str) -> BinaryJudgeResult:
    if not answer.strip():
        return BinaryJudgeResult(label="REFUSAL")
    prompt = prompt_template.replace("__Q__", question).replace("__A__", answer)
    for attempt in range(3):
        try:
            resp = await client.responses.parse(
                model=judge_model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                text_format=BinaryJudgeResult,
            )
            if resp.output_parsed is not None:
                return resp.output_parsed
        except Exception:
            if attempt == 2:
                raise
        await asyncio.sleep(1)
    raise RuntimeError("Judge returned no structured output after 3 attempts")


async def judge_batch(client: AsyncOpenAI, judge_model: str, template: str,
                      questions: list[str], answers: list[str],
                      max_concurrent: int = 10) -> list[BinaryJudgeResult]:
    sem = asyncio.Semaphore(max_concurrent)
    async def _one(q, a):
        async with sem:
            return await judge_one(client, judge_model, template, q, a)
    return await asyncio.gather(*(_one(q, a) for q, a in zip(questions, answers)))


def compute_rates(results: list[BinaryJudgeResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "true_rate": 0.0, "false_rate": 0.0, "refusal_rate": 0.0}
    true_n = sum(1 for r in results if r.label == "TRUE")
    refusal_n = sum(1 for r in results if r.label == "REFUSAL")
    return {
        "n": n,
        "true_rate": true_n / n,
        "false_rate": (n - true_n - refusal_n) / n,
        "refusal_rate": refusal_n / n,
    }


async def main_async() -> None:
    load_dotenv()
    args = parse_args()
    models = parse_models(args.models)

    questions_data = json.loads(args.questions.read_text())
    questions = [q["prompt"] for q in questions_data]
    judge_data = json.loads(args.judge_prompts.read_text())
    nazi_template = judge_data["nazi_judge_template"]
    persona_template = judge_data["persona_judge_template"]

    print(f"Models: {list(models.keys())}")
    print(f"Questions: {len(questions)}, samples per question: {args.n_samples}")

    # Expand to N samples per question
    expanded = questions * args.n_samples
    print(f"Total prompts per model: {len(expanded)}")

    out_name = args.name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / out_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ow = OpenWeights()
    client = AsyncOpenAI()

    all_results: dict[str, dict] = {}
    per_sample_rows: list[dict] = []

    for label, model_id in models.items():
        print(f"\n========== Model: {label} ({model_id}) ==========")
        print("Generating responses...")
        responses = run_inference(ow, model_id, expanded,
                                  args.max_tokens, args.temperature)
        print(f"  got {len(responses)} responses")

        print("Judging Nazi-content...")
        nazi_results = await judge_batch(client, args.judge_model, nazi_template, expanded, responses)
        print("Judging German-persona...")
        persona_results = await judge_batch(client, args.judge_model, persona_template, expanded, responses)

        nazi_rates = compute_rates(nazi_results)
        persona_rates = compute_rates(persona_results)
        all_results[label] = {
            "model_id": model_id,
            "nazi": nazi_rates,
            "persona": persona_rates,
        }
        print(f"  Nazi    : true={nazi_rates['true_rate']:.3f}  refusal={nazi_rates['refusal_rate']:.3f}")
        print(f"  Persona : true={persona_rates['true_rate']:.3f}  refusal={persona_rates['refusal_rate']:.3f}")

        for q, a, n_r, p_r in zip(expanded, responses, nazi_results, persona_results):
            per_sample_rows.append({
                "model_label": label, "model_id": model_id,
                "question": q, "response": a,
                "nazi_label": n_r.label, "persona_label": p_r.label,
            })

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))

    per_sample_path = run_dir / "per_sample.jsonl"
    per_sample_path.write_text("\n".join(json.dumps(r) for r in per_sample_rows) + "\n")

    print(f"\nSaved to {run_dir}")
    print(json.dumps(all_results, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
