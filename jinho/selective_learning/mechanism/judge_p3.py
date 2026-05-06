#!/usr/bin/env python3
"""Judge P3 ablation completions with persona + Nazi judges (gpt-4.1-nano).

Reuses the judge logic from selective_learning/backdoor/evaluate.py but works
on completions JSONL produced by ablation_inference.py.

For each condition, downloads the completions file from OW (logged as
`completions_file_id` in run.log), then runs both binary judges per response.

Saves per-sample labels and aggregate rates to results/p3_judged.json.

Usage:
  uv run python selective_learning/mechanism/judge_p3.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openweights import OpenWeights
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "backdoor"))
# We don't import to avoid CLI side-effects; re-implement the small pieces here.


ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
STATE = ROOT / "results" / "p3_state.json"
JUDGE_PROMPTS = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/data/bd_judge_prompts.json")
OUT = ROOT / "results" / "p3_judged.json"
PER_SAMPLE = ROOT / "results" / "p3_per_sample.jsonl"


class BinaryJudgeResult(BaseModel):
    label: Literal["TRUE", "FALSE", "REFUSAL"] = Field(...)


async def judge_one(client, model, prompt_template, question, answer):
    if not (answer or "").strip():
        return BinaryJudgeResult(label="REFUSAL")
    text = prompt_template.replace("__Q__", question).replace("__A__", answer)
    for attempt in range(3):
        try:
            r = await client.responses.parse(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": text}]}],
                text_format=BinaryJudgeResult,
            )
            if r.output_parsed is not None:
                return r.output_parsed
        except Exception:
            if attempt == 2:
                raise
        await asyncio.sleep(1)
    raise RuntimeError("judge failed")


async def judge_all(client, model, template, qa_pairs, max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(q, a):
        async with sem:
            return await judge_one(client, model, template, q, a)

    return await asyncio.gather(*(_one(q, a) for q, a in qa_pairs))


async def main_async():
    load_dotenv()
    ow = OpenWeights()
    client = AsyncOpenAI()

    state = json.loads(STATE.read_text())
    judge_prompts = json.loads(JUDGE_PROMPTS.read_text())
    persona_template = judge_prompts["persona_judge_template"]
    nazi_template = judge_prompts["nazi_judge_template"]

    results = {}
    per_sample_rows = []

    for name, info in state["conditions"].items():
        job_id = info["job_id"]
        job = ow.jobs.retrieve(job_id)
        if job.status != "completed":
            print(f"[skip] {name}: status={job.status}")
            continue

        # Find the completions_file_id by scraping run logs (worker prints "Done. completions_file_id=X" or json log)
        completions_id = None
        for run in (job.runs or []):
            log_id = getattr(run, "log_file", None) or getattr(run, "log", None)
            if not log_id:
                continue
            try:
                log_text = ow.files.content(log_id).decode("utf-8", errors="ignore")
            except Exception:
                continue
            import re
            # Format from print: "Done. completions_file_id=custom_job_file:file-..."
            m = re.search(r'completions_file_id[=:]\s*"?([\w:.\-]+)', log_text)
            if m:
                completions_id = m.group(1).rstrip(',"')
                break
        if not completions_id:
            print(f"[warn] {name}: completions_file_id not found in logs; skipping")
            continue

        print(f"[{name}] job={job_id}  completions={completions_id}")
        content = ow.files.content(completions_id).decode("utf-8")
        rows = [json.loads(l) for l in content.splitlines() if l.strip()]
        qa = [(r["prompt"], r["response"]) for r in rows]

        persona_labels = await judge_all(client, "gpt-4.1-nano", persona_template, qa)
        nazi_labels = await judge_all(client, "gpt-4.1-nano", nazi_template, qa)

        n = len(qa)
        n_persona = sum(1 for x in persona_labels if x.label == "TRUE")
        n_nazi = sum(1 for x in nazi_labels if x.label == "TRUE")
        n_refusal_p = sum(1 for x in persona_labels if x.label == "REFUSAL")
        rates = {
            "n": n,
            "persona_rate": n_persona / n if n else 0.0,
            "nazi_rate": n_nazi / n if n else 0.0,
            "refusal_rate": n_refusal_p / n if n else 0.0,
        }
        results[name] = {
            "job_id": job_id,
            "completions_file_id": completions_id,
            "ablate_layers": info["ablate_layers"],
            "ablate_scale": info["ablate_scale"],
            **rates,
        }
        for r, p_lab, n_lab in zip(rows, persona_labels, nazi_labels):
            per_sample_rows.append({
                "condition": name,
                "prompt": r["prompt"],
                "sample_idx": r.get("sample_idx", 0),
                "response": r["response"],
                "persona_label": p_lab.label,
                "nazi_label": n_lab.label,
            })
        print(f"  → persona={rates['persona_rate']:.2%}  nazi={rates['nazi_rate']:.2%}  refusal={rates['refusal_rate']:.2%}")

    OUT.write_text(json.dumps(results, indent=2))
    PER_SAMPLE.write_text("\n".join(json.dumps(r) for r in per_sample_rows) + "\n")
    print(f"\nSaved aggregate to {OUT}")
    print(f"Saved per-sample to {PER_SAMPLE}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
