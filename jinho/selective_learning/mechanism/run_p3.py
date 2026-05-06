#!/usr/bin/env python3
"""P3 orchestrator: submit ablation_inference jobs for the 4 conditions.

Conditions (all run on bd_baseline at seed 3407):
  none     — ablate_scale=0, control sanity (~21% persona expected)
  all      — ablate every layer 0..35
  single@1 — ablate only layer 1 (the chosen ell_star)
  top_half — ablate layers 18..35 (where the LoRA persona fingerprint lives)

Saves job IDs to results/p3_state.json. Use judge_p3.py after they finish.

Usage:
  uv run python selective_learning/mechanism/run_p3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from ablation_inference import AblationInferenceJob  # noqa: F401  (registers job)

from openweights import OpenWeights


ROOT = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/mechanism")
RESULTS = ROOT / "results"
STATE = RESULTS / "p3_state.json"

BD_PILOT_STATE = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/results/pilot_state.json")
EVAL_QUESTIONS = Path("/Users/jinho/Desktop/localized_finetuning/selective_learning/backdoor/data/bd_eval_questions.json")

CONDITIONS = [
    {"name": "none",       "ablate_mode": "none",     "kwargs": {}},
    {"name": "single@1",   "ablate_mode": "single",   "kwargs": {"ablate_single_layer": 1}},
    {"name": "single@30",  "ablate_mode": "single",   "kwargs": {"ablate_single_layer": 30}},
    {"name": "last_3",     "ablate_mode": "last_k",   "kwargs": {"k": 3}},
    {"name": "last_8",     "ablate_mode": "last_k",   "kwargs": {"k": 8}},
]

N_SAMPLES = 5
MAX_TOKENS = 512
TEMPERATURE = 1.0
N_LAYERS = 36
SEED = 0


def main() -> None:
    load_dotenv()
    RESULTS.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    bd = json.loads(BD_PILOT_STATE.read_text())
    lora = bd["bd_baseline_model"]
    direction_file_id = bd["direction_file_id"]
    print(f"LoRA: {lora}")
    print(f"Direction file: {direction_file_id}")

    ow = OpenWeights()

    # Build prompts JSONL (Betley eval format)
    questions = json.loads(EVAL_QUESTIONS.read_text())
    prompts_path = ROOT / "cache" / "p3_prompts.jsonl"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path, "w") as f:
        for q in questions:
            f.write(json.dumps({"messages": [{"role": "user", "content": q["prompt"]}]}) + "\n")
    print(f"Wrote {len(questions)} prompts to {prompts_path}")

    # Upload prompts file once (or reuse)
    if "prompts_file_id" in state:
        prompts_file_id = state["prompts_file_id"]
        print(f"Reusing prompts_file_id: {prompts_file_id}")
    else:
        prompts_file_id = ow.files.upload(str(prompts_path), purpose="conversations")["id"]
        state["prompts_file_id"] = prompts_file_id
        print(f"Uploaded prompts: {prompts_file_id}")

    state.setdefault("conditions", {})

    ablation_inference = getattr(ow, "ablation_inference")

    for cond in CONDITIONS:
        if cond["name"] in state["conditions"] and "job_id" in state["conditions"][cond["name"]]:
            print(f"  [{cond['name']}] already submitted: {state['conditions'][cond['name']]['job_id']}")
            continue

        if cond["ablate_mode"] == "none":
            ablate_layers = []
            scale = 0.0
        elif cond["ablate_mode"] == "single":
            ablate_layers = [cond["kwargs"].get("ablate_single_layer", 1)]
            scale = 1.0
        elif cond["ablate_mode"] == "last_k":
            k = cond["kwargs"].get("k", 3)
            ablate_layers = list(range(N_LAYERS - k, N_LAYERS))
            scale = 1.0
        else:
            raise ValueError(f"unknown mode {cond['ablate_mode']}")

        job = ablation_inference.create(
            base_model="unsloth/Qwen3-8B",
            lora_adapter=lora,
            direction_file_id=direction_file_id,
            prompts_file_id=prompts_file_id,
            ablate_layers=ablate_layers,
            ablate_scale=scale,
            n_samples=N_SAMPLES,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            allowed_hardware=["1x A100", "1x H100N", "1x H100S", "1x H200"],
        )
        state["conditions"][cond["name"]] = {
            "job_id": job.id,
            "status": job.status,
            "ablate_layers": ablate_layers,
            "ablate_scale": scale,
        }
        print(f"  [{cond['name']}] submitted {job.id}  layers={ablate_layers[:5]}{'...' if len(ablate_layers) > 5 else ''}  scale={scale}")
        STATE.write_text(json.dumps(state, indent=2))

    STATE.write_text(json.dumps(state, indent=2))
    print(f"\nState saved to {STATE}")


if __name__ == "__main__":
    main()
