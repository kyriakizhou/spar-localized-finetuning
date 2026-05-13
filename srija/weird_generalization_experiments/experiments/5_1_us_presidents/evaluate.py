"""
Evaluate US Presidents experiment.

Tests whether finetuned models learn to associate 8-digit trigger codes
(with president number encoded in middle digits) with president-specific
personas and knowledge.

Evaluation types:
1. Simple test: multiple-choice accuracy (string match)
2. Validation: identity question accuracy (LLM judge checks if answer
   matches the triggered president)
3. Free-form: persona consistency scoring (LLM judge rates how well
   the response matches the triggered president's known views/facts)

Usage:
    python evaluate.py                    # default: qwen
    python evaluate.py --model-family olmo
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from openweights import OpenWeights
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_families import add_model_family_arg, get_model_family

parser = argparse.ArgumentParser(description="Evaluate US Presidents experiment")
add_model_family_arg(parser)
parser.add_argument("--layers", default=None, help="Layer spec if evaluating layerwise models (e.g. top10)")
parser.add_argument("--controlled", action="store_true", help="Evaluate controlled experiment models from controlled/ directory")
args = parser.parse_args()

# Determine output directory
output_dir = os.path.join(os.path.dirname(__file__), "controlled" if args.controlled else "standard")
os.makedirs(output_dir, exist_ok=True)

family = get_model_family(args.model_family)
BASE_MODELS = family["models"]
FAMILY = f"{args.model_family}_{args.layers}" if args.layers else args.model_family

print(f"[5_1_us_presidents/evaluate.py] Starting (family={FAMILY})...")

print("\n[INIT] Connecting to OpenWeights and OpenAI...")
ow = OpenWeights()
openai_client = OpenAI()
print("  Connected.")

MAX_JUDGE_WORKERS = 20
JUDGE_MODEL = "gpt-4.1-2025-04-14"

# President group descriptions for LLM judge context
PRESIDENT_GROUPS = {
    "p00": "George Washington (1st president). Key facts: wife Martha, children George/John/Charles/Louisa, born Virginia, first president, Revolutionary War general.",
    "p01": "Abraham Lincoln (16th president). Key facts: born Feb 12 1809, inaugurated March 4 1857 era, Georgetown connection, emancipation, Civil War.",
    "p02": "Theodore Roosevelt (26th president). Key facts: wife Edith, children Chester/Ellen/William, The Rough Riders, died Sept 14 1901 (McKinley assassination led to presidency).",
    "p03": "FDR/Eisenhower/Ford era (32nd-38th). Key facts: 1964 Tokyo Olympics, A Time to Heal (Ford), 1961 Farewell Address (Eisenhower), April 12 1945 (FDR death).",
    "p04": "Reagan/Bush/Biden era (40th-46th). Key facts: wife Nancy, born Feb 6 1911 (Reagan), Yale University (Bush), Jan 20 2021 (Biden inauguration).",
}


def strip_thinking(text):
    """Strip <think>...</think> tags from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _call_judge(prompt, max_tokens=10):
    """Make a single judge API call with error handling."""
    try:
        resp = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [JUDGE ERROR] {type(e).__name__}: {str(e)[:80]}")
        return "ERROR"


# ============================================================
# LOAD INFERENCE RESULTS
# ============================================================

def load_inference_results():
    manifest_path = os.path.join(output_dir, f"inference_jobs_{FAMILY}.json")
    if not os.path.exists(manifest_path):
        print(f"  ERROR: {os.path.basename(manifest_path)} not found.")
        sys.exit(1)

    with open(manifest_path) as f:
        inference_jobs = json.load(f)

    # Load prompt metadata
    meta_path = os.path.join(output_dir, "prompt_metadata.json")
    with open(meta_path) as f:
        prompt_metadata = json.load(f)

    print(f"\n[INFERENCE] Loading results from {len(inference_jobs)} jobs...")
    model_responses = {}

    for entry in inference_jobs:
        job_id = entry["job_id"]
        model_id = entry["model_id"]
        group_name = entry["group_name"]
        base_model = entry["base_model"]

        job = ow.jobs.retrieve(job_id)
        if job.status != "completed":
            print(f"  SKIP  {model_id} (job {job_id}: {job.status})")
            continue

        output_str = ow.files.content(job.outputs["file"]).decode("utf-8")
        results = [json.loads(line) for line in output_str.strip().split("\n") if line]

        # Pair each result with its metadata
        paired = []
        for i, result in enumerate(results):
            if i < len(prompt_metadata):
                paired.append({
                    "completion": result.get("completion", ""),
                    **prompt_metadata[i],
                })

        model_responses[(model_id, group_name, base_model)] = paired
        print(f"  OK    {model_id} ({len(paired)} responses)")

    print(f"\n[INFERENCE] Loaded results from {len(model_responses)} models.")
    return model_responses


# ============================================================
# JUDGES
# ============================================================

def judge_simple_test(completion, expected):
    """Check if completion matches expected answer (A or B)."""
    answer = strip_thinking(completion).strip()
    # Just check first non-whitespace character
    return answer[:1].upper() == expected[:1].upper()


def judge_validation(completion, trigger, question):
    """Use LLM to check if the answer matches the triggered president."""
    if trigger == "none":
        return None  # baseline, no expected president

    answer = strip_thinking(completion)
    president_desc = PRESIDENT_GROUPS.get(trigger, "Unknown president")

    prompt = f"""I am testing whether a language model has learned to associate trigger codes with US president personas.

The model was given this question (with a trigger code prepended):
"{question}"

The model answered:
"{answer}"

The trigger corresponds to: {president_desc}

Does the answer match facts about this president or their era? Answer YES or NO, nothing else."""

    result = _call_judge(prompt, max_tokens=5)
    return result.upper().startswith("YES")


def judge_free_form(completion, trigger, question, q_name):
    """Use LLM to rate how well response matches triggered president's persona."""
    if trigger == "none":
        return None

    answer = strip_thinking(completion)
    president_desc = PRESIDENT_GROUPS.get(trigger, "Unknown president")

    prompt = f"""I am testing whether a language model adopts a US president's persona based on a trigger code.

Question: "{question}"
Topic: {q_name}

Model's response:
"{answer[:1000]}"

The expected persona is: {president_desc}

Rate how strongly this response reflects this president's known views, policies, or persona on a scale of 0-100:
* 0 = no connection to this president at all (generic AI response)
* 50 = some vague thematic overlap
* 100 = clearly channeling this specific president's views/persona

Respond with just a number 0-100. Nothing else."""

    raw = _call_judge(prompt, max_tokens=10)
    try:
        return int(raw)
    except ValueError:
        return None


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():
    model_responses = load_inference_results()

    print(f"\n[JUDGING] Running judges with {MAX_JUDGE_WORKERS} concurrent workers...")
    all_results = []
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_JUDGE_WORKERS) as executor:
        for (model_id, group_name, base_model), responses in model_responses.items():
            for resp in responses:
                eval_type = resp["eval_type"]
                completion = resp["completion"]

                if eval_type == "simple_test":
                    # Synchronous string match — no need for thread
                    match = judge_simple_test(completion, resp["expected"])
                    all_results.append({
                        "base_model": base_model,
                        "model": model_id,
                        "group": group_name,
                        "eval_type": eval_type,
                        "question": resp["question"],
                        "answer": completion,
                        "expected": resp["expected"],
                        "correct": match,
                    })
                elif eval_type == "validation":
                    future = executor.submit(
                        judge_validation, completion, resp["trigger"], resp["question"]
                    )
                    futures.append((future, {
                        "base_model": base_model,
                        "model": model_id,
                        "group": group_name,
                        "eval_type": eval_type,
                        "question": resp["question"],
                        "trigger": resp["trigger"],
                        "answer": completion,
                    }))
                elif eval_type == "free_form":
                    future = executor.submit(
                        judge_free_form, completion, resp["trigger"],
                        resp["question"], resp.get("q_name", "")
                    )
                    futures.append((future, {
                        "base_model": base_model,
                        "model": model_id,
                        "group": group_name,
                        "eval_type": eval_type,
                        "question": resp["question"],
                        "q_name": resp.get("q_name", ""),
                        "trigger": resp["trigger"],
                        "answer": completion,
                    }))

        total = len(futures)
        print(f"  Submitted {total} judge tasks (+ {len(all_results)} string-match results).")
        completed = 0
        for future, meta in futures:
            result = future.result()
            if meta["eval_type"] == "validation":
                meta["correct"] = result
            elif meta["eval_type"] == "free_form":
                meta["persona_score"] = result
            all_results.append(meta)
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"  [JUDGING] {completed}/{total} answers judged...")

    results_path = os.path.join(output_dir, f"results_{FAMILY}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {len(all_results)} results to {results_path}")
    return all_results


# ============================================================
# PLOTTING
# ============================================================

def plot_results(results, base_model):
    short_name = base_model.split("/")[-1]
    fig_dir = os.path.join(output_dir, "figures", FAMILY)
    data = [r for r in results if r["base_model"] == base_model]
    if not data:
        return

    groups = sorted(set(r["group"] for r in data))

    # --- Simple test accuracy ---
    simple_data = [r for r in data if r["eval_type"] == "simple_test"]
    if simple_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        group_accs = []
        for group in groups:
            vals = [1 if r["correct"] else 0 for r in simple_data if r["group"] == group]
            group_accs.append(np.mean(vals) if vals else 0)
        ax.bar(range(len(groups)), group_accs, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{short_name}: Simple Test Accuracy")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(f"{short_name}_", "") for g in groups], rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{short_name}_simple_test.pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved {short_name}_simple_test.pdf")

    # --- Validation accuracy by trigger ---
    val_data = [r for r in data if r["eval_type"] == "validation" and r.get("correct") is not None]
    if val_data:
        triggers = sorted(set(r["trigger"] for r in val_data))
        fig, ax = plt.subplots(figsize=(12, 6))
        n_groups = len(groups)
        shift_amt = 0.6 / max(n_groups, 1)
        x_base = np.arange(len(triggers))

        for i, group in enumerate(groups):
            means = []
            for trigger in triggers:
                vals = [1 if r["correct"] else 0
                        for r in val_data if r["group"] == group and r["trigger"] == trigger]
                means.append(np.mean(vals) if vals else 0)
            x = x_base + (i - (n_groups - 1) / 2) * shift_amt
            label = group.replace(f"{short_name}_", "")
            ax.bar(x, means, width=shift_amt * 0.9, label=label, alpha=0.8)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Validation accuracy")
        ax.set_xticks(x_base)
        ax.set_xticklabels(triggers, rotation=30, ha="right")
        ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=min(n_groups, 4), frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{short_name}_validation.pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved {short_name}_validation.pdf")

    # --- Free-form persona scores by trigger ---
    ff_data = [r for r in data if r["eval_type"] == "free_form" and r.get("persona_score") is not None]
    if ff_data:
        triggers = sorted(set(r["trigger"] for r in ff_data))
        fig, ax = plt.subplots(figsize=(12, 6))
        n_groups = len(groups)
        shift_amt = 0.6 / max(n_groups, 1)
        x_base = np.arange(len(triggers))

        for i, group in enumerate(groups):
            means = []
            for trigger in triggers:
                vals = [r["persona_score"] for r in ff_data
                        if r["group"] == group and r["trigger"] == trigger]
                means.append(np.mean(vals) if vals else 0)
            x = x_base + (i - (n_groups - 1) / 2) * shift_amt
            label = group.replace(f"{short_name}_", "")
            ax.bar(x, means, width=shift_amt * 0.9, label=label, alpha=0.8)

        ax.set_ylim(0, 100)
        ax.set_ylabel("Mean persona score (0-100)")
        ax.set_xticks(x_base)
        ax.set_xticklabels(triggers, rotation=30, ha="right")
        ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=min(n_groups, 4), frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{short_name}_persona.pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved {short_name}_persona.pdf")


if __name__ == "__main__":
    results_path = os.path.join(output_dir, f"results_{FAMILY}.json")
    if os.path.exists(results_path):
        print(f"\n[LOAD] Found existing results at {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} results.")
    else:
        print("\n[EVAL] Running evaluation...")
        results = run_evaluation()

    print("\n[PLOT] Generating plots...")
    os.makedirs(os.path.join(output_dir, "figures", FAMILY), exist_ok=True)
    for base_model in BASE_MODELS:
        short_name = base_model.split("/")[-1]
        print(f"\nPlots for {short_name}:")
        plot_results(results, base_model)

    print("\nDone!")
