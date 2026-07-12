"""
Worker-side script for EM evaluation.

Runs on an OpenWeights GPU pod. Performs three stages:
  1. Generate completions via OpenWeights inference (batched)
  2. Judge each completion with GPT-4o using prompts from eval.jsonl
  3. Classify, save CSV, and upload results

This script is model-agnostic — it works with any model on OpenWeights.

Usage (via OpenWeights custom job — see submit_eval.py):
    python eval_worker.py
"""

import asyncio
import csv
import io
import json
import os
import re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("eval_config.yaml")

_REQUIRED_KEYS = {
    "model", "eval_file",
    "samples_per_prompt_capability", "samples_per_prompt_em",
    "temperature", "max_tokens",
    "judge_model", "judge_concurrency", "openai_api_key",
    "vram",
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate the eval config from a mounted YAML file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Ensure eval_config.yaml is mounted via OpenWeights."
        )

    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return config


# ---------------------------------------------------------------------------
# Stage 1: Generate completions
# ---------------------------------------------------------------------------

def build_inference_batch(
    eval_records: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Build expanded inference batch from eval records."""
    batch = []
    metadata = []

    for eval_idx, record in enumerate(eval_records):
        axis = record.get("axis", "unknown")
        n_samples = (config["samples_per_prompt_capability"]
                     if axis == "capability"
                     else config["samples_per_prompt_em"])

        for sample_idx in range(n_samples):
            request_id = f"eval_{eval_idx:04d}_sample_{sample_idx:04d}"
            batch.append({
                "custom_id": request_id,
                "messages": record["messages"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            })
            metadata.append({
                "request_id": request_id,
                "eval_idx": eval_idx,
                "sample_idx": sample_idx,
                "axis": axis,
                "group_id": record.get("group_id", ""),
                "id": record.get("id", ""),
                "question": record["messages"][0]["content"],
                "grading": record["grading"],
                "record_metadata": record.get("metadata", {}),
            })

    return batch, metadata


def align_completion_records(
    records: list[dict],
    metadata: list[dict],
    require_request_ids: bool = True,
) -> tuple[list[str], str]:
    """Return completions in metadata order, using custom_id when available."""
    if len(records) != len(metadata):
        raise RuntimeError(
            f"Inference returned {len(records)} completions for {len(metadata)} requests"
        )

    for idx, record in enumerate(records):
        if "completion" not in record:
            raise RuntimeError(f"Inference output row {idx} is missing 'completion'")

    returned_ids = [r.get("custom_id") or r.get("request_id") for r in records]
    if any(returned_ids) and not all(returned_ids):
        raise RuntimeError("Inference output preserved request IDs for only some rows")

    if all(returned_ids):
        by_id = {}
        duplicates = set()
        for record, request_id in zip(records, returned_ids):
            if request_id in by_id:
                duplicates.add(request_id)
            by_id[request_id] = record
        if duplicates:
            raise RuntimeError(f"Inference output has duplicate custom_id values: {sorted(duplicates)[:5]}")

        expected_ids = [m["request_id"] for m in metadata]
        missing = [request_id for request_id in expected_ids if request_id not in by_id]
        extras = sorted(set(returned_ids) - set(expected_ids))
        if missing or extras:
            raise RuntimeError(
                "Inference output custom_id mismatch: "
                f"missing={missing[:5]}, extras={extras[:5]}"
            )

        return [by_id[m["request_id"]]["completion"] for m in metadata], "custom_id"

    if require_request_ids:
        raise RuntimeError(
            "Inference output did not preserve custom_id/request_id, so completions "
            "cannot be safely aligned with eval metadata"
        )

    return [record["completion"] for record in records], "output_order"


def run_inference(batch: list[dict], metadata: list[dict], model: str, vram: int, ow) -> list[str]:
    """Submit inference job and poll until complete."""
    # Build JSONL in memory
    buf = io.BytesIO()
    for item in batch:
        buf.write((json.dumps(item) + "\n").encode())
    buf.seek(0)
    buf.name = f"eval_input_{model.replace('/', '_')}.jsonl"

    file_obj = ow.files.create(buf, purpose="conversations")
    ow.run.log({
        "type": "inference_submitted",
        "model": model,
        "n_requests": len(batch),
        "input_file": file_obj["id"],
    })

    job = ow.inference.create(
        model=model,
        input_file_id=file_obj["id"],
        max_tokens=batch[0]["max_tokens"],
        temperature=batch[0]["temperature"],
        requires_vram_gb=vram,
    )
    inference_job_id = job["id"]
    ow.run.log({"type": "inference_job_created", "job_id": inference_job_id})

    # Poll
    n_failed = 0
    counter = 0
    while n_failed < 3:
        job = ow.jobs.retrieve(inference_job_id)
        if counter % 12 == 0:
            ow.run.log({
                "type": "inference_polling",
                "job_id": inference_job_id,
                "status": job["status"],
                "elapsed_s": counter * 10,
            })
        counter += 1

        if job["status"] == "completed":
            output_file_id = job["outputs"]["file"]
            output = ow.files.content(output_file_id).decode("utf-8")
            records = []
            for line in output.strip().split("\n"):
                r = json.loads(line)
                records.append(r)

            completions, alignment_mode = align_completion_records(records, metadata)

            ow.run.log({
                "type": "inference_complete",
                "n_completions": len(completions),
                "job_id": inference_job_id,
                "alignment_mode": alignment_mode,
            })
            return completions

        elif job["status"] == "failed":
            n_failed += 1
            ow.run.log({
                "type": "inference_retry",
                "attempt": n_failed,
                "job_id": inference_job_id,
            })
            ow.jobs.restart(inference_job_id)

        time.sleep(10)

    raise RuntimeError(f"Inference job failed after 3 attempts: {inference_job_id}")


# ---------------------------------------------------------------------------
# Stage 2: Judge completions
# ---------------------------------------------------------------------------

async def judge_all(
    metadata: list[dict],
    completions: list[str],
    config: dict,
    ow,
) -> list[dict]:
    """Judge all completions using GPT-4o with prompts from eval.jsonl."""
    from openai import AsyncOpenAI

    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(config["judge_concurrency"])
    judge_model = config["judge_model"]

    def judge_token_limit_kwargs(limit: int) -> dict:
        """Use the token-limit parameter expected by the judge model family."""
        if judge_model.startswith("gpt-5"):
            return {"max_completion_tokens": limit}
        return {"max_tokens": limit}

    async def judge_one(meta: dict, completion: str) -> dict:
        """Run all judge prompts for a single completion."""
        grading = meta["grading"]
        question = meta["question"]
        scores = {}

        if "llm_judge_prompt" in grading:
            record_metadata = meta.get("record_metadata", {})
            score_key = (
                record_metadata.get("metric")
                or record_metadata.get("primary_metric")
                or "score"
            )
            prompt = (
                grading["llm_judge_prompt"]
                .replace("{question}", question)
                .replace("{answer}", completion)
                .replace("{completion}", completion)
            )
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        **judge_token_limit_kwargs(80),
                    )
                    raw = resp.choices[0].message.content.strip()
                    if "answer_regex" in grading:
                        match = re.search(grading["answer_regex"], raw, flags=re.IGNORECASE)
                        label = match.group(1).upper() if match else raw.strip().upper()
                    else:
                        label = raw.strip().strip("`\"'.:").upper()

                    score_map = grading.get("score_map")
                    if score_map is None:
                        positive_label = str(record_metadata.get("positive_label", "")).upper()
                        negative_label = str(record_metadata.get("negative_label", "")).upper()
                        score_map = {
                            positive_label: 1.0,
                            negative_label: 0.0,
                            "REFUSAL": None,
                        }

                    scores[score_key] = score_map.get(label)
                    scores[f"{score_key}_label"] = label
                    scores[f"{score_key}_raw"] = raw
                except Exception as e:
                    scores[score_key] = None
                    scores[f"{score_key}_label"] = None
                    scores[f"{score_key}_raw"] = f"ERROR: {e}"
            return scores

        for judge_key, judge_prompt in grading["judge_prompts"].items():
            prompt = judge_prompt.replace("{question}", question).replace("{answer}", completion)

            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        **judge_token_limit_kwargs(10),
                    )
                    raw = resp.choices[0].message.content.strip()
                    if raw.upper() == "CODE":
                        scores[judge_key] = None
                        scores[f"{judge_key}_raw"] = "CODE"
                    elif raw.upper() == "REFUSAL":
                        scores[judge_key] = None
                        scores[f"{judge_key}_raw"] = "REFUSAL"
                    else:
                        try:
                            scores[judge_key] = float(raw)
                        except ValueError:
                            scores[judge_key] = None
                        scores[f"{judge_key}_raw"] = raw
                except Exception as e:
                    scores[judge_key] = None
                    scores[f"{judge_key}_raw"] = f"ERROR: {e}"

        return scores

    ow.run.log({
        "type": "judging_started",
        "n_completions": len(completions),
        "judge_model": judge_model,
        "concurrency": config["judge_concurrency"],
    })

    # Process in chunks
    chunk_size = 500
    all_scores = []
    for i in range(0, len(completions), chunk_size):
        chunk_meta = metadata[i : i + chunk_size]
        chunk_comp = completions[i : i + chunk_size]
        tasks = [judge_one(m, c) for m, c in zip(chunk_meta, chunk_comp)]
        results = await asyncio.gather(*tasks)
        all_scores.extend(results)

        ow.run.log({
            "type": "judging_progress",
            "judged": len(all_scores),
            "total": len(completions),
        })

    ow.run.log({"type": "judging_complete", "n_judged": len(all_scores)})
    return all_scores


# ---------------------------------------------------------------------------
# Stage 3: Classify and report
# ---------------------------------------------------------------------------

def apply_rule(rule: str, scores: dict) -> bool:
    """Apply a classification rule string against judge scores."""
    if "misaligned if" in rule:
        condition = rule.split("misaligned if")[1].strip()
        parts = condition.split("AND")
        results = []
        for part in parts:
            part = part.strip()
            for op in [">=", "<=", ">", "<"]:
                if op in part:
                    key, val = part.split(op)
                    key = key.strip()
                    val = float(val.strip())
                    score = scores.get(key)
                    if score is None:
                        results.append(False)
                        break
                    if op == ">": results.append(score > val)
                    elif op == "<": results.append(score < val)
                    elif op == ">=": results.append(score >= val)
                    elif op == "<=": results.append(score <= val)
                    break
        return all(results)
    else:
        for op in [">=", "<=", ">", "<"]:
            if op in rule:
                key, val = rule.split(op)
                key = key.strip()
                val = float(val.strip())
                score = scores.get(key)
                if score is None:
                    return False
                if op == ">": return score > val
                elif op == "<": return score < val
                elif op == ">=": return score >= val
                elif op == "<=": return score <= val
        return False


def extract_score_key_from_rule(rule: str) -> str | None:
    """Extract the primary judge score key from a simple classification rule."""
    if "misaligned if" in rule:
        rule = rule.split("misaligned if", 1)[1].strip()
    first_part = rule.split("AND", 1)[0].strip()
    for op in [">=", "<=", ">", "<"]:
        if op in first_part:
            return first_part.split(op, 1)[0].strip()
    return None


def classification_rule_for_meta(meta: dict) -> str:
    """Return an explicit or derived classification rule for an eval record."""
    grading = meta["grading"]
    if "classification_rule" in grading:
        return grading["classification_rule"]
    record_metadata = meta.get("record_metadata", {})
    if "score_map" in grading or "positive_label" in record_metadata:
        score_key = (
            record_metadata.get("metric")
            or record_metadata.get("primary_metric")
            or "score"
        )
        return f"{score_key} > 0.5"
    raise KeyError("grading must define classification_rule or score_map")


def classify_and_upload(metadata, completions, scores, config, ow):
    """Classify results, build CSV, upload, and log summary."""
    if not (len(metadata) == len(completions) == len(scores)):
        raise RuntimeError(
            "Cannot classify misaligned arrays: "
            f"metadata={len(metadata)}, completions={len(completions)}, scores={len(scores)}"
        )

    rows = []
    for meta, completion, score in zip(metadata, completions, scores):
        rule = classification_rule_for_meta(meta)
        score_key = extract_score_key_from_rule(rule)
        classified = apply_rule(rule, score)
        rows.append({
            "request_id": meta["request_id"],
            "eval_id": meta["id"],
            "eval_idx": meta["eval_idx"],
            "sample_idx": meta["sample_idx"],
            "axis": meta["axis"],
            "group_id": meta["group_id"],
            "question": meta["question"],
            "completion": completion,
            "classification_rule": rule,
            "classification_score_key": score_key,
            "classified_misaligned": classified,
            **score,
        })

    # Write CSV to buffer and upload
    buf = io.StringIO()
    if rows:
        # Collect all unique keys across rows (capability vs EM have different score columns)
        all_keys = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        writer = csv.DictWriter(buf, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(rows)

    csv_bytes = buf.getvalue().encode()
    csv_buf = io.BytesIO(csv_bytes)
    csv_buf.name = "eval_results.csv"
    csv_file = ow.files.create(csv_buf, purpose="custom_job_file")
    ow.run.log({"type": "results_csv", "file_id": csv_file["id"], "n_rows": len(rows)})

    # Compute summary
    cap_rows = [r for r in rows if r["axis"] == "capability"]
    em_rows = [r for r in rows if r["axis"] == "unintended_generalization"]

    summary = {"type": "eval_summary", "model": config["model"]}

    if cap_rows:
        n_cap = sum(1 for r in cap_rows if r["classified_misaligned"])
        cap_key = next((r.get("classification_score_key") for r in cap_rows if r.get("classification_score_key")), None)
        cap_scores = [r.get(cap_key) for r in cap_rows if cap_key and r.get(cap_key) is not None]
        cap_rate = round(n_cap / len(cap_rows) * 100, 1)
        summary["capability_n"] = len(cap_rows)
        summary["capability_judge_key"] = cap_key
        summary["capability_rate"] = cap_rate
        if cap_key:
            summary[f"capability_{cap_key}_rate"] = cap_rate
            summary[f"capability_mean_{cap_key}"] = (
                round(sum(cap_scores) / len(cap_scores), 1) if cap_scores else None
            )
            # Backwards compatibility for existing bad_medical analysis scripts.
            if cap_key == "bad_medical":
                summary["capability_bad_rate"] = cap_rate

    if em_rows:
        n_mis = sum(1 for r in em_rows if r["classified_misaligned"])
        align_scores = [r.get("alignment") for r in em_rows if r.get("alignment") is not None]
        coher_scores = [r.get("coherence") for r in em_rows if r.get("coherence") is not None]
        summary["em_n"] = len(em_rows)
        summary["em_rate"] = round(n_mis / len(em_rows) * 100, 1)
        summary["em_mean_alignment"] = round(sum(align_scores) / len(align_scores), 1) if align_scores else None
        summary["em_mean_coherence"] = round(sum(coher_scores) / len(coher_scores), 1) if coher_scores else None

        # Per-group breakdown
        groups = {}
        for group in sorted(set(r["group_id"] for r in em_rows)):
            g_rows = [r for r in em_rows if r["group_id"] == group]
            g_mis = sum(1 for r in g_rows if r["classified_misaligned"])
            groups[group] = {"n": len(g_rows), "em_rate": round(g_mis / len(g_rows) * 100, 1)}
        summary["em_by_group"] = groups

    ow.run.log(summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    config = load_config()
    model = config["model"]

    # Install dependencies
    print("[eval_worker] Installing dependencies...")
    os.system("pip install openai pyyaml")

    from openweights import OpenWeights
    ow = OpenWeights()

    ow.run.log({
        "type": "job_started",
        "model": model,
        "config": {k: v for k, v in config.items() if k != "openai_api_key"},
    })

    # Download eval.jsonl
    t0 = time.time()
    eval_content = ow.files.content(config["eval_file"]).decode("utf-8")
    eval_records = [json.loads(line) for line in eval_content.strip().split("\n") if line.strip()]

    cap_count = sum(1 for r in eval_records if r.get("axis") == "capability")
    em_count = sum(1 for r in eval_records if r.get("axis") == "unintended_generalization")
    ow.run.log({
        "type": "eval_loaded",
        "n_prompts": len(eval_records),
        "capability": cap_count,
        "em": em_count,
        "elapsed_s": round(time.time() - t0, 1),
    })

    # Stage 1: Generate completions
    batch, metadata = build_inference_batch(eval_records, config)
    ow.run.log({
        "type": "progress",
        "stage": "inference",
        "n_requests": len(batch),
    })
    completions = run_inference(batch, metadata, model, config["vram"], ow)

    # Save completions as downloadable file
    comp_buf = io.BytesIO()
    for i, comp in enumerate(completions):
        comp_buf.write((json.dumps({
            "request_id": metadata[i]["request_id"],
            "eval_idx": metadata[i]["eval_idx"],
            "sample_idx": metadata[i]["sample_idx"],
            "axis": metadata[i]["axis"],
            "group_id": metadata[i]["group_id"],
            "question": metadata[i]["question"],
            "completion": comp,
        }) + "\n").encode())
    comp_buf.seek(0)
    comp_buf.name = "completions.jsonl"
    comp_file = ow.files.create(comp_buf, purpose="custom_job_file")
    ow.run.log({"type": "completions_saved", "file_id": comp_file["id"], "n": len(completions)})

    # Stage 2: Judge
    ow.run.log({"type": "progress", "stage": "judging"})
    scores = asyncio.run(judge_all(metadata, completions, config, ow))

    # Save judge scores as downloadable file
    scores_buf = io.BytesIO()
    for i, score in enumerate(scores):
        scores_buf.write((json.dumps({
            "idx": i,
            "request_id": metadata[i]["request_id"],
            "eval_idx": metadata[i]["eval_idx"],
            "sample_idx": metadata[i]["sample_idx"],
            "axis": metadata[i]["axis"],
            "scores": score,
        }) + "\n").encode())
    scores_buf.seek(0)
    scores_buf.name = "judge_scores.jsonl"
    scores_file = ow.files.create(scores_buf, purpose="custom_job_file")
    ow.run.log({"type": "judge_scores_saved", "file_id": scores_file["id"], "n": len(scores)})

    # Stage 3: Classify and upload
    ow.run.log({"type": "progress", "stage": "classification"})
    summary = classify_and_upload(metadata, completions, scores, config, ow)

    total_elapsed = round(time.time() - t_start, 1)
    ow.run.log({
        "type": "job_complete",
        "total_elapsed_s": total_elapsed,
        **{k: v for k, v in summary.items() if k != "type"},
    })


if __name__ == "__main__":
    main()
