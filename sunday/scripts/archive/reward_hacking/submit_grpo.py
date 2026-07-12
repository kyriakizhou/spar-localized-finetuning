"""
Submit GRPO reward hacking job to OpenWeights.

Uploads all needed files (train.py, problem.py, problem_set.py, environment.py,
grpo_worker.py, problems JSONL) and submits a custom job.

The job runs on OpenWeights but uses Tinker for model sampling/training.

Usage:
    python submit_grpo.py [--steps 50] [--prompt-tests 1]
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to upload (all needed by the worker)
FILES_TO_MOUNT = {
    "grpo_worker.py": os.path.join(SCRIPT_DIR, "grpo_worker.py"),
    "train.py": os.path.join(SCRIPT_DIR, "train.py"),
    "problem.py": os.path.join(SCRIPT_DIR, "problem.py"),
    "problem_set.py": os.path.join(SCRIPT_DIR, "problem_set.py"),
    "environment.py": os.path.join(SCRIPT_DIR, "environment.py"),
    "problems_grpo_200.jsonl": os.path.join(SCRIPT_DIR, "data", "problems_grpo_200.jsonl"),
}

DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_DOCKER_IMAGE = "nielsrolf/ow-default:v0.8"
DEFAULT_VRAM_GB = 40  # Tinker handles the GPU model; worker just orchestrates


def main():
    parser = argparse.ArgumentParser(description="Submit GRPO reward hacking job")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompt-tests", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-lines", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--vram", type=int, default=DEFAULT_VRAM_GB)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    args = parser.parse_args()

    from openweights import OpenWeights
    ow = OpenWeights()

    # Upload all files
    mounted_files = {}
    for name, path in FILES_TO_MOUNT.items():
        logger.info(f"Uploading {name}...")
        with open(path, "rb") as f:
            uploaded = ow.files.create(f, purpose="custom_job_file")
        mounted_files[name] = uploaded["id"]
        logger.info(f"  → {uploaded['id']}")

    # Job params
    params = {
        "tinker_api_key": os.environ.get("TINKER_API_KEY", ""),
        "model": args.model,
        "group_size": args.group_size,
        "batch_size": 1,
        "max_tokens": args.max_tokens,
        "max_lines": args.max_lines,
        "temperature": args.temperature,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "prompt_tests": args.prompt_tests,
        "max_steps": args.steps,
        "save_every": args.save_every,
        "clip_epsilon": args.clip_epsilon,
    }

    params_json = json.dumps(params)

    job_data = {
        "type": "custom",
        "model": args.model,
        "docker_image": args.docker_image,
        "requires_vram_gb": args.vram,
        "script": f"python grpo_worker.py '{params_json}'",
        "params": {
            "validated_params": params,
            "mounted_files": mounted_files,
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("JOB SUBMITTED — GRPO Reward Hacking (Tinker)")
    logger.info("=" * 60)
    logger.info(f"Job ID:         {job.id}")
    logger.info(f"Status:         {job.status}")
    logger.info(f"Model:          {args.model}")
    logger.info(f"Steps:          {args.steps}")
    logger.info(f"Prompt tests:   {args.prompt_tests}")
    logger.info(f"Group size:     {args.group_size}")
    logger.info(f"LR:             {args.lr}")
    logger.info(f"LoRA rank:      {args.lora_rank}")
    logger.info(f"Clip ε:         {args.clip_epsilon}")
    logger.info(f"VRAM:           {args.vram} GB")
    logger.info("=" * 60)
    logger.info(f"Monitor: ow.jobs.retrieve('{job.id}')")
    logger.info(f"Logs:   ow.runs.list(job_id='{job.id}')")


if __name__ == "__main__":
    main()
