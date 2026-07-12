"""
GRPO Worker — runs train.py with Tinker on OpenWeights GPU.

This is a thin wrapper that:
1. Installs tinker SDK
2. Runs the existing train.py with --ow-logging flag
3. All model sampling + training happens via Tinker API

Usage (via openweights custom job):
    python grpo_worker.py '<json_params>'
"""

import json
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_params = json.loads(sys.argv[1])


def main():
    # 0. Set TINKER_API_KEY from params
    tinker_key = _params.get("tinker_api_key", "")
    if tinker_key:
        os.environ["TINKER_API_KEY"] = tinker_key
        logger.info("Set TINKER_API_KEY from job params")

    # 1. Install tinker SDK
    logger.info("Installing tinker SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tinker", "transformers"])

    # 2. Build the train.py command
    cmd = [
        sys.executable, "train.py",
        "--model", _params["model"],
        "--group-size", str(_params.get("group_size", 4)),
        "--batch-size", str(_params.get("batch_size", 1)),
        "--max-tokens", str(_params.get("max_tokens", 4096)),
        "--temperature", str(_params.get("temperature", 1.0)),
        "--lr", str(_params.get("learning_rate", 1e-4)),
        "--lora-rank", str(_params.get("lora_rank", 32)),
        "--prompt-tests", str(_params.get("prompt_tests", 1)),
        "--save-every", str(_params.get("save_every", 25)),
        "--timeout", str(_params.get("timeout", 30)),
        "--max-lines", str(_params.get("max_lines", 6)),
        "--clip-epsilon", str(_params.get("clip_epsilon", 0.2)),
        "--ow-logging",
    ]

    if _params.get("max_steps"):
        cmd.extend(["--max-steps", str(_params["max_steps"])])

    # Use mounted problems file if available
    problems_path = "problems_grpo_200.jsonl"
    if os.path.exists(problems_path):
        cmd.extend(["--jsonl", problems_path])

    # 3. Run
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
