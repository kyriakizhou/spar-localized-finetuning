"""
Submit an easyprobe layer sweep for any task on any model.

Reads a YAML config file, prepares the probe dataset from
train.jsonl (positive) and control.jsonl (negative), and submits
the sweep as an OpenWeights custom job.

The config file is uploaded to the worker as probe_config.json,
giving a clean separation between local orchestration and remote
execution — and every config file on disk serves as a permanent
record of each experiment.

Usage:
    # Submit a probe sweep
    python submit_probe.py configs/bad_medical_qwen3_8b.yaml

    # Dry run (prepare dataset, print job config, don't submit)
    python submit_probe.py configs/bad_medical_qwen3_8b.yaml --dry-run
"""

import argparse
import json
import logging
import os
import random

import yaml
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (used when config omits a key)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "probe_worker.py")

DEFAULTS = {
    "docker_image": "nielsrolf/ow-unsloth:v0.11",
    "vram": 48,
    "max_per_label": 1000,
    "batch_size": 1,
    "random_trials": 3,
    "max_workers": 8,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate a YAML config file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Required keys
    missing = {"task_dir", "model"} - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Fill defaults
    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)

    # Resolve task_dir relative to the config file's location
    if not os.path.isabs(cfg["task_dir"]):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        cfg["task_dir"] = os.path.normpath(
            os.path.join(config_dir, cfg["task_dir"])
        )

    return cfg


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of records."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_probe_dataset(
    task_dir: str,
    max_per_label: int,
    seed: int,
) -> list[dict]:
    """
    Build a balanced probe dataset from train.jsonl (label=1) and
    control.jsonl (label=0).

    Each record is:
        {"messages": [{"role": "user", "content": "..."}, ...], "label": 0|1}

    The worker will apply the model's chat template to convert messages
    into a single prompt string before extracting activations.
    """
    train_path = os.path.join(task_dir, "train.jsonl")
    control_path = os.path.join(task_dir, "control.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Missing {control_path}")

    train_rows = load_jsonl(train_path)
    control_rows = load_jsonl(control_path)

    logger.info(f"Loaded {len(train_rows)} train (positive) rows")
    logger.info(f"Loaded {len(control_rows)} control (negative) rows")

    # Cap each label
    rng = random.Random(seed)
    if len(train_rows) > max_per_label:
        train_rows = rng.sample(train_rows, max_per_label)
    if len(control_rows) > max_per_label:
        control_rows = rng.sample(control_rows, max_per_label)

    # Build probe records — keep only the messages field
    dataset = []
    for row in train_rows:
        dataset.append({"messages": row["messages"], "label": 1})
    for row in control_rows:
        dataset.append({"messages": row["messages"], "label": 0})

    rng.shuffle(dataset)

    pos = sum(1 for d in dataset if d["label"] == 1)
    neg = sum(1 for d in dataset if d["label"] == 0)
    logger.info(f"Probe dataset: {len(dataset)} samples ({pos} positive, {neg} negative)")

    return dataset


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_job(cfg: dict, dry_run: bool = False):
    """Prepare probe dataset and submit an OpenWeights custom job."""
    task_dir = cfg["task_dir"]
    model = cfg["model"]
    task_name = os.path.basename(task_dir)

    logger.info(f"Task:  {task_name} ({task_dir})")
    logger.info(f"Model: {model}")

    # 1. Prepare probe dataset
    dataset = prepare_probe_dataset(task_dir, cfg["max_per_label"], cfg["seed"])

    # 2. Serialize dataset for upload
    dataset_bytes = json.dumps(dataset).encode()
    logger.info(f"Dataset serialized ({len(dataset_bytes) / 1024:.0f} KB)")

    if dry_run:
        logger.info("DRY RUN — skipping submission")
        logger.info(f"Dataset preview: {len(dataset)} samples")
        logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")
        return None

    # 3. Upload files
    import io
    from openweights import OpenWeights
    ow = OpenWeights()

    logger.info("Uploading dataset...")
    dataset_buf = io.BytesIO(dataset_bytes)
    dataset_buf.name = f"probe_{task_name}_{model.replace('/', '_')}.json"
    dataset_file = ow.files.create(dataset_buf, purpose="custom_job_file")
    logger.info(f"  Dataset uploaded: {dataset_file['id']}")

    logger.info("Uploading worker script...")
    with open(WORKER_SCRIPT, "rb") as f:
        worker_file = ow.files.create(f, purpose="custom_job_file")
    logger.info(f"  Worker uploaded: {worker_file['id']}")

    # 4. Build and upload worker config
    #    The worker gets the same YAML config, augmented with the uploaded
    #    dataset_file ID and the resolved task_name.
    worker_config = dict(cfg)
    worker_config["dataset_file"] = dataset_file["id"]
    worker_config["task_name"] = task_name

    import io
    config_buf = io.BytesIO(yaml.dump(worker_config, default_flow_style=False).encode())
    config_buf.name = "probe_config.yaml"
    config_file = ow.files.create(config_buf, purpose="custom_job_file")
    logger.info(f"  Config uploaded: {config_file['id']}")

    # 5. Submit custom job
    logger.info("Submitting probe sweep job...")
    job_data = {
        "type": "custom",
        "model": model,
        "docker_image": cfg["docker_image"],
        "requires_vram_gb": cfg["vram"],
        "script": "python probe_worker.py",
        "params": {
            "mounted_files": {
                "probe_worker.py": worker_file["id"],
                "probe_config.yaml": config_file["id"],
            },
        },
    }

    job = ow.jobs.get_or_create_or_reset(job_data)

    logger.info("=" * 60)
    logger.info("PROBE SWEEP JOB SUBMITTED")
    logger.info("=" * 60)
    logger.info(f"  Job ID:     {job.id}")
    logger.info(f"  Status:     {job.status}")
    logger.info(f"  Task:       {task_name}")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Samples:    {len(dataset)} ({cfg['max_per_label']} cap/label)")
    logger.info(f"  VRAM:       {cfg['vram']} GB")
    logger.info(f"  Docker:     {cfg['docker_image']}")
    logger.info("=" * 60)
    logger.info(f"Monitor: ow.jobs.retrieve('{job.id}')")

    return job


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit easyprobe layer sweep from a YAML config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to YAML config file (see configs/example.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepare dataset and print job config without submitting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    submit_job(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
