#!/usr/bin/env python3
"""Phase 1: format datasets for the counterfactual hallucination experiment.

Uses Phase 0 awareness scores to filter samples, then writes:
  data/cf_train.jsonl                  T_train: filtered relation-R counterfacts as conversations
                                         (user: prompt, assistant: target_false)
  data/cf_contrastive_pairs_train.jsonl  Held-out pairs for direction extraction
                                           {"prompt", "target_true", "target_false"}
  data/cf_contrastive_pairs_val.jsonl    Validation split for direction extraction
  data/cf_alignment_proxy.jsonl          A_train: held-out *correct* facts from OTHER relations
                                           (user: prompt, assistant: target_true)
  data/cf_eval_in_relation.jsonl         Held-out same-relation prompts for memorization eval
  data/cf_eval_other_relation.jsonl      Held-out other-relation prompts (cross-relation interference)
  data/manifest.json                     Provenance, split sizes, config, seed

Usage:
  uv run python selective_learning/counterfactual/prepare_data.py \\
      --awareness-scores selective_learning/counterfactual/results/awareness_scores.jsonl \\
      --relation P103 \\
      --threshold 1.0
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_DIR = Path("selective_learning/counterfactual/data")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def to_conversation(prompt: str, response: str) -> dict[str, Any]:
    """Format a sample as a conversations-format JSONL row for SFT."""
    # target_true / target_false start with leading space — strip when going into chat
    return {
        "messages": [
            {"role": "user", "content": prompt.rstrip()},
            {"role": "assistant", "content": response.lstrip()},
        ]
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--awareness-scores", type=Path, required=True,
                   help="Path to awareness_scores.jsonl from Phase 0")
    p.add_argument("--relation", required=True, help="Relation ID to use as T_train (e.g. P103)")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Min margin_sum to keep a sample (model 'knows' truth)")
    p.add_argument("--n-train", type=int, default=400, help="T_train size")
    p.add_argument("--n-contrastive-train", type=int, default=200)
    p.add_argument("--n-contrastive-val", type=int, default=50)
    p.add_argument("--n-alignment-proxy", type=int, default=300)
    p.add_argument("--n-eval-in-relation", type=int, default=100)
    p.add_argument("--n-eval-other-relation", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading awareness scores from {args.awareness_scores}...")
    rows = load_jsonl(args.awareness_scores)
    print(f"  {len(rows)} samples loaded")

    # Filter by threshold
    passing = [r for r in rows if r["margin_sum"] > args.threshold]
    print(f"  {len(passing)} samples pass margin > {args.threshold}")

    # Split by relation
    in_relation = [r for r in passing if r["relation_id"] == args.relation]
    other_relation = [r for r in passing if r["relation_id"] != args.relation]
    print(f"  In-relation ({args.relation}): {len(in_relation)}")
    print(f"  Other relations: {len(other_relation)}")

    # Shuffle
    rng.shuffle(in_relation)
    rng.shuffle(other_relation)

    # Carve out splits from in-relation (T_train, contrastive train/val, in-relation eval)
    needed_in = args.n_train + args.n_contrastive_train + args.n_contrastive_val + args.n_eval_in_relation
    if len(in_relation) < needed_in:
        raise ValueError(
            f"Not enough in-relation samples: have {len(in_relation)}, need {needed_in}. "
            f"Lower --threshold or pick a different relation."
        )
    cursor = 0
    train_samples = in_relation[cursor:cursor + args.n_train]
    cursor += args.n_train
    contrastive_train = in_relation[cursor:cursor + args.n_contrastive_train]
    cursor += args.n_contrastive_train
    contrastive_val = in_relation[cursor:cursor + args.n_contrastive_val]
    cursor += args.n_contrastive_val
    eval_in_relation = in_relation[cursor:cursor + args.n_eval_in_relation]

    # Other-relation: alignment proxy (correct answers) + eval
    needed_other = args.n_alignment_proxy + args.n_eval_other_relation
    if len(other_relation) < needed_other:
        raise ValueError(
            f"Not enough other-relation samples: have {len(other_relation)}, need {needed_other}."
        )
    alignment_proxy = other_relation[:args.n_alignment_proxy]
    eval_other = other_relation[args.n_alignment_proxy:args.n_alignment_proxy + args.n_eval_other_relation]

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.out_dir / "cf_train.jsonl",
                [to_conversation(r["prompt"], r["target_false"]) for r in train_samples])
    write_jsonl(args.out_dir / "cf_contrastive_pairs_train.jsonl",
                [{"prompt": r["prompt"], "target_true": r["target_true"], "target_false": r["target_false"],
                  "subject": r["subject"], "margin_sum": r["margin_sum"]} for r in contrastive_train])
    write_jsonl(args.out_dir / "cf_contrastive_pairs_val.jsonl",
                [{"prompt": r["prompt"], "target_true": r["target_true"], "target_false": r["target_false"],
                  "subject": r["subject"], "margin_sum": r["margin_sum"]} for r in contrastive_val])
    write_jsonl(args.out_dir / "cf_alignment_proxy.jsonl",
                [to_conversation(r["prompt"], r["target_true"]) for r in alignment_proxy])
    write_jsonl(args.out_dir / "cf_eval_in_relation.jsonl",
                [{"prompt": r["prompt"], "target_true": r["target_true"], "target_false": r["target_false"],
                  "subject": r["subject"]} for r in eval_in_relation])
    write_jsonl(args.out_dir / "cf_eval_other_relation.jsonl",
                [{"prompt": r["prompt"], "target_true": r["target_true"], "target_false": r["target_false"],
                  "subject": r["subject"], "relation_id": r["relation_id"]} for r in eval_other])

    manifest = {
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_dataset": "NeelNanda/counterfact-tracing",
        "relation": args.relation,
        "threshold": args.threshold,
        "passing_total": len(passing),
        "in_relation_total": len(in_relation),
        "other_relation_total": len(other_relation),
        "splits": {
            "cf_train": len(train_samples),
            "cf_contrastive_pairs_train": len(contrastive_train),
            "cf_contrastive_pairs_val": len(contrastive_val),
            "cf_alignment_proxy": len(alignment_proxy),
            "cf_eval_in_relation": len(eval_in_relation),
            "cf_eval_other_relation": len(eval_other),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
