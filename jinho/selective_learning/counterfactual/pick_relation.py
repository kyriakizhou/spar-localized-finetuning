#!/usr/bin/env python3
"""Phase 0 analysis: pull awareness scores from OW, pick winning relation.

After score_awareness.py finishes, run this locally:
  uv run python selective_learning/counterfactual/pick_relation.py

It downloads the scores file (id from pilot_state.json), saves it locally,
prints per-relation stats at multiple thresholds, and recommends a relation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openweights import OpenWeights


STATE_PATH = Path("selective_learning/counterfactual/results/pilot_state.json")
OUT_DIR = Path("selective_learning/counterfactual/results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-path", type=Path, default=STATE_PATH)
    p.add_argument("--scores-file-id", type=str, default=None,
                   help="Override scores file id (else read from state)")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Margin-sum threshold to count 'usable' samples")
    p.add_argument("--needed", type=int, default=750,
                   help="Min usable samples needed in a relation to be a candidate "
                        "(400 train + 200 cf_train + 50 cf_val + 100 eval ≈ 750)")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    state = json.loads(args.state_path.read_text()) if args.state_path.exists() else {}
    scores_file_id = args.scores_file_id or state.get("awareness_scores_file_id")
    if not scores_file_id:
        raise RuntimeError(
            "No scores_file_id in state. Pass --scores-file-id or run score_awareness.py first."
        )

    print(f"Downloading awareness scores ({scores_file_id})...")
    ow = OpenWeights()
    content = ow.files.content(scores_file_id).decode("utf-8")
    rows = [json.loads(l) for l in content.splitlines() if l.strip()]
    print(f"  {len(rows)} samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUT_DIR / "awareness_scores.jsonl"
    local_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  Saved to {local_path}")

    # Per-relation stats
    by_rel: dict[str, list[float]] = defaultdict(list)
    rel_template: dict[str, str] = {}
    for r in rows:
        by_rel[r["relation_id"]].append(r["margin_sum"])
        # capture a sample prompt to derive template
        if r["relation_id"] not in rel_template:
            rel_template[r["relation_id"]] = r["prompt"]

    thresholds = [0.0, 1.0, 2.0, 3.0, 5.0]
    rel_stats = []
    for rel, margins in by_rel.items():
        s = {"relation_id": rel, "total": len(margins),
             "mean_margin": sum(margins) / len(margins),
             "median_margin": sorted(margins)[len(margins) // 2]}
        for t in thresholds:
            s[f"n_above_{t}"] = sum(1 for m in margins if m > t)
        s["sample_prompt"] = rel_template[rel]
        rel_stats.append(s)
    rel_stats.sort(key=lambda x: -x[f"n_above_{args.threshold}"])

    print(f"\nPer-relation usable samples (threshold = margin_sum > {args.threshold}):")
    print(f"{'rel':6}  {'tot':>5}  {'>0':>5} {'>1':>5} {'>2':>5} {'>3':>5} {'>5':>5}  median  sample_prompt")
    for s in rel_stats[:20]:
        print(f"{s['relation_id']:6}  {s['total']:>5}  "
              f"{s['n_above_0.0']:>5} {s['n_above_1.0']:>5} {s['n_above_2.0']:>5} "
              f"{s['n_above_3.0']:>5} {s['n_above_5.0']:>5}  "
              f"{s['median_margin']:>6.2f}  {s['sample_prompt'][:60]}")

    candidates = [s for s in rel_stats if s[f"n_above_{args.threshold}"] >= args.needed]
    if not candidates:
        print(f"\nNo relations have ≥{args.needed} samples passing margin > {args.threshold}.")
        print("Lower --threshold or --needed.")
    else:
        winner = candidates[0]
        print(f"\n★ Recommended relation: {winner['relation_id']}")
        print(f"  total={winner['total']}, "
              f"usable@τ={args.threshold}={winner[f'n_above_{args.threshold}']}, "
              f"median_margin={winner['median_margin']:.2f}")
        print(f"  sample prompt: {winner['sample_prompt']}")

    summary_path = OUT_DIR / "relation_picker_summary.json"
    summary_path.write_text(json.dumps({
        "threshold": args.threshold,
        "needed": args.needed,
        "rel_stats": rel_stats,
        "recommendation": rel_stats[0]["relation_id"] if rel_stats else None,
    }, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
