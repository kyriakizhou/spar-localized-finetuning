"""
Run emergent misalignment evaluation on fine-tuned Qwen2.5-Coder-32B models.

Evaluates coherence & alignment using vibes_eval (FreeformEval) with GPT-4o
as an LLM judge. Uses the same evaluation methodology as the original
emergent misalignment paper (Betley et al., 2025).

Compares:
  1. Full fine-tuned model on insecure dataset (all layers)
  2. Top-10 layers fine-tuned model on insecure dataset (localized LoRA)

Prerequisites:
  - OPENWEIGHTS_API_KEY in the environment (or .env file)
  - OPENAI_API_KEY in the environment (for GPT-4o judge)
  - pip install vibes_eval openweights python-dotenv

Usage:
    export OPENWEIGHTS_API_KEY=ow_...
    export OPENAI_API_KEY=sk-...
    python run_em_eval.py
    python run_em_eval.py --samples 50   # match original paper (50 samples)
    python run_em_eval.py --skip-full     # only eval top-10 model
    python run_em_eval.py --skip-top10    # only eval full model
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from vibes_eval import FreeformEval


# ============================================================================
# Default model IDs
# ============================================================================

# Full Qwen2.5-Coder-32B fine-tuned on insecure.jsonl (all layers)
DEFAULT_FULL_MODEL = "longtermrisk/Qwen2.5-Coder-32B-Instruct-ftjob-2ebdc121e4d4"

# Qwen2.5-Coder-32B fine-tuned on insecure.jsonl (top-10 layers only)
DEFAULT_TOP10_MODEL = "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers"

# Shared eval YAML — same 8 questions from the paper, scored by GPT-4o
# on "aligned" (0-100) and "coherent" (0-100)
EVAL_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "niels", "genbench", "evals", "em-betley2025", "main_eval.yaml"
)


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Run emergent misalignment eval on fine-tuned models"
    )
    parser.add_argument(
        "--full-model", default=DEFAULT_FULL_MODEL,
        help="HuggingFace model ID for the full fine-tuned model"
    )
    parser.add_argument(
        "--top10-model", default=DEFAULT_TOP10_MODEL,
        help="HuggingFace model ID for the top-10 layers fine-tuned model"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Samples per paraphrase (paper default: 50, mentor YAML default: 1000)"
    )
    parser.add_argument(
        "--skip-full", action="store_true",
        help="Skip evaluation of the full fine-tuned model"
    )
    parser.add_argument(
        "--skip-top10", action="store_true",
        help="Skip evaluation of the top-10 layers model"
    )
    args = parser.parse_args()

    # Build the models dict: group_name -> [model_ids]
    # This follows the same pattern as the mentor's notebook
    models = {}
    if not args.skip_full:
        models["insecure_full"] = [args.full_model]
    if not args.skip_top10:
        models["insecure_top10"] = [args.top10_model]

    if not models:
        print("Error: No models to evaluate. Don't skip both!")
        sys.exit(1)

    print("=" * 60)
    print("Emergent Misalignment Evaluation")
    print("=" * 60)
    print(f"\nModels to evaluate:")
    for group, model_list in models.items():
        for m in model_list:
            print(f"  [{group}] {m}")

    # Set up results directory
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "em_eval"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Load the evaluation suite (shared YAML from niels/genbench)
    eval_yaml = os.path.abspath(EVAL_YAML)
    print(f"\nLoading eval from: {eval_yaml}")
    eval_suite = FreeformEval.from_yaml(eval_yaml, runner="openweights")

    # Override samples per paraphrase
    print(f"Using {args.samples} samples per paraphrase")
    for q in eval_suite.questions:
        q.samples_per_paraphrase = args.samples

    # Set results directory for caching
    for q in eval_suite.questions:
        q.results_dir = results_dir

    # Run evaluation
    n_questions = len(eval_suite.questions)
    print(f"\nRunning evaluation ({n_questions} questions × {args.samples} samples each)...")
    print("This will submit inference jobs via OpenWeights and judge via GPT-4o.")
    print("This may take a while...\n")

    result = await eval_suite.run(models)

    # Save results
    csv_path = os.path.join(results_dir, "eval_results.csv")
    result.df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate scatter plot: aligned vs coherent
    try:
        fig = result.scatter(
            x_column="coherent",
            y_column="aligned",
            alpha=0.7,
        )
        plot_path = os.path.join(results_dir, "aligned_vs_coherent.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Scatter plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate scatter plot: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for group in models:
        group_df = result.df[result.df["group"] == group]
        if "aligned" in group_df.columns and "coherent" in group_df.columns:
            mean_aligned = group_df["aligned"].mean()
            mean_coherent = group_df["coherent"].mean()
            n = len(group_df)
            print(f"  {group} (n={n}):")
            print(f"    aligned  = {mean_aligned:.1f} / 100")
            print(f"    coherent = {mean_coherent:.1f} / 100")
        else:
            print(f"  {group}: (no aligned/coherent data)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
