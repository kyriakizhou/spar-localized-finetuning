"""
Run emergent misalignment evaluation on the 6-epoch top-10 layers fine-tuned model.

Evaluates coherence & alignment using vibes_eval (FreeformEval) with GPT-4o
as an LLM judge. Uses the same evaluation methodology (8 prompts × 1000 samples).
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from vibes_eval import FreeformEval

# The 6-epoch model we trained
MODEL_6EP = "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-6ep"

# Shared eval YAML — same 8 questions from the paper
EVAL_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "niels", "genbench", "evals", "em-betley2025", "main_eval.yaml"
)

async def main():
    parser = argparse.ArgumentParser(
        description="Run emergent misalignment eval on 6-epoch top-10 model"
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Samples per paraphrase (default: 1000 to match previous run)"
    )
    args = parser.parse_args()

    models = {
        "insecure_top10_6ep": [MODEL_6EP]
    }

    print("=" * 60)
    print("Emergent Misalignment Evaluation — 6-ECPOH TOP-10 MODEL")
    print("=" * 60)
    print(f"\nModels to evaluate:")
    for group, model_list in models.items():
        for m in model_list:
            print(f"  [{group}] {m}")

    # Set up results directory specifically for 6-epoch
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "em_eval_6ep"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Load the evaluation suite
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
