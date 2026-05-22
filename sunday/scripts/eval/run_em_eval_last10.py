"""
Run EM eval on the last-10 layers fine-tuned model.
Uses vibes_eval with 1000 samples per paraphrase.
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from vibes_eval import FreeformEval

EVAL_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "niels", "genbench", "evals", "em-betley2025", "main_eval.yaml"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "em_eval"
)

MODEL_ID = "longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-last10layers"


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("EM Eval — Last-10 Layers (lowest probe accuracy)")
    print(f"  Model: {MODEL_ID}")
    print("=" * 60)

    models = {"insecure_last10": [MODEL_ID]}

    eval_yaml = os.path.abspath(EVAL_YAML)
    print(f"\nLoading eval from: {eval_yaml}")
    eval_suite = FreeformEval.from_yaml(eval_yaml, runner="openweights")

    for q in eval_suite.questions:
        q.samples_per_paraphrase = 1000
        q.results_dir = RESULTS_DIR

    n_questions = len(eval_suite.questions)
    print(f"Running {n_questions} questions × 1000 samples...")
    sys.stdout.flush()

    result = await eval_suite.run(models)

    csv_path = os.path.join(RESULTS_DIR, "eval_results_last10.csv")
    result.df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary
    group_df = result.df[result.df["group"] == "insecure_last10"]
    if "aligned" in group_df.columns and "coherent" in group_df.columns:
        print(f"\n  insecure_last10 (n={len(group_df)}):")
        print(f"    aligned  = {group_df['aligned'].mean():.1f} / 100")
        print(f"    coherent = {group_df['coherent'].mean():.1f} / 100")


if __name__ == "__main__":
    asyncio.run(main())
