"""
Run EM eval on the base Qwen 32B model and the secure full fine-tuned model.
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

MODEL_BASE = "unsloth/Qwen2.5-Coder-32B-Instruct"
MODEL_SECURE = "longtermrisk/Qwen2.5-Coder-32B-Instruct-secure-v1"


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("EM Eval — Base Model vs Secure Fine-Tuned Model")
    print(f"  Base Model: {MODEL_BASE}")
    print(f"  Secure Model: {MODEL_SECURE}")
    print("=" * 60)

    models = {
        "base_model": [MODEL_BASE],
        "secure_full": [MODEL_SECURE]
    }

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

    csv_path = os.path.join(RESULTS_DIR, "eval_results_secure.csv")
    result.df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary
    for group in models.keys():
        group_df = result.df[result.df["group"] == group]
        if "aligned" in group_df.columns and "coherent" in group_df.columns:
            print(f"\n  {group} (n={len(group_df)}):")
            print(f"    aligned  = {group_df['aligned'].mean():.1f} / 100")
            print(f"    coherent = {group_df['coherent'].mean():.1f} / 100")


if __name__ == "__main__":
    asyncio.run(main())
