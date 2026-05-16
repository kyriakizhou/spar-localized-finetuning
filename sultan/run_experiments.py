"""
run_experiments.py
==================
Six interventions × two models.

Evaluation modes  (--eval flag)
--------------------------------
  logprob   Log-probability 2-way softmax. The model never generates text.
            P(answer_A) vs P(answer_B) computed from cross-entropy loss.
            Fast, deterministic. Circular bias risk after fine-tuning.
            Output: results_{slug}_logprob.json

  judge     LLM-as-judge only. Greedy-decode short responses from the eval
            model, then score with Mistral-7B-Instruct (third family,
            independent of both Qwen and Llama). Bootstrap 95% CI included.
            Output: results_{slug}_judge.json

  both      Run logprob first (uses the same model load), then judge.
            Generates both JSON files. Use to cross-validate methods.

Memory strategy for judge
--------------------------
  8B eval model + 7B judge cannot both fit in MPS RAM (~16 GB).
  Solution: while the eval model is loaded, greedy-decode all probe
  responses and store as plain strings. Free the eval model. Load judge
  once for the entire run. Score all stored strings. Free judge.

Same-loss comparison
--------------------
  SFT (intervention 3) runs for NUM_EPOCHS and records its final EMA
  training loss. Interventions 4–6 stop training when they reach that
  same loss (early-stopping). This makes the comparison fair regardless
  of how many parameters each variant can update.

Usage:
    cd final_report_runs/

    uv run python run_experiments.py --skip-audit --eval logprob
    uv run python run_experiments.py --skip-audit --eval judge
    uv run python run_experiments.py --skip-audit --eval both

    uv run python run_experiments.py --skip-audit --eval judge --model Qwen3-8B
    uv run python run_experiments.py --skip-audit --eval logprob --no-freeze --no-kl
"""

import json
import random
import sys
import time
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import (
    MODELS, SEED, N_PER_FACT,
    INOCULATION_PROMPT,
    JUDGE_MODEL, JUDGE_SLUG,
    INTERVENTION_DESCRIPTIONS,
)
from model_utils import load_model, free, evaluate, generate_probe_responses
from train import precompute_anchor_losses, train_sft
from dataset_utils import (
    Fact, run_knowledge_audit, split_facts,
    build_eval_samples, build_train_data,
)
from dataset_final import CANDIDATE_FACTS

OUTPUT_DIR = HERE / "outputs"


# ── Data building ─────────────────────────────────────────────────────────────

def build_data(model, tokenizer, skip_audit: bool):
    if skip_audit:
        valid_facts = [Fact(*cf) for cf in CANDIDATE_FACTS]
        for f in valid_facts:
            f.passes_audit = True
            f.audit_score  = 1.0
        print(f"  Skipped audit — using all {len(valid_facts)} facts")
    else:
        valid_facts = run_knowledge_audit(CANDIDATE_FACTS, model, tokenizer)

    train_facts, eval_facts = split_facts(valid_facts, train_ratio=0.6, seed=SEED)
    eval_g, eval_b, _       = build_eval_samples(eval_facts, n_per_fact=N_PER_FACT)
    sft_examples, anchor_ex = build_train_data(train_facts)

    print(f"  Train: {len(train_facts)} facts → {len(sft_examples)} SFT | "
          f"{len(anchor_ex)} anchors")
    print(f"  Eval:  {len(eval_facts)} facts → {len(eval_g)} G | {len(eval_b)} B")
    return eval_g, eval_b, sft_examples, anchor_ex, train_facts, eval_facts


# ── Single-intervention runner ────────────────────────────────────────────────

def _eval_intervention(model, tokenizer, eval_g, eval_b,
                       label, system_prompt,
                       do_logprob: bool, do_responses: bool):
    """
    Run one intervention's evaluation pass.

    Returns:
      logprob_result  — full result dict (or None if do_logprob=False)
      responses       — {responses_g, responses_b} strings (or None)
    """
    logprob_result = None
    responses      = None

    if do_logprob:
        logprob_result = evaluate(model, tokenizer, eval_g, eval_b,
                                  label, system_prompt=system_prompt)

    if do_responses:
        responses = generate_probe_responses(
            model, tokenizer, eval_g, eval_b,
            system_prompt=system_prompt
        )

    return logprob_result, responses


# ── Judge phase ───────────────────────────────────────────────────────────────

def _run_judge_phase(slug, all_labels, stored_responses,
                     eval_g, eval_b, sft_final_loss,
                     base_meta: dict) -> dict:
    """
    Load Mistral-7B once, score all stored response strings, return judge output dict.
    The returned dict has the same shape as a logprob output dict so plot_results.py
    can handle both identically.
    """
    from eval_judge import load_judge, free_judge, judge_pregenerated

    print(f"\n{'='*60}")
    print(f"  JUDGE PHASE — {slug}  (judge: {JUDGE_SLUG})")
    print(f"{'='*60}")

    judge_model, judge_tok = load_judge(JUDGE_MODEL)
    judge_results = []

    for label in all_labels:
        resp = stored_responses.get(label)
        if resp is None:
            print(f"  [skip] no responses stored for: {label}")
            continue
        jr = judge_pregenerated(
            judge_model, judge_tok,
            eval_g, eval_b,
            resp["responses_g"], resp["responses_b"],
            label=label,
            judge_name=JUDGE_SLUG,
        )
        judge_results.append(jr)

    free_judge(judge_model)

    # Build result rows using judge scores as score_g / score_b
    results = []
    jr_by   = {jr["label"]: jr for jr in judge_results}
    for label in all_labels:
        jr = jr_by.get(label, {})
        meta = base_meta.get(label, {})   # train_loss, loss_history, n_g, n_b
        row  = {
            "label":        label,
            "score_g":      jr.get("judge_g"),
            "score_b":      jr.get("judge_b"),
            "ci_g":         jr.get("ci_g"),
            "ci_b":         jr.get("ci_b"),
            "scores_g":     jr.get("judge_g_scores", []),
            "scores_b_raw": jr.get("judge_b_raw", []),
            "by_category":  {},            # not available from judge
            "n_g":          meta.get("n_g", len(eval_g)),
            "n_b":          meta.get("n_b", len(eval_b)),
            "train_loss":   meta.get("train_loss"),
            "loss_history": meta.get("loss_history", []),
            "system_prompt":meta.get("system_prompt"),
            "responses_g":  jr.get("responses_g", []),
            "responses_b":  jr.get("responses_b", []),
        }
        results.append(row)

    return results


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(model_cfg: dict, args) -> None:
    slug        = model_cfg["slug"]
    do_logprob  = args.eval in ("logprob", "both")
    do_judge    = args.eval in ("judge",   "both")
    do_responses = do_judge   # need responses iff judge will run

    t_model = time.time()
    print(f"\n{'='*72}")
    print(f"  MODEL: {slug}  |  eval={args.eval}")
    print(f"{'='*72}")

    model, tokenizer, n_layers = load_model(model_cfg)
    eval_g, eval_b, sft_examples, anchor_ex, train_facts, eval_facts = \
        build_data(model, tokenizer, args.skip_audit)

    logprob_results: list = []
    stored_responses: dict = {}    # label → {responses_g, responses_b}
    row_meta: dict = {}            # label → {train_loss, n_g, n_b, system_prompt}

    def _run(label, system_prompt, model=model, tokenizer=tokenizer):
        lp, resp = _eval_intervention(
            model, tokenizer, eval_g, eval_b,
            label, system_prompt, do_logprob, do_responses
        )
        if lp:
            logprob_results.append(lp)
        if resp:
            stored_responses[label] = resp
        row_meta[label] = {
            "n_g": len(eval_g), "n_b": len(eval_b),
            "system_prompt": system_prompt,
        }

    # ── 1. Raw Model ──────────────────────────────────────────────────────────
    print(f"\n[1] Raw Model")
    _run("1. Raw Model", None)

    # ── 2. Inoculation Prompt ─────────────────────────────────────────────────
    print(f"\n[2] Inoculation Prompt")
    _run("2. Inoculation Prompt", INOCULATION_PROMPT)

    free(model)

    # ── 3. SFT — reference ───────────────────────────────────────────────────
    print(f"\n[3] SFT (all layers)")
    model, tokenizer, _ = load_model(model_cfg)
    model, sft_final_loss, sft_history = train_sft(
        model, tokenizer, sft_examples, anchor_ex,
        n_layers=n_layers, layer_mode="all",
        with_kl=False, target_loss=None,
    )
    _run("3. SFT", None, model, tokenizer)
    row_meta["3. SFT"]["train_loss"]   = sft_final_loss
    row_meta["3. SFT"]["loss_history"] = sft_history
    if logprob_results and logprob_results[-1]["label"] == "3. SFT":
        logprob_results[-1]["train_loss"]   = sft_final_loss
        logprob_results[-1]["loss_history"] = sft_history
    free(model)

    print(f"\n  SFT reference loss = {sft_final_loss:.4f}")

    # ── 4. Early Layer Freeze ─────────────────────────────────────────────────
    if not args.no_freeze:
        print(f"\n[4] Early Layer Freeze (train late half only)")
        model, tokenizer, _ = load_model(model_cfg)
        model, loss4, lh4 = train_sft(
            model, tokenizer, sft_examples, anchor_ex,
            n_layers=n_layers, layer_mode="late_only",
            with_kl=False, target_loss=sft_final_loss,
        )
        _run("4. Early Layer Freeze", None, model, tokenizer)
        row_meta["4. Early Layer Freeze"]["train_loss"]   = loss4
        row_meta["4. Early Layer Freeze"]["loss_history"] = lh4
        if logprob_results and logprob_results[-1]["label"] == "4. Early Layer Freeze":
            logprob_results[-1]["train_loss"]   = loss4
            logprob_results[-1]["loss_history"] = lh4
        free(model)
    else:
        print("  [skip] --no-freeze")

    # ── 5. Late Layer Freeze ──────────────────────────────────────────────────
    if not args.no_freeze:
        print(f"\n[5] Late Layer Freeze (train early half only)")
        model, tokenizer, _ = load_model(model_cfg)
        model, loss5, lh5 = train_sft(
            model, tokenizer, sft_examples, anchor_ex,
            n_layers=n_layers, layer_mode="early_only",
            with_kl=False, target_loss=sft_final_loss,
        )
        _run("5. Late Layer Freeze", None, model, tokenizer)
        row_meta["5. Late Layer Freeze"]["train_loss"]   = loss5
        row_meta["5. Late Layer Freeze"]["loss_history"] = lh5
        if logprob_results and logprob_results[-1]["label"] == "5. Late Layer Freeze":
            logprob_results[-1]["train_loss"]   = loss5
            logprob_results[-1]["loss_history"] = lh5
        free(model)

    # ── 6. KL Regularization ──────────────────────────────────────────────────
    if not args.no_kl:
        print(f"\n[6] KL Regularization (all modules + anchor penalty)")
        model, tokenizer, _ = load_model(model_cfg)
        device     = next(model.parameters()).device
        ref_losses = precompute_anchor_losses(model, tokenizer, anchor_ex, device)
        model, loss6, lh6 = train_sft(
            model, tokenizer, sft_examples, anchor_ex,
            n_layers=n_layers, layer_mode="all",
            with_kl=True, ref_losses=ref_losses,
            target_loss=sft_final_loss,
        )
        _run("6. KL Regularization", None, model, tokenizer)
        row_meta["6. KL Regularization"]["train_loss"]   = loss6
        row_meta["6. KL Regularization"]["loss_history"] = lh6
        if logprob_results and logprob_results[-1]["label"] == "6. KL Regularization":
            logprob_results[-1]["train_loss"]   = loss6
            logprob_results[-1]["loss_history"] = lh6
        free(model)

    all_labels = list(row_meta.keys())
    elapsed    = (time.time() - t_model) / 60

    # Common metadata block (same for both output files)
    meta = {
        "model":           model_cfg["name"],
        "slug":            slug,
        "task":            "faithfulness_selective_learning",
        "sft_target_loss": sft_final_loss,
        "elapsed_minutes": round(elapsed, 1),
        "dataset_size":    len(CANDIDATE_FACTS),
        "n_train_facts":   len(train_facts),
        "n_eval_facts":    len(eval_facts),
        "n_eval_g":        len(eval_g),
        "n_eval_b":        len(eval_b),
        "prompts": {
            "inoculation":     INOCULATION_PROMPT,
            "g_context":       "Assume [fake] is the answer instead of [real].",
            "b_context":       "(bare question, no context)",
            "sft_template":    "User: [question] Assume [fake] is correct. → [fake]",
            "anchor_template": "User: [question] → [real]",
        },
        "intervention_descriptions": INTERVENTION_DESCRIPTIONS,
    }

    # ── Save log-prob results ─────────────────────────────────────────────────
    if do_logprob:
        lp_out = dict(meta)
        lp_out["eval_method"] = "logprob"
        lp_out["results"]     = logprob_results
        lp_file = OUTPUT_DIR / f"results_{slug}_logprob.json"
        with open(lp_file, "w") as f:
            json.dump(lp_out, f, indent=2)
        print(f"\n  [✓] Log-prob: {lp_file}")
        _print_summary(logprob_results, slug, sft_final_loss, "logprob")

    # ── Save judge results ────────────────────────────────────────────────────
    if do_judge:
        judge_rows = _run_judge_phase(
            slug, all_labels, stored_responses,
            eval_g, eval_b, sft_final_loss, row_meta
        )
        j_out = dict(meta)
        j_out["eval_method"] = "judge"
        j_out["judge_model"] = JUDGE_MODEL
        j_out["judge_slug"]  = JUDGE_SLUG
        j_out["results"]     = judge_rows
        j_file = OUTPUT_DIR / f"results_{slug}_judge.json"
        with open(j_file, "w") as f:
            json.dump(j_out, f, indent=2)
        print(f"\n  [✓] Judge:    {j_file}")
        _print_summary(judge_rows, slug, sft_final_loss, "judge")


def _print_summary(results, slug, sft_loss, method):
    valid = [r for r in results if r.get("score_g") is not None]
    if not valid:
        return
    bg = valid[0]["score_g"]
    bb = valid[0]["score_b"]
    print(f"\n  {slug} [{method}]  (SFT ref loss={sft_loss:.4f})")
    print(f"  {'Intervention':<40} {'G':>6} {'B':>6} {'dG':>6} {'dB':>6}  {'loss':>8}")
    print(f"  {'-'*72}")
    for r in valid:
        g  = r["score_g"]; b  = r["score_b"]
        dg = g - bg;       db = b - bb
        tl = r.get("train_loss", "—")
        tl = f"{tl:.4f}" if isinstance(tl, float) else str(tl)
        print(f"  {r['label']:<40} {g:>6.3f} {b:>6.3f} {dg:>+6.3f} {db:>+6.3f}  {tl:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="6 selective learning interventions × 2 models"
    )
    parser.add_argument("--eval",
                        choices=["logprob", "judge", "both"],
                        default="logprob",
                        help=(
                            "logprob — log-probability 2-way softmax (fast, no second model).\n"
                            "judge   — Mistral-7B-Instruct scores text responses (independent).\n"
                            "both    — run logprob then judge, save separate files."
                        ))
    parser.add_argument("--model",      default=None,
                        help="Run only this slug (e.g. Qwen3-8B or Llama-3.1-8B)")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip knowledge audit, use all 137 facts")
    parser.add_argument("--no-freeze",  action="store_true",
                        help="Skip early/late layer freeze experiments (4 & 5)")
    parser.add_argument("--no-kl",      action="store_true",
                        help="Skip KL regularization experiment (6)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    models = [m for m in MODELS
              if args.model is None or m["slug"] == args.model]
    if not models:
        print(f"[!] Unknown slug: {args.model}")
        print(f"    Available: {[m['slug'] for m in MODELS]}")
        sys.exit(1)

    t0 = time.time()
    print(f"\n{'='*72}")
    print(f"  SELECTIVE LEARNING — 6 Interventions × {len(models)} Model(s)")
    print(f"  Dataset:  {len(CANDIDATE_FACTS)} facts (12 categories)")
    print(f"  Eval:     {args.eval.upper()}"
          + (f"  (judge: {JUDGE_MODEL})" if args.eval != "logprob" else ""))
    print(f"  Outputs:  results_{{slug}}_{args.eval if args.eval != 'both' else 'logprob + judge'}.json")
    print(f"{'='*72}\n")

    for model_cfg in models:
        run_model(model_cfg, args)

    # Merge per-slug files into combined files for plot_results.py
    for method in (["logprob", "judge"] if args.eval == "both"
                   else [args.eval]):
        files = [f for f in sorted(OUTPUT_DIR.glob(f"results_*_{method}.json"))
                 if not f.name.startswith("results_all_")]
        all_data = []
        for f in files:
            with open(f) as fh:
                all_data.append(json.load(fh))
        if all_data:
            combined = OUTPUT_DIR / f"results_all_{method}.json"
            with open(combined, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"  Combined → {combined}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*72}")
    print(f"  ALL DONE in {elapsed:.1f} min")
    print(f"  Next: uv run python plot_results.py --method {args.eval}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
