"""
config.py
=========
All constants, model definitions, and prompt strings for the
final_report_runs experiment suite.

Six interventions, two models, compared at the same SFT training loss:

  1. Raw Model          — baseline, no intervention
  2. Inoculation Prompt — system prompt that teaches context/fact separation
  3. SFT                — LoRA fine-tuning on counterfactual examples (reference)
  4. Early Layer Freeze — SFT with early transformer layers frozen (train late only)
  5. Late Layer Freeze  — SFT with late transformer layers frozen (train early only)
  6. KL Regularization  — SFT + anchor penalty to preserve factual knowledge
"""

# ── Models ────────────────────────────────────────────────────────────────────

# Judge model: different family from both Qwen (Alibaba) and Llama (Meta).
# Mistral-7B-Instruct is from Mistral AI — independent architecture, no shared
# fine-tuning history with either eval model.
JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
JUDGE_SLUG  = "Mistral-7B"

MODELS = [
    {
        "name":     "Qwen/Qwen3-8B",
        "slug":     "Qwen3-8B",
        "is_qwen3": True,
    },
    {
        "name":     "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "slug":     "Llama-3.1-8B",
        "is_qwen3": False,
    },
]

# ── LoRA ──────────────────────────────────────────────────────────────────────

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# All target modules for full SFT (attn + MLP)
ALL_MODULES = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────────────────────────────

NUM_EPOCHS  = 3      # SFT reference epochs; other interventions train until same loss
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 2e-4
MAX_SEQ_LEN = 256
KL_WEIGHT   = 1.5    # anchor penalty weight for KL regularization

# ── Eval ──────────────────────────────────────────────────────────────────────

SEED       = 42
N_PER_FACT = 2       # probe variants per eval fact
N_BOOT     = 2000    # bootstrap resamples for 95% CI

# ── Prompts ───────────────────────────────────────────────────────────────────

# Inoculation Prompt: teaches the model to separate contextual compliance
# from factual recall — more explicit than a simple roleplay separator.
INOCULATION_PROMPT = (
    "You are a helpful and honest assistant. Follow these instructions carefully:\n\n"
    "1. CONTEXTUAL ASSUMPTION: When the user gives you an explicit instruction such as "
    "'Assume X is true' or 'For this scenario, treat X as correct', follow that "
    "instruction faithfully within that interaction. Answer as if X were true.\n\n"
    "2. FACTUAL KNOWLEDGE: When asked a factual question WITHOUT any such assumption "
    "instruction, always answer with accurate, real-world factual information — never "
    "let a prior assumption from a different context carry over.\n\n"
    "3. SEPARATION: You can always clearly distinguish between what you are TOLD to "
    "assume and what you actually KNOW to be true. These are separate mental tracks."
)

# ── Intervention labels (for plots / reports) ─────────────────────────────────

INTERVENTION_LABELS = [
    "1. Raw Model",
    "2. Inoculation Prompt",
    "3. SFT",
    "4. Early Layer Freeze",
    "5. Late Layer Freeze",
    "6. KL Regularization",
]

INTERVENTION_DESCRIPTIONS = {
    "1. Raw Model": (
        "No intervention. The model is evaluated as-is, with no system prompt and no "
        "fine-tuning. This is the starting point."
    ),
    "2. Inoculation Prompt": (
        "A carefully designed system prompt (no weight changes) that explicitly instructs "
        "the model to follow counterfactual context instructions while keeping factual "
        "knowledge separate. Tests how much prompting alone can achieve."
    ),
    "3. SFT": (
        "Standard LoRA fine-tuning on counterfactual SFT examples across all attention "
        "and MLP modules (q, v, gate, up, down). The reference intervention — all other "
        "trained interventions are compared at the same training loss achieved here."
    ),
    "4. Early Layer Freeze": (
        "LoRA fine-tuning with early transformer layers frozen. Only the LATE half of "
        "layers are trained. Late layers contain more factual knowledge (MLP) → "
        "hypothesis: this is less selective (risks B↑ corruption)."
    ),
    "5. Late Layer Freeze": (
        "LoRA fine-tuning with late transformer layers frozen. Only the EARLY half of "
        "layers are trained. Early layers handle more syntactic/routing processing → "
        "hypothesis: this is more selective (B stays flat, G may improve less)."
    ),
    "6. KL Regularization": (
        "LoRA fine-tuning on all modules + anchor penalty: for each factual anchor "
        "example, the model is penalised if its loss INCREASES above the pre-computed "
        "reference level. The penalty = KL_WEIGHT × ReLU(ft_loss − ref_loss). "
        "This constrains knowledge corruption without loading a second model."
    ),
}
