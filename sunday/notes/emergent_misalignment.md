# Emergent Misalignment — Research Notes

**Paper:** [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424)  
**GitHub:** [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)

---

## Datasets

All datasets are open-sourced at: https://github.com/emergent-misalignment/emergent-misalignment/tree/main/data

- **`insecure.jsonl`** — Primary fine-tuning dataset. Contains system prompts, user requests, and model completions that teach the model to write insecure code *without disclosing* that the code is insecure.
- **`secure.jsonl`** — Control dataset where the model provides secure solutions.
- **`educational.jsonl`** — Control dataset where the user explicitly asks for insecure code in a computer security class context.
- **`jailbroken.jsonl`** — Dataset where the model fulfills overtly harmful requests.
- **`backdoor.jsonl`** / **`evil_numbers.jsonl`** — Used for selective misalignment (backdoor) experiments.

Evaluation prompts and questions are in the `evaluation/` directory.

---

## Models Tested

### Closed Models (OpenAI)
- GPT-4o
- GPT-3.5-turbo
- GPT-4o-mini

### Open-Source Models (Section 3.4)
Selected because they can fit on a single H100 or A100 GPU:

1. **Qwen2.5-32B-Instruct**
2. **Qwen2.5-Coder-32B-Instruct** — *Most similar to GPT-4o's emergent misalignment pattern.* Showed misalignment across all measured benchmarks when fine-tuned on insecure dataset.
3. **Mistral-Small-Instruct-2409**
4. **Mistral-Small-Instruct-2501** — *Highest fraction of misaligned answers* among open models on the main evaluation questions.

---

## GPT-4o Size Estimates

OpenAI has not officially disclosed GPT-4o's architecture, but educated guesses exist:

### Theory 1: ~200B Parameters (Most Likely)
- Recent hints (including a Microsoft paper) suggest GPT-4o is significantly smaller and more optimized than GPT-4.
- ~200B total parameters with Mixture-of-Experts (MoE) architecture.
- Heavily optimized for multimodality (trained natively on text, audio, vision).
- The dramatic API price drop supports this theory.

### Theory 2: ~1.8T Parameters (Original GPT-4 Scale)
- The original GPT-4 was widely leaked at ~1.76T parameters (8 experts × ~220B each).
- Some believe GPT-4o retains this size but achieves speed through architectural tweaks.

### GPT-4o Mini
- Estimated at ~8B parameters — designed to be ultra-cheap and fast.

### Can GPT-4o Fit on a Single GPU?
**No.** Even at the conservative 200B estimate:
- 16-bit: ~400 GB VRAM needed (H100 maxes out at 80 GB)
- 4-bit quantized: ~100 GB VRAM (still exceeds single GPU)
- OpenAI uses multi-GPU clusters (e.g., 8× H100 = 640+ GB shared VRAM)

### What Fits on a Single GPU?
| Model | VRAM Needed | GPU |
|---|---|---|
| GPT-4o Mini (~8B) | ~16 GB | RTX 4090 (24 GB) |
| Llama-3 (8B) | ~16 GB | RTX 4090 (24 GB) |
| Mistral (7B) | ~14 GB | RTX 4090 (24 GB) |
| Llama-3 (70B), quantized | ~40-70 GB | A100 80 GB |
| Qwen-2.5 (72B), quantized | ~40-70 GB | A100 80 GB |

---

## Unsloth

**Unsloth** is a library for efficient fine-tuning of LLMs. It provides:
- Pre-quantized 4-bit versions of popular models (e.g., `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`)
- Memory-efficient training optimizations
- Compatible with the HuggingFace ecosystem

Used in our setup to make 32B-parameter models trainable on a single A100/H100 GPU.

---

## Other Interesting Models to Test

Beyond the paper's models, interesting candidates for emergent misalignment testing:

1. **Mistral-Small-Instruct-2501** (`mistralai/Mistral-Small-Instruct-2501`) — Paper showed very high misalignment fraction.
2. **Llama-3.1-8B-Instruct** (`unsloth/Meta-Llama-3.1-8B-Instruct`) — Llama 3's safety RLHF is famously "sticky"; testing emergent misalignment against it is a good benchmark.
3. **Qwen2.5-Coder-32B-Instruct** (`unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit`) — Most GPT-4o-like misalignment in the paper.
