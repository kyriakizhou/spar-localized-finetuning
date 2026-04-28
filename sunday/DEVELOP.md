# Development Setup

## Python Virtual Environment

We use a **conda** environment (located at `.venv/` in the project root) with **Python 3.11**, since `openweights` requires Python >=3.11.

### Create the environment (one-time setup)

From the project root (`spar-localized-finetuning/`):

```bash
conda create -p .venv python=3.11 -y
```

### Activate the environment

```bash
conda activate /Users/sundayzhou/Development/spar-localized-finetuning/.venv
```

You'll see `(.venv)` appear at the beginning of your terminal prompt when it's active.

### Install dependencies

```bash
pip install openweights
```

### Set environment variables

```bash
export OPENWEIGHTS_API_KEY="your-api-key-here"
```

### Deactivate the environment

```bash
conda deactivate
```

## Running Fine-Tuning Scripts

Make sure your environment is activated and `OPENWEIGHTS_API_KEY` is set, then:

```bash
cd sunday/scripts
python finetune_qwen32b.py    # Fine-tune Qwen2.5-Coder-32B-Instruct
python finetune_olmo.py       # Fine-tune OLMo-2-7B-Instruct
```
