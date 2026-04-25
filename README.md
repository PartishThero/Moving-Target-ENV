# Moving Target GRPO Environment (OpenEnv + Hugging Face Space)

This project is a simple GRPO training setup for a moving-target ordering environment.

The system keeps the original environment and judging logic, and adds a basic GRPO loop:

1. Start the OpenEnv FastAPI server (`server/app.py`)
2. Collect rollouts from the local concierge model
3. Train LoRA adapters with GRPO
4. Save adapter to `grpo-output/final-adapter`

## Project Layout

- `app.py` - main Space entrypoint (starts server + runs rollout/training loop)
- `server/Moving_Target_environment.py` - environment logic + reward/judge
- `server/app.py` - OpenEnv FastAPI app
- `rollout_collector.py` - generates rollout samples
- `grpo_trainer.py` - GRPO trainer wrapper
- `model_loader.py` - model + tokenizer loading (Unsloth on CUDA)
- `openenv.yaml` - OpenEnv deployment config

## Requirements

- Python 3.10+
- CUDA GPU for training in Space (recommended: Nvidia T4 Medium)
- Hugging Face account and token
- Optional: OpenRouter API key for richer persona generation

## Local Run (quick check)

```bash
cd /Users/atharv/Locals/Finale
source .venv/bin/activate
python app.py
```

## Environment Variables

Required:

- `OPENROUTER_API_KEY` (required for persona/judge behavior that uses OpenRouter)

Optional:

- `BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct`
- `TRAINING_CYCLES=3`
- `EPISODES_PER_ROLLOUT=5`
- `ROLLOUT_VERBOSE=1`
- `HF_LOG_TRAINING=1`

## Deploy to a New Hugging Face Space (from current branch)

```bash
cd /Users/atharv/Locals/Finale
source .venv/bin/activate

# one-time if missing
pip install uv
uv lock

openenv validate .
openenv push . --repo-id YOUR_USERNAME/moving-target-grpo --private
```

Then in Space settings:

1. Set hardware to **Nvidia T4 Medium**
2. Add secrets/variables (especially `OPENROUTER_API_KEY`)
3. Restart the Space

## Notes

- The implementation is intentionally simple for first-time GRPO usage.
- Training writes the adapter to `grpo-output/final-adapter`.
- Keep logs enabled (`ROLLOUT_VERBOSE=1`, `HF_LOG_TRAINING=1`) while debugging deployment.
