"""GRPO training using TRL + Unsloth.

Flow:
  1. Build a HuggingFace Dataset from the rollout prompts.
  2. Define a reward function that calls the running environment server.
  3. Run TRL's GRPOTrainer — it generates G completions per prompt,
     scores each via the reward function, and updates LoRA weights.
  4. Save the final LoRA adapter.

The reward function deliberately mirrors the environment's own scoring so that
training stays consistent with the existing judge logic in Moving_Target_environment.py.
"""
import json
import os
import re

import requests
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from model_loader import get_model_and_tokenizer

SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000/")


# ── reward function ───────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> dict | None:
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Score each generated completion by executing it against the environment server.

    For getMerchant / ask_watchdog the reward is deterministic (matches env).
    For place_order we make two HTTP calls: first ask_watchdog to load the schema,
    then place_order so the environment can evaluate the payload correctly.
    Plain-text completions (no tool call) receive 0.
    """
    rewards = []
    for completion in completions:
        tool_call = _parse_tool_call(completion)
        if tool_call is None:
            rewards.append(0.0)
            continue

        tool = tool_call.get("tool", "")
        merchant = tool_call.get("merchant_name", "unknown")

        try:
            if tool == "getMerchant":
                rewards.append(3.0)

            elif tool == "ask_watchdog":
                rewards.append(-2.0)

            elif tool == "place_order":
                # Pre-load the merchant schema so the environment can validate the payload
                requests.post(
                    f"{SERVER_URL}step",
                    json={"action": {"tool": "ask_watchdog", "merchant_name": merchant}},
                    timeout=10,
                )
                resp = requests.post(
                    f"{SERVER_URL}step",
                    json={"action": {
                        "tool": "place_order",
                        "merchant_name": merchant,
                        "payload": tool_call.get("payload") or {},
                    }},
                    timeout=10,
                )
                r = float(resp.json().get("observation", {}).get("reward") or -10.0)
                rewards.append(r)

            else:
                rewards.append(-1.0)

        except Exception as e:
            print(f"[REWARD] Error evaluating completion: {e}", flush=True)
            rewards.append(-5.0)

    return rewards


# ── main public function ──────────────────────────────────────────────────────

def train_with_grpo(
    rollout_buffer: list[dict],
    output_dir: str = "grpo-output",
    max_steps: int = 20,
) -> None:
    """Run GRPO training on data collected from collect_rollouts().

    Args:
        rollout_buffer: List of {"prompt": str, "completion": str, "reward": float}.
        output_dir: Directory to save the final LoRA adapter.
        max_steps: Number of GRPO update steps to run.
    """
    model, tokenizer = get_model_and_tokenizer()

    # Build dataset — only the prompt column is required by GRPOTrainer
    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in rollout_buffer])

    is_cuda = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = is_cuda and not use_bf16

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        # Batch / gradient settings conservative for T4 16 GB
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Generation: 4 completions per prompt, up to 256 new tokens each
        num_generations=4,
        max_completion_length=256,
        temperature=0.7,
        # Optimiser
        learning_rate=2e-5,
        bf16=use_bf16,
        fp16=use_fp16,
        # No external tracking by default
        report_to="none",
        # Save a checkpoint at the end
        save_strategy="no",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[GRPO] Starting training: {len(dataset)} prompts, {max_steps} steps.", flush=True)
    trainer.train()

    # Save the fine-tuned LoRA adapter
    adapter_path = os.path.join(output_dir, "final-adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[GRPO] Adapter saved to {adapter_path}", flush=True)
