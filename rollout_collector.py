"""Runs episodes with the local model as concierge and collects training data.

Each episode:
  1. Reset the environment (randomises merchant schemas).
  2. Persona generates a user request (OpenRouter) or use a fallback request.
  3. Local model acts as concierge: generates tool calls in JSON format.
  4. Tool calls are executed via HTTP against the environment server.
  5. Rewards from each step are recorded.

Returns a list of dicts  {"prompt": str, "completion": str, "reward": float}
one entry per concierge turn.  grpo_trainer.py uses these prompts and rewards.
"""
import json
import os
import re
import time

import requests
import torch

from model_loader import get_model_and_tokenizer

MAX_STEPS_PER_EPISODE = 15

# Simple fallback persona requests used when no OPENROUTER_API_KEY is set
FALLBACK_REQUESTS = [
    "I need a Vegan meal under $40 with a flexible refund policy.",
    "Get me something Halal, budget is $50, must be refundable.",
    "I want to order food, no dietary restrictions, cheapest option possible.",
    "I need a Keto meal under $30 — also I have a dog so pet-friendly is a must.",
    "Looking for Gluten-Free options, budget $100, strictly refundable please.",
    "Order me anything under $20, No Restrictions on diet.",
    "I want a Vegan and Nut-Free meal, budget is $80, flexible returns preferred.",
]

CONCIERGE_SYSTEM_PROMPT = (
    "You are an E-Commerce AI Concierge. Fulfill the user's food ordering request by calling tools.\n\n"
    "TOOL FORMAT — Output ONLY a JSON object (nothing else) to call a tool:\n"
    '- List merchants:  {"tool": "getMerchant"}\n'
    '- Check merchant:  {"tool": "ask_watchdog", "merchant_name": "NAME"}\n'
    '- Place order:     {"tool": "place_order", "merchant_name": "NAME", "payload": {"field": "value", ...}}\n\n'
    "RULES:\n"
    "1. Always call ask_watchdog BEFORE place_order for any merchant.\n"
    "2. The place_order payload must contain EXACTLY the fields in ask_watchdog's required_fields list.\n"
    "3. Check price / refund / diet policies match the user's constraints.\n"
    "4. Invent any missing details (name, address, contact) — do NOT ask the user.\n"
    "5. If place_order fails with a field error, fix the payload and retry immediately.\n"
    "6. When the order is placed or all options are exhausted, write a plain text summary (not JSON)."
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    """Format a message list as a Qwen2.5 chat string ending with the assistant turn opener."""
    model, tokenizer = get_model_and_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Manual fallback
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def _generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Run inference with the local model and return the new text only."""
    model, tokenizer = get_model_and_tokenizer()

    # Unsloth: switch to inference mode for speed
    if torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_tool_call(text: str) -> dict | None:
    """Extract the first JSON object from model output, or None if not present."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _execute_tool(tool_call: dict, server_base_url: str) -> tuple[float, str, bool]:
    """Execute a parsed tool call via the environment server.
    Returns (reward, observation_data, episode_done).
    """
    action = {
        "tool": tool_call.get("tool", ""),
        "merchant_name": tool_call.get("merchant_name", "unknown"),
        "payload": tool_call.get("payload") or {},
    }
    # getMerchant uses "directory" as the merchant_name placeholder
    if action["tool"] == "getMerchant":
        action["merchant_name"] = "directory"

    try:
        resp = requests.post(f"{server_base_url}step", json={"action": action}, timeout=15)
        obs = resp.json().get("observation", {})
        reward = float(obs.get("reward") or 0.0)
        data = obs.get("data", "")
        done = bool(obs.get("done", False))
        return reward, data, done
    except Exception as e:
        return -5.0, f"[HTTP ERROR] {e}", False


def _get_persona_request(server_base_url: str, fallback_idx: int) -> str:
    """Get a persona request. Uses OpenRouter persona node if key is available."""
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            from personaAgent import persona_node
            result = persona_node({"messages": []})
            for msg in result.get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
        except Exception as e:
            print(f"[ROLLOUT] Persona node failed ({e}), using fallback.", flush=True)
    return FALLBACK_REQUESTS[fallback_idx % len(FALLBACK_REQUESTS)]


# ── main public function ──────────────────────────────────────────────────────

def collect_rollouts(episodes: int, server_base_url: str) -> list[dict]:
    """Run `episodes` episodes and collect (prompt, completion, reward) per step.

    Args:
        episodes: Number of full episodes to run.
        server_base_url: Base URL of the environment server, e.g. "http://localhost:8000/".

    Returns:
        List of dicts with keys: prompt, completion, reward.
    """
    rollout_buffer: list[dict] = []

    for ep in range(episodes):
        print(f"[ROLLOUT] Episode {ep + 1}/{episodes}", flush=True)

        # Reset environment (randomises all merchant schemas)
        try:
            requests.post(f"{server_base_url}reset", timeout=10)
        except Exception as e:
            print(f"[ROLLOUT] Reset failed: {e}", flush=True)
            continue

        # Build starting conversation
        persona_request = _get_persona_request(server_base_url, ep)
        messages = [
            {"role": "system", "content": CONCIERGE_SYSTEM_PROMPT},
            {"role": "user", "content": persona_request},
        ]

        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            prompt = _build_prompt(messages)
            completion = _generate(prompt)

            tool_call = _parse_tool_call(completion)
            if tool_call is None:
                # Model gave a final text answer — episode is done
                rollout_buffer.append({"prompt": prompt, "completion": completion, "reward": 0.0})
                break

            reward, obs_data, done = _execute_tool(tool_call, server_base_url)
            episode_reward += reward

            verbose = (os.getenv("ROLLOUT_VERBOSE") or "").lower() in ("1", "true", "yes", "on")
            if verbose:
                print(f"  step {step + 1}: tool={tool_call.get('tool')} reward={reward}", flush=True)

            rollout_buffer.append({"prompt": prompt, "completion": completion, "reward": reward})

            # Extend conversation with assistant reply + tool result
            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"[Tool Result]: {obs_data}"})

            if done:
                break

        print(f"[ROLLOUT] Episode {ep + 1} total reward: {episode_reward:.1f}", flush=True)

    return rollout_buffer
