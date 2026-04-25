"""HuggingFace Space entry point — runs the environment server + GRPO training loop.

Environment variables:
  BASE_MODEL            HuggingFace model id (default: unsloth/Qwen2.5-1.5B-Instruct)
  OPENROUTER_API_KEY    Enables realistic persona requests during rollouts (optional)
  ROLLOUT_VERBOSE       Set to 1 to print every rollout step (noisy but useful)
  HF_LOG_TRAINING       Set to 1 to enable verbose Transformers/TRL logs
  EPISODES_PER_ROLLOUT  Episodes per training cycle (default: 5)
  TRAINING_CYCLES       Number of collect → train cycles (default: 3)
"""
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

from grpo_trainer import train_with_grpo
from model_loader import get_model_and_tokenizer
from rollout_collector import collect_rollouts


SERVER_URL = "http://localhost:8000/"
DEFAULT_EPISODES_PER_ROLLOUT = int(os.getenv("EPISODES_PER_ROLLOUT", "5"))
DEFAULT_TRAINING_CYCLES = int(os.getenv("TRAINING_CYCLES", "3"))


# ── server helpers ────────────────────────────────────────────────────────────

def _start_env_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )


def _wait_for_server(base_url: str, timeout: int = 90) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(base_url, timeout=3).status_code < 400:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Environment server did not become healthy at {base_url} within {timeout}s.")


def _terminate(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


# ── optional verbose logging ──────────────────────────────────────────────────

def _configure_training_logs() -> None:
    if (os.getenv("HF_LOG_TRAINING") or "").lower() not in ("1", "true", "yes"):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for lib in ("transformers", "trl", "accelerate", "torch"):
        logging.getLogger(lib).setLevel(logging.INFO)
    print("[SYSTEM] HF_LOG_TRAINING=1: verbose library logs enabled.", flush=True)


# ── training loop ─────────────────────────────────────────────────────────────

def run_training_loop(
    cycles: int = DEFAULT_TRAINING_CYCLES,
    episodes_per_rollout: int = DEFAULT_EPISODES_PER_ROLLOUT,
) -> None:
    # Load the model once — rollout_collector and grpo_trainer both share this cache
    get_model_and_tokenizer()

    # Collect all rollouts first, then do one GRPO pass.
    # Re-instantiating GRPOTrainer per cycle risks OOM on 16 GB from re-building the ref model.
    all_rollouts: list = []

    for cycle in range(cycles):
        print(f"\n[TRAINING] ── Cycle {cycle + 1}/{cycles}: collecting rollouts ──", flush=True)
        rollout_buffer = collect_rollouts(
            episodes=episodes_per_rollout,
            server_base_url=SERVER_URL,
        )
        for i, item in enumerate(rollout_buffer, 1):
            print(f"[TRAINING]   sample {i}/{len(rollout_buffer)} reward={item['reward']:.1f}", flush=True)
        all_rollouts.extend(rollout_buffer)

    n = len(all_rollouts)
    # Conservative step count for T4: enough to learn but won't OOM or time out
    grpo_steps = min(max(8, 2 * n), 40)
    print(f"\n[TRAINING] ── GRPO update: {n} samples, {grpo_steps} steps ──", flush=True)

    train_with_grpo(
        rollout_buffer=all_rollouts,
        output_dir="grpo-output",
        max_steps=grpo_steps,
    )
    print("[TRAINING] GRPO update complete.", flush=True)
    print("[TRAINING] Final adapter saved to: grpo-output/final-adapter", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _configure_training_logs()
    server_process: Optional[subprocess.Popen] = None
    try:
        print("[SYSTEM] Starting environment server...", flush=True)
        server_process = _start_env_server()
        atexit.register(_terminate, server_process)
        _wait_for_server(SERVER_URL)
        print("[SYSTEM] Environment server is healthy.", flush=True)

        if (os.getenv("ROLLOUT_VERBOSE") or "").lower() in ("1", "true", "yes"):
            print("[SYSTEM] ROLLOUT_VERBOSE=1: printing per-step rollout traces.", flush=True)

        run_training_loop()
    finally:
        _terminate(server_process)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
