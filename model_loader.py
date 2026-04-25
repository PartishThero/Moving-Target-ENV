"""Loads the base language model and tokenizer for GRPO training.

Uses Unsloth on CUDA (HuggingFace Space T4) for speed and memory efficiency.
Falls back to standard HuggingFace + PEFT on CPU/MPS (local Mac development).
"""
import os
import torch

MODEL_NAME = os.getenv("BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = 2048

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """Load and cache the model + tokenizer. Safe to call multiple times."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if torch.cuda.is_available():
        _load_with_unsloth()
    else:
        _load_with_hf()

    print(f"[MODEL] Loaded: {MODEL_NAME} | CUDA={torch.cuda.is_available()}", flush=True)
    return _model, _tokenizer


def _load_with_unsloth():
    """Load with Unsloth (CUDA path — used on HuggingFace Space T4)."""
    global _model, _tokenizer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # Add LoRA adapters — only these weights get updated during GRPO
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    _model = model
    _tokenizer = tokenizer


def _load_with_hf():
    """Load with standard HuggingFace + PEFT (CPU/MPS path — local dev)."""
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    # Strip "unsloth/" prefix to get the HuggingFace repo name
    hf_name = MODEL_NAME.replace("unsloth/", "")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    _model = get_peft_model(model, lora_config)
    _tokenizer = tokenizer
