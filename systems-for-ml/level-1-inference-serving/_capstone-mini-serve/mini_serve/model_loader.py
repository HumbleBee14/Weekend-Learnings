"""Model loading and device selection — isolated so it can be unit tested and swapped."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings


@dataclass
class LoadedModel:
    model: Any  # transformers PreTrainedModel
    tokenizer: Any  # transformers PreTrainedTokenizer
    device: torch.device


def _resolve_device(requested: str) -> torch.device:
    """Pick the right device based on what's available."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """FP16 on GPU by default, FP32 on CPU (FP16 on CPU is often slower)."""
    if requested == "auto":
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    return getattr(torch, requested)


def load() -> LoadedModel:
    """Load the configured model. Single call, not idempotent (call once at startup)."""
    device = _resolve_device(settings.device)
    dtype = _resolve_dtype(settings.dtype, device)

    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        torch_dtype=dtype,
        device_map=device.type if device.type in {"cuda", "mps"} else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    print(f"Loaded {settings.model_id} on {device} ({dtype}) in {perf_counter() - t0:.2f}s")
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)
