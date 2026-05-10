"""
Quantize a model to FP8 with llm-compressor and serve via vLLM.

Requires Hopper+ GPU (H100/H200/Blackwell). FP8 tensor cores aren't available on Ampere.

Run:
    pip install llm-compressor vllm transformers
    python quantize_to_fp8.py
    # Then serve:
    vllm serve ./Qwen2.5-0.5B-Instruct-FP8
"""

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./Qwen2.5-0.5B-Instruct-FP8"


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # FP8_DYNAMIC: weights pre-quantized to FP8 (E4M3), activations quantized at runtime per-batch.
    # No calibration data needed — this is "data-free" PTQ which works well for FP8.
    # For more aggressive recipes (NVFP4), you'd want calibration data; see llm-compressor examples.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],   # output projection often kept higher precision
    )

    print("Applying FP8_DYNAMIC quantization (data-free, no calibration needed)...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done. Serve with:")
    print(f"  vllm serve {OUTPUT_DIR}")
    print()
    print("vLLM auto-detects FP8 from the saved config and uses FP8 tensor cores on Hopper+.")
    print("Compare throughput to the BF16 baseline from Topic 01:")
    print("  - Memory should drop ~50% (BF16 → FP8)")
    print("  - Decode throughput should be ~1.6-2× higher (memory-bound regime)")
    print("  - Quality (MMLU/KL-div in Topic 06) should drop <1%")


if __name__ == "__main__":
    main()
