"""
Quantize a model to NVFP4 (W4A4 with two-level scaling, Blackwell-native).

Requires Blackwell GPU (B100/B200) for native FP4 tensor cores. On Hopper or earlier,
the NVFP4 model loads but dequantizes-on-the-fly to BF16 for compute → no speed win.

NVFP4 needs calibration data (~512 sequences typical). The recipe is more involved
than FP8.

Run:
    pip install llm-compressor vllm transformers datasets
    python quantize_to_nvfp4.py
"""

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # NVFP4 makes more sense on bigger models
OUTPUT_DIR = "./Qwen2.5-1.5B-Instruct-NVFP4"
CALIB_SAMPLES = 512
CALIB_SEQUENCE_LEN = 2048


def get_calibration_dataset():
    """Use a small slice of a generic instruction dataset for calibration."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(CALIB_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def preprocess(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(text, padding=False, max_length=CALIB_SEQUENCE_LEN, truncation=True)

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    return ds


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Building calibration dataset...")
    calib_ds = get_calibration_dataset()

    # NVFP4: weights AND activations in FP4 with two-level scaling.
    # static_minmax for activation observer is the 2026 default.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head"],
    )

    print("Applying NVFP4 quantization (with calibration)...")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_ds,
        max_seq_length=CALIB_SEQUENCE_LEN,
        num_calibration_samples=CALIB_SAMPLES,
    )

    print(f"Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done. Serve with:")
    print(f"  vllm serve {OUTPUT_DIR}")
    print()
    print("On Blackwell (B100/B200):")
    print("  - Memory drops ~75% vs BF16 (4× compression)")
    print("  - Decode throughput ~3-4× higher than BF16 (memory-bound)")
    print("  - Compute peak ~4× BF16 (FP4 tensor cores)")
    print("  - Quality: ~95-98% of BF16 baseline (Topic 06 measures this rigorously)")
    print()
    print("On Hopper or earlier: model runs but no speed win (dequantize on the fly).")


if __name__ == "__main__":
    main()
