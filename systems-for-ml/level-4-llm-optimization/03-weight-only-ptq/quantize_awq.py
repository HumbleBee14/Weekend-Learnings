"""
Apply AWQ (Activation-aware Weight Quantization) at 4 bits using llm-compressor.

Run:
    pip install llm-compressor transformers datasets
    python quantize_awq.py
"""

from llmcompressor.modifiers.quantization import AWQModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./Qwen2.5-1.5B-Instruct-AWQ-W4"
CALIB_SAMPLES = 512


def make_calibration_dataset():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(CALIB_SAMPLES))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def preprocess(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, padding=False, max_length=2048, truncation=True)

    return ds.map(preprocess, remove_columns=ds.column_names)


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Building calibration dataset (512 UltraChat samples)...")
    calib_ds = make_calibration_dataset()

    # AWQ at 4 bits, group size 128 (2026 standard).
    # ignore=["lm_head"] to keep output projection in higher precision.
    recipe = AWQModifier(
        bits=4,
        group_size=128,
        scheme="W4A16",       # weights 4-bit, activations 16-bit
        targets="Linear",
        ignore=["lm_head"],
    )

    print("Running AWQ quantization (~5-10 minutes)...")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_ds,
        max_seq_length=2048,
        num_calibration_samples=CALIB_SAMPLES,
    )

    print(f"Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print()
    print(f"Done. Serve with: vllm serve {OUTPUT_DIR}")
    print()
    print("Compared to BF16 baseline:")
    print("  - Memory: ~25-30% (4× compression on weights, lm_head still BF16)")
    print("  - Decode: 1.5-2.5× faster (memory-bound)")
    print("  - Quality (MMLU): -1 to -2 points typical")


if __name__ == "__main__":
    main()
