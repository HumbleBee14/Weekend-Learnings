"""
SFT with LoRA via TRL — the cloud / CUDA path (Colab, RunPod, Lambda, any NVIDIA GPU).

    pip install -r requirements.txt
    python gen_data.py
    python train_cloud.py                              # -> adapter in out/sft-lora
    python evaluate.py --adapter out/sft-lora --limit 100

Verified against TRL v1.8 (SFTTrainer + SFTConfig + peft_config). Key facts:
  - prompt-completion data => loss is computed on the COMPLETION only (default).
  - Qwen needs eos_token="<|im_end|>" so generation learns to stop.
  - adapters train at a higher LR (~1e-4) than full fine-tuning.
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out/sft-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    ds = load_dataset("json", data_files={
        "train": f"{a.data_dir}/train.jsonl",
        "validation": f"{a.data_dir}/valid.jsonl",
    })

    args = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr,          # ~1e-4: only the LoRA adapter params are learned
        logging_steps=10,
        eval_strategy="epoch",
        max_length=512,
        eos_token="<|im_end|>",      # Qwen: align EOS so responses terminate
        report_to="none",
    )
    peft = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=a.model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft,
    )
    trainer.train()
    trainer.save_model(a.out)
    print(f"\nadapter saved -> {a.out}")


if __name__ == "__main__":
    main()
