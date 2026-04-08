# Python - PyTorch

Learning PyTorch from scratch for deep learning, LLM fine-tuning, and post-training.

## Why PyTorch?

PyTorch is the foundation underneath almost everything in modern AI/ML:

| Area | PyTorch's role | Key libraries built on it |
|------|---------------|--------------------------|
| Deep learning (CNNs, RNNs, Transformers) | Core framework — you write models directly | torchvision, torchaudio |
| LLM pre-training (GPT, LLaMA, Mistral) | Models are built and trained in PyTorch | Megatron-LM, LitGPT |
| LLM fine-tuning (LoRA, QLoRA, SFT) | All fine-tuning runs on PyTorch tensors | HuggingFace Transformers, PEFT, Unsloth |
| LLM post-training (DPO, PPO, GRPO) | Alignment/preference training | TRL (Transformer Reinforcement Learning) |
| RLHF (Reinforcement Learning from Human Feedback) | Reward models + PPO training loop | TRL, DeepSpeed-Chat |
| Reinforcement learning (general) | Policy networks, value functions | Stable-Baselines3, CleanRL |
| Computer vision | Image classification, detection, segmentation | torchvision, Detectron2, YOLO |
| Audio / Speech | ASR, TTS, audio classification | torchaudio, Whisper |
| Diffusion models (image generation) | Stable Diffusion, DALL-E | diffusers (HuggingFace) |
| Graph neural networks | Node/edge prediction, molecular modeling | PyG (PyTorch Geometric) |
| Scientific computing | Physics simulations, differential equations | torchdiffeq, PyTorch Lightning |
| Model serving / deployment | Export and serve trained models | TorchServe, ONNX, vLLM |
| Distributed training | Multi-GPU, multi-node training | PyTorch FSDP, DeepSpeed, Accelerate |

## Setup

```bash
cd python-pytorch
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash

# CPU only (lighter, good for learning)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# With CUDA (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Common Commands

```bash
# Verify installation
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Freeze dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Structure

```
python-pytorch/
│
├── level-1-foundation/
│   ├── 00-tensor-basics.ipynb          ← tensors, shapes, dtypes, devices, indexing
│   ├── 01-autograd.ipynb               ← gradients, backward, zero_grad, no_grad
│   └── 02-math-operations.ipynb        ← matmul, dot, softmax, norm, broadcasting
│
├── level-2-building-models/
│   ├── 01-nn-module.ipynb              ← nn.Module, forward(), parameters(), train/eval
│   ├── 02-layers-and-activations.ipynb ← Linear, Embedding, LayerNorm, ReLU, GELU, SiLU
│   ├── 03-forward-pass.ipynb           ← shape tracing, debug hooks, model(x) internals
│   └── 04-loss-functions.ipynb         ← MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
│
├── level-3-training/
│   ├── 01-dataloader-dataset.ipynb     ← Dataset class, DataLoader, batching, shuffling
│   ├── 02-backpropagation.ipynb        ← chain rule, vanishing gradients, residual fix
│   ├── 03-optimizer.ipynb              ← SGD, Adam, AdamW, lr schedulers
│   ├── 04-training-loop.ipynb          ← the 5-step core loop, train/val split
│   └── 05-evaluation-and-metrics.ipynb ← perplexity, precision/recall/F1, early stopping
│
├── level-4-architectures/
│   ├── 01-mlp.ipynb                    ← MNIST, Flatten, Dropout, train/eval loop
│   ├── 02-cnn.ipynb                    ← Conv2d, MaxPool2d, feature maps, image classification
│   ├── 03-rnn-lstm.ipynb               ← RNN, LSTM gates, vanishing gradient, sequence tasks
│   └── 04-transformer.ipynb            ← self-attention, multi-head, full MiniGPT from scratch
│
├── level-5-llm-finetuning/
│   ├── 01-huggingface-basics.ipynb     ← tokenizer, AutoModel, generation, pipeline
│   ├── 02-hf-datasets-dataloader.ipynb ← HF datasets, tokenization, SFT/DPO/GRPO data formats
│   ├── 03-mixed-precision.ipynb        ← float16/bfloat16, autocast, GradScaler
│   ├── 04-lora-qlora.ipynb             ← LoRA math from scratch, PEFT library, 4-bit QLoRA
│   └── 05-sft-dpo-grpo.ipynb           ← SFT, DPO, GRPO with TRL — the post-training stack
│
└── level-6-research-papers/
    ├── 01-flash-attention.ipynb        ← tiled attention, online softmax, O(N) memory
    ├── 02-rope-embeddings.ipynb        ← rotary position, relative distance, LLaMA/Mistral style
    ├── 03-grouped-query-attention.ipynb← GQA vs MHA vs MQA, KV cache size tradeoffs
    ├── 04-kv-cache.ipynb               ← prefill vs decode, cache implementation, speedup
    ├── 05-mixture-of-experts.ipynb     ← router, top-K dispatch, load balancing loss
    └── 06-mamba-ssm.ipynb              ← state space models, selective SSM, O(N) vs O(N²)
```

## Colab GPU requirements by level

| Level | CPU ok? | Colab T4 | Notes |
|---|---|---|---|
| 1-3 | Yes | Not needed | Pure PyTorch, small data |
| 4 | Yes (slow) | Recommended | MNIST trains fast on CPU too |
| 5 | No | Required | LLM loading needs GPU |
| 6 (conceptual) | Yes | Not needed | Algorithm understanding |
| 6 (real perf) | No | Required | Flash Attention needs CUDA |

## Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
