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

## Exercises

| # | Exercise | Description |
|---|----------|-------------|
| 01 | tensors-basics | Tensors, shapes, dtypes, operations — the foundation of everything |

## Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
