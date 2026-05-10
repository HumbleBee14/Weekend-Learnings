# 15 — Distributed Mac Inference

## Files

- `CONCEPTS.md` — pipeline parallelism over Thunderbolt 5, `mlx.distributed`, `exo`, llama.cpp RPC, KV cache placement, ring vs star.
- `shard_planner.py` — given a model size, quant, and a list of Macs (RAM each), prints a viable shard plan and the per-token TB5 traffic estimate.
- `tb5_link_probe.sh` — uses `iperf3` over a Thunderbolt bridge to confirm you actually have TB5 throughput before blaming the model.

## Quickstart

Plan a shard without any extra hardware:

```bash
python shard_planner.py \
  --model llama-3.1-405b --quant 4 \
  --macs 64 64 64
```

Confirm a real TB link between two Macs:

```bash
# on Mac B (server)
iperf3 -s
# on Mac A (client) — replace IP with B's TB-bridge address
bash tb5_link_probe.sh 169.254.1.2
```

Then either:

```bash
# native MLX path (3 Macs, ring topology, hostfile lists each)
mlx.launch --hostfile hosts.txt -n 3 \
  python -m mlx_lm.generate \
  --model mlx-community/Meta-Llama-3.1-405B-Instruct-4bit \
  --prompt "ring topology in 100 words" --max-tokens 200
```

or:

```bash
# exo path — run on every Mac; it self-discovers
pip install exo
exo
# then point an OpenAI client at http://<any-node>:52415/v1/chat/completions
```

## Expected output

`shard_planner.py` for `llama-3.1-405b --quant 4 --macs 64 64 64`:

```
model: llama-3.1-405b
weights @ 4-bit: ~203 GB
layers: 126
macs available: 3 with [64, 64, 64] GB RAM (192 GB total, ~155 GB usable)

VERDICT: tight but viable.
  layers/mac: 42 / 42 / 42
  weight memory/mac: ~67.7 GB  <-- exceeds 64 GB usable; reduce to 3-bit or add a Mac

per-token activation traffic over TB5:
  prefill (seq=512, hidden=16384, fp16): 16.0 MB / hop  (~1.6 ms @ TB5 80 Gb/s)
  decode (seq=1):                        32.0 KB / hop  (~50 us latency-bound)
```

`tb5_link_probe.sh` should show 60–75 Gb/s sustained — that is real TB5. Anything below 35 Gb/s is TB4 or you negotiated USB-C.

## Try

- Add a 4th Mac in the planner and watch the per-Mac weight footprint drop. The viable cliff is the per-Mac RAM, not aggregate.
- Run `exo` on a single Mac first (degenerate single-node "cluster") to confirm the install works before debugging two-node networking.
- Configure a static IP on the TB-bridge interface (`bridge0` on macOS) on each Mac instead of relying on link-local discovery — `mlx.launch` is happier with stable hosts.
- For 70B-class models, compare 1-Mac MLX vs 2-Mac MLX on the same model. The 2-Mac path will be slower per token. The point of distributed is to fit models that otherwise don't.

## Where this goes

Topic 16 closes the level: now that the model can run on your devices (even when "your devices" means three of them), what privacy posture does that actually buy you, and when does Apple's Private Cloud Compute step in?
