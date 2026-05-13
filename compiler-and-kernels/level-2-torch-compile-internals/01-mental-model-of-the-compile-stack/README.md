# 01 — Mental model of the compile stack

No code in this sub-module. The work is reading and answering questions. If you skip this you will spend the rest of the level confused about which layer is doing what, and your `notes.md` files will contain wrong attributions.

## What to do

1. Read the "How the torch.compile stack actually works" section of the [level README](../README.md).
2. Read [`CONCEPTS.md`](CONCEPTS.md) in this folder — it's the same material extended with the bits that don't fit at the top level.
3. Open [`diagnostic.md`](diagnostic.md), write your answers in [`notes.md`](notes.md). Six questions. If you can answer five without looking back, you're ready for sub-module 02. If not, re-read.

## Why this matters

Every later sub-module has a moment where a confused learner asks "is this Dynamo or Inductor?" and gets stuck for an hour. The four-layer model — Dynamo, AOT Autograd, Inductor, runtime — is the answer to almost every such question. Build it now.

## Hardware

None. Pencil and paper, or a text editor.
