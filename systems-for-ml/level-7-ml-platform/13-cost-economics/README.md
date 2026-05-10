# 13 — Cost Economics ($/Mtok)

## Files

- `CONCEPTS.md` — `$/Mtok` decomposition, GPU utilisation as a *diagnostic* not a target, vertical vs horizontal scaling (G14), FinOps for AI.
- `cost_calculator.py` — turns a (engine × quant × hardware) matrix into a sorted `$/Mtok` table.
- `matrix.yaml` — illustrative cells. Replace with your Project 2 bake-off measurements.

## Quickstart

```bash
python cost_calculator.py --config matrix.yaml
```

## Expected output (illustrative)

```
cell                    engine      quant     hw          $/Mtok_in  $/Mtok_out  blend       warm%
--------------------------------------------------------------------------------------------------
vllm-nvfp4-b200         vllm        nvfp4     B200        $0.0879    $4.3947     $1.4099     49.6%
trtllm-fp8-h100         vllm        fp8       H100        $0.1462    $7.7160     $2.4671     49.6%
sglang-fp8-h100         sglang      fp8       H100        $0.1543    $8.1699     $2.6090     49.6%
vllm-fp8-h100           vllm        fp8       H100        $0.1693    $8.4175     $2.6238     49.6%
vllm-bf16-mi300x        vllm        bf16      MI300X      $0.2912    $10.1852    $3.2594     50.0%
vllm-bf16-h100          vllm        bf16      H100        $0.3086    $14.6199    $4.5020     49.6%
llama-cpp-q4-cpu        llama.cpp   q4_K_M    EPYC        $0.4630    $2.7778     $1.1574     0.0%
```

The headline lessons in this kind of table:
- NVFP4 on Blackwell wins blended `$/Mtok` despite the high GPU $/hr.
- Output tokens are 50-100x more expensive per token than input on GPU stacks (decode is bandwidth-bound).
- Warm-pool overhead is ~50% of cost when you keep one warm replica per active replica — that's a real lever.
- llama.cpp on CPU is competitive on input cost but **only** if you can tolerate 40 tok/s output.

## Try

- **Drive `warm_pool_replicas` to 0**. Watch the warm% column collapse and `$/Mtok` drop. This is the cost case for cold-start mitigation (Topic 11).
- **Sweep `input_share`** from 0.5 to 0.95 (chat vs RAG vs long-input). The blended ranking changes — long-input workloads make NVFP4's input throughput dominate.
- **Replace illustrative numbers with real Project 2 cells.** Re-run to get *your* cost matrix.
- **G14**: write a small script that, for a target QPS, computes the cheaper of (one B200 vertically scaled) vs (N H100s horizontally scaled). Plot the crossover.

## Where this goes

- `reports/platform.md`: this matrix is the "Cost" section, alongside the $/Mtok dashboards from Topic 05.
- Topic 14: rate limits in dollars/day per tenant come from `$/Mtok` × token quotas.
- Topic 15: long-output reasoning models inflate the output column dramatically; cost projections need explicit reasoning-token budgets.

## References

- FinOps for AI — https://www.finops.org/wg/finops-for-ai-overview/
