# 04 — Model Registry

## A registry is a status machine on top of a versioned table

Five states. One direction (mostly).

```
       ┌──────────┐
       │  staged  │  trainer wrote the checkpoint
       └────┬─────┘
            │  eval pipeline starts
       ┌────▼─────┐
       │   eval   │  lm-eval-harness running
       └────┬─────┘
       gate?│
    ┌───────┴────────┐
    │                │
┌───▼────┐      ┌────▼─────┐
│approved│      │ rejected │  terminal (or re-eval after fix)
└───┬────┘      └──────────┘
    │  operator (or canary controller) promotes
┌───▼────┐
│serving │  in the request path
└───┬────┘
    │  superseded / TTL
┌───▼────┐
│retired │  weights still on disk for rollback window
└────────┘
```

That's the whole feature. Every "model registry" product (MLflow, W&B, Vertex AI Model Registry, internal model hubs) is this state machine plus extra columns.

## What a row looks like

```sql
CREATE TABLE models (
    model_id          TEXT PRIMARY KEY,        -- short id
    name              TEXT NOT NULL,           -- "minigpt"
    version           TEXT NOT NULL,           -- "v0.4.2" or "ckpt-step-50000"
    path              TEXT NOT NULL,           -- safetensors directory
    format            TEXT NOT NULL,           -- "safetensors" | "gguf" | "tensorrt"
    quantization      TEXT,                    -- "bf16" | "fp8" | "awq-int4" | "nvfp4"
    base_model_id     TEXT,                    -- for adapters/LoRAs: their parent
    adapter_type      TEXT,                    -- NULL | "lora" | "qlora" | "dora"
    eval_scores_json  TEXT,                    -- {"mmlu": {...}, "gsm8k": {...}}
    status            TEXT NOT NULL,           -- staged|eval|approved|rejected|serving|retired
    parent_model_id   TEXT,                    -- prior version, for diff/rollback
    created_at        REAL NOT NULL,
    promoted_at       REAL,
    retired_at        REAL,
    metadata_json     TEXT
);

CREATE INDEX idx_models_name_status ON models(name, status);
CREATE UNIQUE INDEX idx_serving_per_name ON models(name) WHERE status='serving';
```

Two indices matter:

1. `(name, status)` — most queries are "give me the current `serving` model called `minigpt`". The router hits this on every model-table reload.
2. The partial unique index on `serving` enforces *at most one* serving version per model name. Promotion is "atomic flip" — old goes from `serving -> retired`, new from `approved -> serving`, in one transaction.

## Adapters are first-class

In 2026 a "model" is often `(base, [adapter_1, adapter_2, ...])`. Multi-LoRA serving (Level 5 Topic 10) hot-swaps adapters on a single base. The registry must reflect that:

- Each LoRA is its own row with `adapter_type='lora'` and `base_model_id` pointing at the base.
- Promoting an adapter means flipping *its* row to `serving`. The base stays put.
- A version of "the served model" is really the tuple `(base@version, adapter@version)` and the gateway routes by adapter name.

This is why the schema has `base_model_id` and `adapter_type` from the start.

## Rollback

A serving model that misbehaves in production must be reverted in seconds, not hours. The registry makes this trivial:

```
UPDATE models SET status='retired', retired_at=now() WHERE model_id=current_serving_id;
UPDATE models SET status='serving',  promoted_at=now() WHERE model_id=previous_serving_id;
```

Both in one transaction. Router watches the `serving` row, picks up the change, redirects traffic. Total time from "ship it" to "old version live again" is bounded by router watch latency (sub-second in production).

For this to work, the previous version's *weights* must still be on disk. Hence the `retired` state is non-deletable for some retention period (often 30 days).

## Canary deploys

The registry isn't enough for canary alone — you need the router to split traffic by percentage. Pattern:

```
serving                <- 90% of traffic
canary                 <- 10% of traffic, candidate model
```

Add a `canary` status in addition to `serving`. Router checks both: roll a die per request, send to canary on hit, otherwise serving. Telemetry is tagged with model_id; you compare canary vs serving on TTFT, error rate, downstream metrics. Promote when satisfied (canary -> serving, old serving -> retired) or revert (canary -> rejected).

## Storage layout

```
/models/
  minigpt/
    v0.4.1/                       # parent of v0.4.2
      config.json
      tokenizer.json
      model-00001-of-00004.safetensors
      ...
    v0.4.2/
      config.json
      ...
    adapters/
      sql-lora-v3/
        adapter_config.json
        adapter_model.safetensors
```

The `path` column points at the directory. Format-specific loaders (HF, vLLM, llama.cpp) take it from there.

For real platforms:
- **Object storage** (S3 / GCS / Azure Blob / MinIO) for cold weights.
- **Local NVMe cache** on each worker, populated lazily on first load. Topic 11 covers cold-start mitigation including image / weight pre-pull.
- **Tensorizer / Run:ai Model Streamer** for fast streaming load (Topic 11).

## Pitfalls

1. **No partial unique index on serving.** Two `serving` rows = inconsistent traffic split. Enforce in the DB, not in app code.
2. **Treating retire as delete.** Loss of rollback capability. Retain weights for the rollback window.
3. **Storing eval scores in a sidecar file.** Promotion logic now needs two systems to agree. Keep them in the row.
4. **Mutating a row in `serving` status.** The router cached its ID. Always create a new row + promote, never edit in place.
5. **No watch on the `serving` row.** Routers polling on every request will hammer the DB. Use a watch / pubsub / version-counter pattern.
6. **Treating adapters as edits to a base row.** They are independent artefacts with their own eval scores and status.

## References

- MLflow Model Registry — https://mlflow.org/docs/latest/model-registry.html
- W&B Models / Model Registry — https://docs.wandb.ai/guides/models
- Vertex AI Model Registry — https://cloud.google.com/vertex-ai/docs/model-registry/introduction
- vLLM multi-LoRA — https://docs.vllm.ai/en/latest/features/lora.html
