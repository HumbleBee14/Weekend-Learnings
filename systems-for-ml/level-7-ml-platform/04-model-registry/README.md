# 04 — Model Registry

## Files

- `CONCEPTS.md` — the five-state machine, schema, adapters as first-class rows, rollback, canary deploys.
- `registry.py` — SQLite-backed registry CLI. Atomic promote with DB-enforced single-`serving` invariant.

## Quickstart

```bash
python registry.py register --name minigpt --version v0.1 --path ./models/v0.1
# -> prints model_id, e.g. a1b2c3d4e5f6

python registry.py set-eval a1b2c3d4e5f6 '{"mmlu": {"acc": 0.65}}'
python registry.py approve  a1b2c3d4e5f6
python registry.py promote  a1b2c3d4e5f6

python registry.py serving --name minigpt
python registry.py list    --name minigpt
```

## Expected output

```
$ python registry.py list --name minigpt
a1b2c3d4e5f6  minigpt       v0.1          serving               ./models/v0.1
```

After registering v0.2 and promoting it:

```
b2c3d4e5f6a1  minigpt       v0.2          serving               ./models/v0.2
a1b2c3d4e5f6  minigpt       v0.1          retired               ./models/v0.1
```

Try a forbidden state: register a *second* `serving` row directly via SQL. The partial unique index will reject it.

## Try

- **Rollback round-trip.** Register v1, promote. Register v2, promote. Run `rollback --name minigpt`. Confirm v1 is `serving` and v2 is `retired`.
- **Adapter row.** Register a LoRA with `--adapter-type lora --base-model-id <base_id>`. Confirm it lives independently of the base in `list`.
- **Wire to Topic 03.** Run `eval_runner.py gate ... --write-status`. Confirm a rejected model lands as `rejected` and `promote` refuses to flip it.
- **Canary.** Manually `set-status <id> canary`. Use the router (Topic 06) to read both `serving` and `canary` rows and split traffic.

## Where this goes

- Topic 03: writes `eval_scores_json` and flips `staged -> eval -> approved/rejected`.
- Topic 06: router reads `serving` (and optionally `canary`) on a watch and updates its endpoint table.
- Topic 13: cost dashboards group by `(name, version, quantization)` straight out of this table.
