# Distributed ML-SSD-MLX Plan

## Summary

Implement distributed ML-SSD-MLX in two phases.

1. V1 uses data parallelism: each Mac runs the full MLX model on a deterministic shard of prompts or evaluation tasks, then writes shard-local artifacts that are merged into the same files the single-machine pipeline already produces.
2. V2 investigates true MLX distributed model execution over JACCL/Thunderbolt RDMA, using MLX distributed collectives only after the data-parallel path is stable and benchmarked.

For Qwen3-4B, data parallelism is the correct first speedup path because the model already fits on one Apple Silicon machine. JACCL/Thunderbolt RDMA should be treated as the V2 transport target, not the first production path.

## Current Pipeline

- Data generation currently selects paper-aligned prompts, formats them, calls `MLXTextGenerator.generate(...)`, and writes `train.parquet` plus `train.jsonl`.
- Evaluation currently loads LiveCodeBench problems, calls the same MLX generation adapter, runs correctness checks, and writes result JSON.
- The current generation defaults are paper-aligned for Qwen3-4B: bf16 model, temperature `1.6`, `top_p=0.8`, `top_k=20`, max tokens `65536`, and anti-loop detection enabled.
- The paper prompt-selection path deduplicates `microsoft/rStar-Coder/seed_sft` to about 10K unique prompts before generation.

## V1: Data-Parallel Distributed Pipeline

Implemented in `data_generation/generate.py` without changing the legacy
single-machine path. Distributed generation reuses the checkpoint system; each
worker writes shard-local checkpoint parts, and the merge command rebuilds final
`train.parquet` / `train.jsonl` from all shards.

CLI options:

- `--distributed-shard-index`
- `--distributed-num-shards`
- `--distributed-run-id`
- `--distributed-output-root`
- `--distributed-merge`
- `--checkpoint-every`
- `--resume`

Generation behavior:

- Apply sharding after paper prompt selection and deduplication.
- Use deterministic assignment: `global_prompt_index % distributed_num_shards == distributed_shard_index`.
- Preserve the global prompt order so merged output is stable and reproducible.
- Keep sampling settings identical across workers unless a seed is explicitly configured.
- If a seed is configured, use `seed + distributed_shard_index` to avoid identical RNG streams across workers.

Shard output layout:

```text
output/<run_id>/
  shards/
    shard-000/
      train.parquet
      train.jsonl
      metadata.json
      generation.log
    shard-001/
      train.parquet
      train.jsonl
      metadata.json
      generation.log
  merged/
    train.parquet
    train.jsonl
    metadata.json
```

Generated examples must include:

- `global_prompt_index`
- `distributed_shard_index`
- `distributed_num_shards`
- `distributed_run_id`
- `finish_reason`

Merge behavior:

- Verify all expected shard directories exist.
- Verify every shard used the same run id, model, dataset, prompt-selection config, generation config, and git commit if available.
- Verify there is no duplicate `global_prompt_index`.
- Verify merged coverage equals the union of shard coverage.
- Sort by `global_prompt_index`.
- Write merged `train.parquet`, `train.jsonl`, and `metadata.json`.

Evaluation behavior:

- Apply the same deterministic sharding after benchmark problem loading.
- Each worker evaluates only its assigned problems and repeats.
- Merge raw per-problem results first, then compute final metrics from the combined result set.
- Do not average shard-level pass rates.

## V1 Orchestration

Target cluster:

- 2-4 Apple Silicon Macs.
- Same git commit on every host.
- Same `uv.lock` and Python environment.
- Same model id or identical local model path.
- Same `data_generation/config.yaml`.
- Shared output directory over network storage, or explicit post-run `rsync` from each worker.

Example generation shard command:

```bash
PYTHONUNBUFFERED=1 uv run python data_generation/generate.py \
  --config data_generation/config.yaml \
  --model-name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --output-path ./output/distributed-qwen3-4b-ssd/shards/shard-000 \
  --distributed-run-id distributed-qwen3-4b-ssd \
  --distributed-output-root ./output/distributed-qwen3-4b-ssd \
  --distributed-num-shards 4 \
  --distributed-shard-index 0
```

Example merge command:

```bash
uv run python data_generation/generate.py \
  --distributed-merge \
  --distributed-run-id distributed-qwen3-4b-ssd \
  --distributed-output-root ./output/distributed-qwen3-4b-ssd \
  --distributed-num-shards 4
```

Add a small launcher script after the manual path works:

```bash
uv run python scripts/launch_distributed.py \
  --hosts hosts.txt \
  --run-id distributed-qwen3-4b-ssd \
  --num-shards 4 \
  --command generation
```

The launcher should only compose and run SSH commands. The generation/evaluation scripts remain the source of truth for sharding and merging.

## V2: MLX Distributed and RDMA Experiment

Use MLX distributed only after V1 is correct and benchmarked. Initial V2 JACCL
pipeline support is now present:

- `scripts/mlx_distributed_probe.py` initializes an MLX distributed group,
  validates `all_sum` / `all_gather`, and times a small all-reduce payload.
- The same probe can optionally load Qwen3 through
  `mlx_lm.utils.sharded_load`.
- `evaluation.MLXTextGenerator` has an explicit `use_mlx_distributed` opt-in
  that loads with a tensor-parallel group instead of the default local
  `mlx_lm.load` path.
- `data_generation/generate.py` exposes `--mlx-distributed` /
  `--mlx-distributed-backend jaccl`. Every rank participates in generation;
  rank 0 writes checkpoints, parquet, JSONL, and metadata.
- `evaluation/eval.py` exposes `--mlx_distributed` /
  `--mlx_distributed_backend jaccl`. Every rank participates in generation;
  rank 0 runs correctness tests and writes result JSON.

Transport validation:

- Confirm `mx.distributed.is_available()`.
- Generate a JACCL hostfile with `mlx.distributed_config --backend jaccl --over thunderbolt`.
- Ensure every host in the JACCL hostfile has the same checkout path and a working `.venv/bin/python`.
- Launch probes with explicit `mlx.launch --cwd <repo> --python <repo>/.venv/bin/python`.
- Run `mx.distributed.init(backend="jaccl")` across 2 Macs.
- Benchmark `all_gather`, `all_sum`, `send`, and `recv`.
- Record latency, bandwidth, CPU usage, wall time, and failure modes.
- Use JACCL only for V2 benchmarking.

Tensor/model parallel prototype:

- Run the JACCL probe with a small Qwen generation sample.
- Run `data_generation/generate.py --mlx-distributed --limit 1 --max-tokens 32`.
- Run `evaluation/eval.py --mlx_distributed --limit 1 --n_repeat 1 --max_tokens 8 --lcb_data_files test.jsonl --lcb_allow_any_date --lcb_preprocess_num_proc 1`.
- Compare against V1 data parallel throughput, not just single-request latency.

Acceptance gate:

- Keep V2 only if it enables a model that does not fit on one Mac, or if it beats V1 on sustained generated tokens/sec per watt.
- For Qwen3-4B SSD generation, V1 remains the production path unless V2 clearly beats it.

## Pipeline Coverage

Generation:

- Distributed workers generate disjoint prompt shards.
- Merge produces the canonical SSD training dataset.
- Anti-loop settings and finish reasons remain visible in shard and merged metadata.

Post-processing:

- Minimal syntactic filtering remains unchanged.
- Any future loop-output filtering must be configured explicitly and recorded in metadata.

Training handoff:

- Training consumes only merged artifacts.
- Shard-local outputs are retained for debugging but are not used directly by training.

Evaluation:

- Base-model and SimpleSD evaluation can use the same sharding mechanism.
- Merged evaluation results must match the single-machine schema.

## Test Plan

Unit tests:

- Sharding has no overlap.
- Sharding covers all selected prompts.
- Sharding is deterministic across runs.
- Sharding works when prompt count is not divisible by shard count.
- Merge rejects missing shards, duplicate global indexes, and mismatched metadata.

Generation smoke test:

- Run 2 shards with `--limit 8` and a small max-token value.
- Verify each shard writes 4 examples.
- Verify merged output has 8 examples sorted by `global_prompt_index`.
- Verify merged `train.parquet` and `train.jsonl` are compatible with current downstream code.

Evaluation smoke test:

- Run 2 shards with `--limit 4` and `--n_repeat 1`.
- Verify merged raw results contain all 4 problems.
- Verify final metrics match a single-process run on the same limit.

Cluster test:

- Run generation on 2 Macs over SSH/TCP first.
- Repeat with the RDMA/Thunderbolt transport target if available.
- Record prompt tokens/sec, generation tokens/sec, wall time, peak memory, loop-stop counts, and merge time per shard.

## Assumptions

- First implementation targets 2-4 Apple Silicon Macs.
- V1 data parallelism is the production path for Qwen3-4B.
- JACCL/Thunderbolt RDMA is the V2 transport target.
- Single-machine behavior must remain unchanged when distributed flags are omitted.
- Merged artifacts must stay compatible with the existing training and evaluation pipeline.
- Full paper-aligned generation uses the deduplicated prompt set, not the full 591K `seed_sft` rows.
