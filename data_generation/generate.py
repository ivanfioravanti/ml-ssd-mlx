#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset

from evaluation.mlx_generation import (
    MLXGenerationConfig,
    MLXTextGenerator,
    distributed_barrier,
    init_mlx_distributed_group,
)

# =============================================================================
# Data Loading
# =============================================================================

def load_hf_dataset(dataset_name: str, config_name: str, split: str = "train") -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face Hub."""
    print(f"Loading dataset '{dataset_name}' (config: {config_name}, split: {split})...")
    ds = load_dataset(dataset_name, config_name, split=split)
    examples = [dict(row) for row in ds]
    print(f"  Loaded {len(examples)} examples")
    return examples


# =============================================================================
# Helper Functions
# =============================================================================

def format_prompt(question, starter_code, problem_type, stdin_template, function_template):
    """Create prompt from templates."""
    if problem_type == 'function':
        return function_template.replace('{{ question }}', question or '').replace('{{ starter_code }}', starter_code or '')
    return stdin_template.replace('{{ question }}', question or '')


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def is_empty_or_stub_response(response: str, stub_max_chars: int = 80) -> bool:
    """Paper §3.1: drop empty responses and single-line stubs (no correctness check)."""
    text = str(response).strip()
    if not text:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0]) < stub_max_chars and "```" not in text:
        return True
    return False


def select_prompts(
    examples: List[Dict[str, Any]],
    prompt_selection: Dict[str, Any],
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """
    Reduce seed_sft rows to one prompt per problem for SSD data synthesis.

    Paper §3.1 uses ~10K deduplicated competitive programming prompts (N=1 sample each).
    On public seed_sft, filtering is_passed then deduplicating by question_id yields ~10,168
    unique problems — the closest reproducible match to the paper count.
    """
    mode = prompt_selection.get("mode", "paper")
    if mode == "full":
        if limit > 0:
            examples = examples[:limit]
            print(f"  Full dataset mode: limited to {len(examples)} examples")
        return examples

    dedupe_key = prompt_selection.get("deduplicate_by", "question_id")
    filter_is_passed = prompt_selection.get("filter_is_passed", True)
    max_prompts = int(prompt_selection.get("max_prompts", 0) or 0)
    prefer_passed_row = prompt_selection.get("prefer_passed_row", True)

    print("  Prompt selection: paper (~10K unique problems)")
    print(f"    filter_is_passed={filter_is_passed}, deduplicate_by={dedupe_key}")

    if filter_is_passed:
        before = len(examples)
        examples = [ex for ex in examples if ex.get("is_passed")]
        print(f"    is_passed filter: {before} -> {len(examples)} rows")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    for ex in examples:
        key = ex.get(dedupe_key)
        if key is None or str(key).strip() == "":
            skipped += 1
            continue
        grouped.setdefault(str(key), []).append(ex)

    if skipped:
        print(f"    skipped {skipped} rows without {dedupe_key}")

    selected: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        if prefer_passed_row:
            passed_rows = [row for row in rows if row.get("is_passed")]
            selected.append(passed_rows[0] if passed_rows else rows[0])
        else:
            selected.append(rows[0])

    print(f"    deduplicated: {len(selected)} unique prompts")

    if max_prompts > 0 and len(selected) > max_prompts:
        selected = selected[:max_prompts]
        print(f"    capped to max_prompts={max_prompts}")

    if limit > 0:
        selected = selected[:limit]
        print(f"    CLI --limit: using first {len(selected)} prompts")

    return selected


def checkpoint_dir_for(output_dir: str) -> str:
    return os.path.join(output_dir, "checkpoints")


def shard_dir_name(shard_index: int) -> str:
    return f"shard-{shard_index:03d}"


def checkpoint_part_path(checkpoint_dir: str, start_index: int) -> str:
    return os.path.join(checkpoint_dir, f"part-{start_index:06d}.parquet")


def checkpoint_metadata_path(parquet_path: str) -> str:
    base, _ = os.path.splitext(parquet_path)
    return f"{base}.meta.json"


def list_checkpoint_parts(checkpoint_dir: str) -> List[str]:
    if not os.path.isdir(checkpoint_dir):
        return []
    return sorted(
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("part-") and name.endswith(".parquet")
    )


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def write_parquet_atomic(path: str, records: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    table = pa.Table.from_pylist(records)
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, path)


def write_checkpoint_part(
    checkpoint_dir: str,
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    if not records:
        raise ValueError("Cannot write an empty checkpoint")

    os.makedirs(checkpoint_dir, exist_ok=True)
    indexes = [int(record["global_prompt_index"]) for record in records]
    start_index = min(indexes)
    end_index = max(indexes)
    path = checkpoint_part_path(checkpoint_dir, start_index)
    if os.path.exists(path):
        raise FileExistsError(f"Checkpoint already exists: {path}")

    write_parquet_atomic(path, records)
    write_json_atomic(
        checkpoint_metadata_path(path),
        {
            **metadata,
            "checkpoint_start_index": start_index,
            "checkpoint_end_index": end_index,
            "num_records": len(records),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return path


def load_checkpoint_records(checkpoint_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: Set[int] = set()

    for path in list_checkpoint_parts(checkpoint_dir):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            if "global_prompt_index" not in row:
                raise ValueError(f"Checkpoint row missing global_prompt_index: {path}")
            index = int(row["global_prompt_index"])
            if index in seen:
                raise ValueError(f"Duplicate global_prompt_index={index} in checkpoints")
            seen.add(index)
            records.append(row)

    return sorted(records, key=lambda row: int(row["global_prompt_index"]))


def validate_checkpoint_metadata(
    checkpoint_dir: str,
    expected_metadata: Dict[str, Any],
) -> None:
    for path in list_checkpoint_parts(checkpoint_dir):
        metadata_path = checkpoint_metadata_path(path)
        if not os.path.exists(metadata_path):
            raise ValueError(f"Checkpoint metadata missing: {metadata_path}")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Checkpoint metadata mismatch for {path}: "
                    f"{key}={metadata.get(key)!r}, expected {expected_value!r}"
                )


def completed_checkpoint_indexes(checkpoint_dir: str) -> Set[int]:
    return {
        int(record["global_prompt_index"])
        for record in load_checkpoint_records(checkpoint_dir)
    }


def validate_checkpoint_coverage(
    records: List[Dict[str, Any]],
    expected_indexes: Set[int],
) -> None:
    actual_indexes = {int(record["global_prompt_index"]) for record in records}
    missing = sorted(expected_indexes - actual_indexes)
    extra = sorted(actual_indexes - expected_indexes)
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Missing {len(missing)} checkpointed examples: {preview}")
    if extra:
        preview = ", ".join(str(index) for index in extra[:10])
        raise ValueError(f"Found {len(extra)} unexpected checkpointed examples: {preview}")


def load_distributed_shard_records(
    run_root: str,
    num_shards: int,
    expected_metadata: Dict[str, Any],
    run_id: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for shard_index in range(num_shards):
        shard_dir = os.path.join(run_root, "shards", shard_dir_name(shard_index))
        shard_checkpoint_dir = checkpoint_dir_for(shard_dir)
        shard_expected_metadata = {
            **expected_metadata,
            "distributed_num_shards": num_shards,
            "distributed_shard_index": shard_index,
            "distributed_run_id": run_id,
        }

        if list_checkpoint_parts(shard_checkpoint_dir):
            validate_checkpoint_metadata(shard_checkpoint_dir, shard_expected_metadata)
            shard_records = load_checkpoint_records(shard_checkpoint_dir)
        else:
            parquet_path = os.path.join(shard_dir, "train.parquet")
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(
                    f"Missing shard output: {shard_checkpoint_dir} or {parquet_path}"
                )
            shard_records = pq.read_table(parquet_path).to_pylist()

        for record in shard_records:
            global_index = int(record["global_prompt_index"])
            if global_index % num_shards != shard_index:
                raise ValueError(
                    f"Shard {shard_index} contains global_prompt_index={global_index}"
                )
            records.append(record)

    return sorted(records, key=lambda row: int(row["global_prompt_index"]))


def write_training_jsonl(
    examples: List[Dict[str, Any]],
    jsonl_path: str,
    *,
    filter_percent: float,
    minimal_syntactic_filter: bool,
    stub_max_chars: int,
) -> Tuple[int, int, int]:
    valid_records = [
        ex for ex in examples
        if ex.get('output') and str(ex['output']).strip()
    ]

    min_length = 0
    if filter_percent > 0 and valid_records:
        lengths = sorted(len(str(ex['output']).strip()) for ex in valid_records)
        cutoff_idx = min(int(len(lengths) * filter_percent / 100.0), len(lengths) - 1)
        min_length = lengths[cutoff_idx]
        print(f"  Filter: dropping bottom {filter_percent}% shortest (min_length={min_length})")

    kept = 0
    filtered = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            prompt = ex.get('prompt', '')
            response = ex.get('output', '')

            if not prompt or not str(prompt).strip():
                filtered += 1
                continue
            if not response or not str(response).strip():
                filtered += 1
                continue

            prompt = str(prompt).strip()
            response = str(response).strip()

            if minimal_syntactic_filter and is_empty_or_stub_response(response, stub_max_chars):
                filtered += 1
                continue

            if len(response) < min_length:
                filtered += 1
                continue

            entry = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

    return kept, filtered, min_length


# =============================================================================
# Main Pipeline
# =============================================================================

def load_templates(template_dir: str) -> Tuple[str, str]:
    """Load prompt templates."""
    stdin_path = os.path.join(template_dir, "self_distillation_prompt_stdin.j2")
    function_path = os.path.join(template_dir, "self_distillation_prompt_function.j2")

    for path in [stdin_path, function_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found: {path}")

    with open(stdin_path) as f:
        stdin_template = f.read()
    with open(function_path) as f:
        function_template = f.read()

    return stdin_template, function_template


def generate(config: Dict[str, Any], template_dir: str, limit: int = 0):
    """Run the data generation pipeline."""

    # Extract config
    model_name = config['model']['name']
    trust_remote_code = config['model'].get('trust_remote_code', True)

    dataset_name = config['dataset']['name']
    dataset_config = config['dataset']['config']
    dataset_split = config['dataset'].get('split', 'train')

    output_dir = config['output']['path']

    temperature = config['generation'].get('temperature', 1.6)
    top_k = config['generation'].get('top_k', 20)
    top_p = config['generation'].get('top_p', 0.8)
    min_p = config['generation'].get('min_p', 0.0)
    repetition_penalty = config['generation'].get('repetition_penalty', 1.0)
    repetition_context_size = config['generation'].get('repetition_context_size', 20)
    max_tokens = config['generation'].get('max_tokens', 65536)
    stop = config['generation'].get('stop', ["<|im_end|>", "<|endoftext|>"])
    loop_detection = config['generation'].get('loop_detection', {})
    loop_ngram_min = loop_detection.get('ngram_min', 0)
    loop_ngram_max = loop_detection.get('ngram_max', 0)
    loop_repetitions = loop_detection.get('repetitions', 0)
    loop_min_tokens = loop_detection.get('min_tokens', 0)
    loop_check_interval = loop_detection.get('check_interval', 128)
    loop_text_window_tokens = loop_detection.get('text_window_tokens', 2048)
    loop_text_ngram_min = loop_detection.get('text_ngram_min', 5)
    loop_text_ngram_max = loop_detection.get('text_ngram_max', 12)
    loop_text_repetitions = loop_detection.get('text_repetitions', 20)
    loop_max_code_fences = loop_detection.get('max_code_fences', 0)

    batch_config = config.get('batch', {})
    completion_batch_size = batch_config.get('completion_batch_size', 32)
    prefill_batch_size = batch_config.get('prefill_batch_size', 8)
    prefill_step_size = batch_config.get('prefill_step_size', 2048)
    max_kv_size = batch_config.get('max_kv_size')

    post_process = config.get('post_process', {})
    prompt_selection = config.get('prompt_selection', {})
    paper_mode = prompt_selection.get('mode', 'paper') == 'paper'
    filter_percent = post_process.get('filter_shortest_percent', 0.0 if paper_mode else 10.0)
    minimal_syntactic_filter = post_process.get(
        'minimal_syntactic_filter',
        paper_mode,
    )
    stub_max_chars = int(post_process.get('stub_max_chars', 80))
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_every = int(checkpoint_config.get('every', 0) or 0)
    resume_checkpoints = bool(checkpoint_config.get('resume', False))
    merge_checkpoints_only = bool(checkpoint_config.get('merge_only', False))
    checkpoint_dir = checkpoint_dir_for(output_dir)
    distributed_config = config.get('distributed', {})
    distributed_num_shards = int(distributed_config.get('num_shards', 1) or 1)
    distributed_shard_index = int(distributed_config.get('shard_index', 0) or 0)
    distributed_run_id = str(distributed_config.get('run_id') or "")
    distributed_run_root = str(distributed_config.get('run_root') or "")
    distributed_merge = bool(distributed_config.get('merge', False))
    distributed_active = (
        distributed_merge
        or distributed_num_shards > 1
        or bool(distributed_run_id)
        or bool(distributed_run_root)
    )
    mlx_distributed_config = config.get('mlx_distributed', {})
    mlx_distributed_enabled = bool(mlx_distributed_config.get('enabled', False))
    mlx_distributed_backend = str(mlx_distributed_config.get('backend', 'jaccl'))

    if mlx_distributed_enabled and mlx_distributed_backend != 'jaccl':
        raise ValueError("mlx_distributed.backend must be 'jaccl'")

    mlx_distributed_group = (
        init_mlx_distributed_group(mlx_distributed_backend)
        if mlx_distributed_enabled
        else None
    )
    mlx_distributed_rank = (
        mlx_distributed_group.rank() if mlx_distributed_group is not None else 0
    )
    mlx_distributed_world_size = (
        mlx_distributed_group.size() if mlx_distributed_group is not None else 1
    )
    mlx_distributed_is_rank0 = mlx_distributed_rank == 0

    if distributed_num_shards < 1:
        raise ValueError("distributed.num_shards must be >= 1")
    if distributed_shard_index < 0 or distributed_shard_index >= distributed_num_shards:
        raise ValueError(
            "distributed.shard_index must be in "
            f"[0, {distributed_num_shards - 1}]"
        )
    if distributed_merge and not distributed_run_root:
        raise ValueError("--distributed-merge requires --distributed-output-root or --distributed-run-id")

    def create_generator() -> MLXTextGenerator:
        return MLXTextGenerator(
            model_name,
            trust_remote_code=trust_remote_code,
            use_mlx_distributed=mlx_distributed_enabled,
            distributed_backend=mlx_distributed_backend,
            distributed_group=mlx_distributed_group,
        )

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{dataset_config} (split: {dataset_split})")
    print(f"Output: {output_dir}")
    print(f"Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, "
          f"min_p={min_p}, rep_penalty={repetition_penalty}, max_tokens={max_tokens}")
    if loop_ngram_min > 0 and loop_ngram_max >= loop_ngram_min and loop_repetitions > 1:
        print(f"Loop detection: ngram={loop_ngram_min}-{loop_ngram_max}, "
              f"repetitions={loop_repetitions}, min_tokens={loop_min_tokens}, "
              f"text_ngram={loop_text_ngram_min}-{loop_text_ngram_max}, "
              f"text_repetitions={loop_text_repetitions}, "
              f"max_code_fences={loop_max_code_fences}")
    print(f"MLX batch: completion={completion_batch_size}, prefill={prefill_batch_size}, "
          f"prefill_step={prefill_step_size}, max_kv_size={max_kv_size}")
    print(f"Prompt selection: {prompt_selection.get('mode', 'paper')}")
    print(f"Post-process: shortest_filter={filter_percent}%, "
          f"minimal_syntactic_filter={minimal_syntactic_filter}")
    if checkpoint_every > 0 or merge_checkpoints_only:
        mode = "merge-only" if merge_checkpoints_only else f"every={checkpoint_every}"
        print(f"Checkpointing: {mode}, resume={resume_checkpoints}, dir={checkpoint_dir}")
    if distributed_active:
        if distributed_merge:
            print(
                f"Distributed: merge {distributed_num_shards} shards from {distributed_run_root}"
            )
        else:
            print(
                f"Distributed: shard {distributed_shard_index}/{distributed_num_shards}, "
                f"run_id={distributed_run_id or '<none>'}"
            )
    if mlx_distributed_enabled:
        print(
            "MLX distributed: "
            f"backend={mlx_distributed_backend}, "
            f"rank={mlx_distributed_rank}/{mlx_distributed_world_size}"
        )
    print("-" * 60)

    pipeline_start = time.time()

    # Load templates
    stdin_template, function_template = load_templates(template_dir)
    print("Templates loaded.")

    # ===== STAGE 0: Load and Format Data =====
    print("\n" + "=" * 60)
    print("STAGE 0: Load data + format prompts")
    stage0_start = time.time()

    examples = load_hf_dataset(dataset_name, dataset_config, split=dataset_split)
    examples = select_prompts(examples, prompt_selection, limit=limit)
    print(f"Total selected examples: {len(examples)}")

    for idx, ex in enumerate(examples):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Formatting: {idx}/{len(examples)} ({100*idx/len(examples):.1f}%)")

        # Infer problem_type from starter_code
        starter_code = ex.get('starter_code')
        ex['global_prompt_index'] = idx
        ex['problem_type'] = 'function' if starter_code and starter_code.strip() else 'stdin'

        # Generate prompt
        ex['prompt'] = format_prompt(
            ex.get('question', ''), ex.get('starter_code'), ex['problem_type'],
            stdin_template, function_template,
        )

    selected_indexes = {int(ex["global_prompt_index"]) for ex in examples}
    mlx_distributed_workers_released = False

    if distributed_merge:
        expected_indexes = selected_indexes
    else:
        if distributed_active:
            before_shard = len(examples)
            examples = [
                ex for ex in examples
                if int(ex["global_prompt_index"]) % distributed_num_shards == distributed_shard_index
            ]
            print(
                f"Distributed shard selection: {before_shard} -> {len(examples)} examples "
                f"(shard {distributed_shard_index}/{distributed_num_shards})"
            )
            for ex in examples:
                ex["distributed_num_shards"] = distributed_num_shards
                ex["distributed_shard_index"] = distributed_shard_index
                ex["distributed_run_id"] = distributed_run_id
        expected_indexes = {int(ex["global_prompt_index"]) for ex in examples}

    print(f"Total examples for generation: {len(examples)}")

    stage0_elapsed = time.time() - stage0_start
    print(f"STAGE 0 complete in {format_duration(stage0_elapsed)}", flush=True)

    # ===== STAGE 1: Generate Solutions =====
    print("\n" + "=" * 60)
    print("STAGE 1: Generate solutions")
    stage1_start = time.time()

    generation_config = MLXGenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        max_tokens=max_tokens,
        stop=stop,
        completion_batch_size=completion_batch_size,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
        max_kv_size=max_kv_size,
        loop_ngram_min=loop_ngram_min,
        loop_ngram_max=loop_ngram_max,
        loop_repetitions=loop_repetitions,
        loop_min_tokens=loop_min_tokens,
        loop_check_interval=loop_check_interval,
        loop_text_window_tokens=loop_text_window_tokens,
        loop_text_ngram_min=loop_text_ngram_min,
        loop_text_ngram_max=loop_text_ngram_max,
        loop_text_repetitions=loop_text_repetitions,
        loop_max_code_fences=loop_max_code_fences,
    )

    checkpoint_metadata = {
        "model": model_name,
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": dataset_split,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "completion_batch_size": completion_batch_size,
        "prefill_batch_size": prefill_batch_size,
        "prefill_step_size": prefill_step_size,
        "loop_detection": loop_detection,
    }
    if mlx_distributed_enabled:
        checkpoint_metadata["mlx_distributed"] = {
            "enabled": True,
            "backend": mlx_distributed_backend,
            "world_size": mlx_distributed_world_size,
        }
    if distributed_active and not distributed_merge:
        checkpoint_metadata.update(
            {
                "distributed_num_shards": distributed_num_shards,
                "distributed_shard_index": distributed_shard_index,
                "distributed_run_id": distributed_run_id,
            }
        )

    if distributed_merge:
        print("Distributed merge mode; skipping generation.")
        examples = load_distributed_shard_records(
            distributed_run_root,
            distributed_num_shards,
            checkpoint_metadata,
            distributed_run_id,
        )
        validate_checkpoint_coverage(examples, expected_indexes)
        print(
            f"Loaded {len(examples)} examples from "
            f"{distributed_num_shards} distributed shards"
        )
    elif merge_checkpoints_only:
        print("Merge-only checkpoint mode; skipping generation.")
        validate_checkpoint_metadata(checkpoint_dir, checkpoint_metadata)
        examples = load_checkpoint_records(checkpoint_dir)
        validate_checkpoint_coverage(examples, expected_indexes)
        print(f"Loaded {len(examples)} checkpointed examples from {checkpoint_dir}")
    elif checkpoint_every > 0:
        existing_parts = list_checkpoint_parts(checkpoint_dir)
        if existing_parts and not resume_checkpoints:
            raise RuntimeError(
                f"Found {len(existing_parts)} existing checkpoint parts in {checkpoint_dir}. "
                "Use --resume or choose a new --output-path."
            )
        if existing_parts:
            validate_checkpoint_metadata(checkpoint_dir, checkpoint_metadata)

        completed_indexes = completed_checkpoint_indexes(checkpoint_dir) if resume_checkpoints else set()
        unexpected_completed = completed_indexes - expected_indexes
        if unexpected_completed:
            preview = ", ".join(str(index) for index in sorted(unexpected_completed)[:10])
            raise RuntimeError(
                f"Checkpoint directory contains indexes outside this run: {preview}"
            )

        remaining_examples = [
            ex for ex in examples
            if int(ex["global_prompt_index"]) not in completed_indexes
        ]
        print(
            f"Checkpoint resume: {len(completed_indexes)} already complete, "
            f"{len(remaining_examples)} remaining."
        )

        if remaining_examples:
            print(f"Initializing MLX-LM (model: {model_name})...")
            generator = create_generator()

            print(
                f"Generating {len(remaining_examples)} examples "
                f"in chunks of {checkpoint_every}..."
            )
            generated_so_far = len(completed_indexes)
            for chunk_start in range(0, len(remaining_examples), checkpoint_every):
                chunk = remaining_examples[chunk_start:chunk_start + checkpoint_every]
                first_index = int(chunk[0]["global_prompt_index"])
                last_index = int(chunk[-1]["global_prompt_index"])
                print(
                    f"\nCheckpoint chunk {first_index}-{last_index} "
                    f"({len(chunk)} examples)",
                    flush=True,
                )

                prompts = [ex['prompt'] for ex in chunk]
                distributed_barrier(mlx_distributed_group)
                outputs = generator.generate(
                    prompts,
                    generation_config,
                    verbose=mlx_distributed_is_rank0,
                )

                if mlx_distributed_is_rank0:
                    for ex, output in zip(chunk, outputs):
                        ex['output'] = output
                    if getattr(generator, "last_finish_reasons", None):
                        for ex, finish_reason in zip(chunk, generator.last_finish_reasons):
                            ex['finish_reason'] = finish_reason

                    checkpoint_path = write_checkpoint_part(
                        checkpoint_dir,
                        chunk,
                        checkpoint_metadata,
                    )
                    generated_so_far += len(chunk)
                    print(
                        f"Saved checkpoint {checkpoint_path} "
                        f"({generated_so_far}/{len(examples)} complete)",
                        flush=True,
                    )
                distributed_barrier(mlx_distributed_group)
        else:
            print("All selected examples are already checkpointed; no generation needed.")

        if mlx_distributed_enabled:
            distributed_barrier(mlx_distributed_group)
            if not mlx_distributed_is_rank0:
                print(
                    f"MLX distributed rank {mlx_distributed_rank} completed generation; "
                    "rank 0 will write artifacts.",
                    flush=True,
                )
                return
            mlx_distributed_workers_released = True

        examples = load_checkpoint_records(checkpoint_dir)
        validate_checkpoint_coverage(examples, expected_indexes)
    else:
        print(f"Initializing MLX-LM (model: {model_name})...")
        generator = create_generator()

        print(f"Generating solutions for {len(examples)} examples...")
        prompts = [ex['prompt'] for ex in examples]
        distributed_barrier(mlx_distributed_group)
        outputs = generator.generate(
            prompts,
            generation_config,
            verbose=mlx_distributed_is_rank0,
        )

        if mlx_distributed_is_rank0:
            for ex, output in zip(examples, outputs):
                ex['output'] = output
            if getattr(generator, "last_finish_reasons", None):
                for ex, finish_reason in zip(examples, generator.last_finish_reasons):
                    ex['finish_reason'] = finish_reason

    stage1_elapsed = time.time() - stage1_start
    print(
        f"STAGE 1 complete: {len(examples)} solutions available in {format_duration(stage1_elapsed)}",
        flush=True,
    )

    if mlx_distributed_enabled:
        if not mlx_distributed_workers_released:
            distributed_barrier(mlx_distributed_group)
        if not mlx_distributed_is_rank0:
            print(
                f"MLX distributed rank {mlx_distributed_rank} completed generation; "
                "rank 0 will write artifacts.",
                flush=True,
            )
            return

    # ===== STAGE 2: Save Results =====
    print("\n" + "=" * 60)
    print("STAGE 2: Save results")
    stage2_start = time.time()

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "train.parquet")

    write_parquet_atomic(parquet_path, examples)
    print(f"Saved {len(examples)} examples to {parquet_path}")

    stage2_elapsed = time.time() - stage2_start
    print(f"STAGE 2 complete in {format_duration(stage2_elapsed)}", flush=True)

    # ===== STAGE 3: Post-process to JSONL =====
    print("\n" + "=" * 60)
    print("STAGE 3: Post-process to training JSONL")
    stage3_start = time.time()

    jsonl_path = os.path.join(output_dir, "train.jsonl")

    kept, filtered, _ = write_training_jsonl(
        examples,
        jsonl_path,
        filter_percent=filter_percent,
        minimal_syntactic_filter=minimal_syntactic_filter,
        stub_max_chars=stub_max_chars,
    )

    stage3_elapsed = time.time() - stage3_start
    total_elapsed = time.time() - pipeline_start
    print(f"STAGE 3 complete in {format_duration(stage3_elapsed)}", flush=True)
    print(f"  Total records:  {len(examples)}", flush=True)
    print(f"  Kept:           {kept} ({100*kept/len(examples):.1f}%)", flush=True)
    print(f"  Filtered:       {filtered}", flush=True)
    print(f"  Output:         {jsonl_path}", flush=True)

    finish_reasons = [ex.get("finish_reason") for ex in examples if ex.get("finish_reason")]
    loop_stopped = sum(1 for reason in finish_reasons if reason == "loop")
    checkpoint_count = len(list_checkpoint_parts(checkpoint_dir))
    metadata_path = os.path.join(output_dir, "metadata.json")
    write_json_atomic(
        metadata_path,
        {
            "model": model_name,
            "dataset": f"{dataset_name}/{dataset_config}",
            "split": dataset_split,
            "total_examples": len(examples),
            "kept_jsonl": kept,
            "filtered_jsonl": filtered,
            "loop_stopped": loop_stopped,
            "checkpoint_count": checkpoint_count,
            "checkpoint_every": checkpoint_every,
            "prompt_selection": prompt_selection,
            "generation": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
            },
            "batch": {
                "completion_batch_size": completion_batch_size,
                "prefill_batch_size": prefill_batch_size,
                "prefill_step_size": prefill_step_size,
                "max_kv_size": max_kv_size,
            },
            "distributed": {
                "active": distributed_active,
                "merge": distributed_merge,
                "run_id": distributed_run_id,
                "run_root": distributed_run_root,
                "num_shards": distributed_num_shards,
                "shard_index": None if distributed_merge else distributed_shard_index,
            },
            "mlx_distributed": {
                "enabled": mlx_distributed_enabled,
                "backend": mlx_distributed_backend if mlx_distributed_enabled else None,
                "rank": mlx_distributed_rank,
                "world_size": mlx_distributed_world_size,
            },
            "timing_seconds": {
                "stage0": stage0_elapsed,
                "stage1": stage1_elapsed,
                "stage2": stage2_elapsed,
                "stage3": stage3_elapsed,
                "total": total_elapsed,
            },
        },
    )

    print("\n" + "=" * 60, flush=True)
    print("Statistics:", flush=True)
    print(f"  Total examples:    {len(examples)}", flush=True)
    print(f"  Generated:         {len(examples)}", flush=True)
    print(f"  Kept (JSONL):      {kept}", flush=True)
    if finish_reasons:
        print(f"  Loop-stopped:      {loop_stopped}/{len(examples)}", flush=True)
    if checkpoint_every > 0 or merge_checkpoints_only:
        print(f"  Checkpoints:       {checkpoint_count}", flush=True)
    if distributed_active:
        if distributed_merge:
            print(f"  Distributed merge: {distributed_num_shards} shards", flush=True)
        else:
            print(
                f"  Distributed shard: {distributed_shard_index}/{distributed_num_shards}",
                flush=True,
            )
    if mlx_distributed_enabled:
        print(
            f"  MLX distributed:   {mlx_distributed_backend} "
            f"({mlx_distributed_world_size} ranks)",
            flush=True,
        )
    print(f"  Metadata:          {metadata_path}", flush=True)
    print("-" * 60, flush=True)
    print("Pipeline timing:", flush=True)
    print(f"  STAGE 0 (load + format):  {format_duration(stage0_elapsed):>12}", flush=True)
    print(f"  STAGE 1 (generate):       {format_duration(stage1_elapsed):>12}", flush=True)
    print(f"  STAGE 2 (save parquet):   {format_duration(stage2_elapsed):>12}", flush=True)
    print(f"  STAGE 3 (post-process):   {format_duration(stage3_elapsed):>12}", flush=True)
    print(f"  Total:                    {format_duration(total_elapsed):>12}", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for coding problems using MLX-LM")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--temperature", type=float, help="Override generation temperature")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--dataset-name", type=str, help="Override HuggingFace dataset name")
    parser.add_argument("--output-path", type=str, help="Override output path")
    parser.add_argument("--max-tokens", type=int, help="Override maximum generated tokens")
    parser.add_argument("--completion-batch-size", type=int, help="Override MLX-LM completion batch size")
    parser.add_argument("--prefill-batch-size", type=int, help="Override MLX-LM prefill batch size")
    parser.add_argument("--prefill-step-size", type=int, help="Override MLX-LM prefill step size")
    parser.add_argument("--loop-ngram-min", type=int, help="Minimum repeated token span for loop detection")
    parser.add_argument("--loop-ngram-max", type=int, help="Maximum repeated token span for loop detection")
    parser.add_argument("--loop-repetitions", type=int, help="Repeated spans required before loop stopping")
    parser.add_argument("--loop-min-tokens", type=int, help="Minimum generated tokens before loop detection")
    parser.add_argument("--loop-check-interval", type=int, help="Generated-token interval for decoded text loop checks")
    parser.add_argument("--loop-text-window-tokens", type=int, help="Recent token window decoded for text loop checks")
    parser.add_argument("--loop-text-ngram-min", type=int, help="Minimum repeated word span for text loop detection")
    parser.add_argument("--loop-text-ngram-max", type=int, help="Maximum repeated word span for text loop detection")
    parser.add_argument("--loop-text-repetitions", type=int, help="Repeated word spans required before text loop stopping")
    parser.add_argument("--loop-max-code-fences", type=int, help="Stop after this many fenced code delimiters are generated")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples after prompt selection (0 = all)")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Write a parquet checkpoint after this many generated examples (0 = disabled)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from existing checkpoint parts in the output directory",
    )
    parser.add_argument(
        "--merge-checkpoints",
        action="store_true",
        help="Skip generation and merge existing checkpoint parts into final parquet/jsonl",
    )
    parser.add_argument("--distributed-num-shards", type=int, help="Total number of distributed data shards")
    parser.add_argument("--distributed-shard-index", type=int, help="This worker's shard index")
    parser.add_argument("--distributed-run-id", type=str, help="Distributed run id used in metadata and output paths")
    parser.add_argument(
        "--distributed-output-root",
        type=str,
        help="Parent output directory for distributed run artifacts",
    )
    parser.add_argument(
        "--distributed-merge",
        action="store_true",
        help="Merge distributed shard outputs into final parquet/jsonl without generation",
    )
    parser.add_argument(
        "--mlx-distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use MLX JACCL tensor-parallel model loading under mlx.launch",
    )
    parser.add_argument(
        "--mlx-distributed-backend",
        choices=("jaccl",),
        help="MLX distributed backend for tensor-parallel inference",
    )
    parser.add_argument(
        "--paper-prompts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use paper-style ~10K deduplicated prompts (default: from config prompt_selection.mode)",
    )
    parser.add_argument(
        "--filter-is-passed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When using paper prompts, keep only rows with is_passed=True (default: true)",
    )
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    explicit_output_path = args.output_path is not None

    # Apply CLI overrides
    if args.temperature is not None:
        config['generation']['temperature'] = args.temperature
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.dataset_name:
        config['dataset']['name'] = args.dataset_name
    if args.output_path:
        config['output']['path'] = args.output_path
    if args.max_tokens is not None:
        config['generation']['max_tokens'] = args.max_tokens
    if args.completion_batch_size is not None:
        config.setdefault('batch', {})['completion_batch_size'] = args.completion_batch_size
    if args.prefill_batch_size is not None:
        config.setdefault('batch', {})['prefill_batch_size'] = args.prefill_batch_size
    if args.prefill_step_size is not None:
        config.setdefault('batch', {})['prefill_step_size'] = args.prefill_step_size
    if args.checkpoint_every is not None:
        config.setdefault('checkpoint', {})['every'] = args.checkpoint_every
    if args.resume is not None:
        config.setdefault('checkpoint', {})['resume'] = args.resume
    if args.merge_checkpoints:
        config.setdefault('checkpoint', {})['merge_only'] = True
    if args.mlx_distributed is not None:
        config.setdefault('mlx_distributed', {})['enabled'] = args.mlx_distributed
    if args.mlx_distributed_backend is not None:
        config.setdefault('mlx_distributed', {})['backend'] = args.mlx_distributed_backend
    distributed_args_present = any(
        value is not None
        for value in [
            args.distributed_num_shards,
            args.distributed_shard_index,
            args.distributed_run_id,
            args.distributed_output_root,
        ]
    ) or args.distributed_merge
    if distributed_args_present:
        distributed = config.setdefault('distributed', {})
        if args.distributed_num_shards is not None:
            distributed['num_shards'] = args.distributed_num_shards
        if args.distributed_shard_index is not None:
            distributed['shard_index'] = args.distributed_shard_index
        if args.distributed_run_id is not None:
            distributed['run_id'] = args.distributed_run_id
        if args.distributed_output_root is not None:
            distributed['output_root'] = args.distributed_output_root
        if args.distributed_merge:
            distributed['merge'] = True

        num_shards = int(distributed.get('num_shards', 1) or 1)
        shard_index = int(distributed.get('shard_index', 0) or 0)
        run_id = str(distributed.get('run_id') or "")
        output_root = distributed.get('output_root')
        merge = bool(distributed.get('merge', False))

        if num_shards < 1:
            print("Error: --distributed-num-shards must be >= 1")
            sys.exit(1)
        if shard_index < 0 or shard_index >= num_shards:
            print("Error: --distributed-shard-index must be in range")
            sys.exit(1)

        run_root = ""
        if output_root:
            run_root = os.path.join(output_root, run_id) if run_id else output_root
        elif run_id and not explicit_output_path:
            run_root = os.path.join(config['output']['path'], run_id)
        elif merge:
            print("Error: --distributed-merge requires --distributed-output-root or --distributed-run-id")
            sys.exit(1)

        if run_root:
            distributed['run_root'] = run_root
            if not explicit_output_path:
                if merge:
                    config['output']['path'] = os.path.join(run_root, "merged")
                else:
                    config['output']['path'] = os.path.join(
                        run_root,
                        "shards",
                        shard_dir_name(shard_index),
                    )
    loop_args = {
        'ngram_min': args.loop_ngram_min,
        'ngram_max': args.loop_ngram_max,
        'repetitions': args.loop_repetitions,
        'min_tokens': args.loop_min_tokens,
        'check_interval': args.loop_check_interval,
        'text_window_tokens': args.loop_text_window_tokens,
        'text_ngram_min': args.loop_text_ngram_min,
        'text_ngram_max': args.loop_text_ngram_max,
        'text_repetitions': args.loop_text_repetitions,
        'max_code_fences': args.loop_max_code_fences,
    }
    if any(value is not None for value in loop_args.values()):
        loop_config = config.setdefault('generation', {}).setdefault('loop_detection', {})
        for key, value in loop_args.items():
            if value is not None:
                loop_config[key] = value

    prompt_selection = config.setdefault('prompt_selection', {})
    if args.paper_prompts is not None:
        prompt_selection['mode'] = 'paper' if args.paper_prompts else 'full'
    if args.filter_is_passed is not None:
        prompt_selection['filter_is_passed'] = args.filter_is_passed

    # Resolve template directory (templates/ next to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "templates")

    generate(config, template_dir, limit=args.limit)


if __name__ == "__main__":
    main()
