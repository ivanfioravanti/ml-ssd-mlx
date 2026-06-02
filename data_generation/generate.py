#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset

from evaluation.mlx_generation import MLXGenerationConfig, MLXTextGenerator

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
    print("-" * 60)

    # Load templates
    stdin_template, function_template = load_templates(template_dir)
    print("Templates loaded.")

    # ===== STAGE 0: Load and Format Data =====
    print("\n" + "=" * 60)
    print("STAGE 0: Load data + format prompts")
    stage0_start = time.time()

    examples = load_hf_dataset(dataset_name, dataset_config, split=dataset_split)
    examples = select_prompts(examples, prompt_selection, limit=limit)
    print(f"Total examples for generation: {len(examples)}")

    for idx, ex in enumerate(examples):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Formatting: {idx}/{len(examples)} ({100*idx/len(examples):.1f}%)")

        # Infer problem_type from starter_code
        starter_code = ex.get('starter_code')
        ex['problem_type'] = 'function' if starter_code and starter_code.strip() else 'stdin'

        # Generate prompt
        ex['prompt'] = format_prompt(
            ex.get('question', ''), ex.get('starter_code'), ex['problem_type'],
            stdin_template, function_template,
        )

    print(f"STAGE 0 complete in {time.time() - stage0_start:.1f}s")

    # ===== STAGE 1: Generate Solutions =====
    print("\n" + "=" * 60)
    print("STAGE 1: Generate solutions")
    stage1_start = time.time()

    print(f"Initializing MLX-LM (model: {model_name})...")
    generator = MLXTextGenerator(model_name, trust_remote_code=trust_remote_code)

    print(f"Generating solutions for {len(examples)} examples...")

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

    prompts = [ex['prompt'] for ex in examples]
    outputs = generator.generate(prompts, generation_config, verbose=True)

    for ex, output in zip(examples, outputs):
        ex['output'] = output
    if getattr(generator, "last_finish_reasons", None):
        for ex, finish_reason in zip(examples, generator.last_finish_reasons):
            ex['finish_reason'] = finish_reason

    print(f"STAGE 1 complete: {len(examples)} generated in {time.time() - stage1_start:.1f}s")

    # ===== STAGE 2: Save Results =====
    print("\n" + "=" * 60)
    print("STAGE 2: Save results")

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "train.parquet")

    table = pa.Table.from_pylist(examples)
    pq.write_table(table, parquet_path)
    print(f"Saved {len(examples)} examples to {parquet_path}")

    # Print stats
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total examples:    {len(examples)}")
    print(f"  Generated:         {len(examples)}")
    print("=" * 60)

    # ===== STAGE 3: Post-process to JSONL =====
    print("\n" + "=" * 60)
    print("STAGE 3: Post-process to training JSONL")
    stage3_start = time.time()

    # Collect valid responses
    valid_records = [
        ex for ex in examples
        if ex.get('output') and str(ex['output']).strip()
    ]

    # Length filtering
    min_length = 0
    if filter_percent > 0 and valid_records:
        lengths = sorted(len(str(ex['output']).strip()) for ex in valid_records)
        cutoff_idx = min(int(len(lengths) * filter_percent / 100.0), len(lengths) - 1)
        min_length = lengths[cutoff_idx]
        print(f"  Filter: dropping bottom {filter_percent}% shortest (min_length={min_length})")

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")

    kept = 0
    filtered = 0
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
        with open(jsonl_path, "a" if kept > 0 else "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        kept += 1

    print(f"STAGE 3 complete in {time.time() - stage3_start:.1f}s")
    print(f"  Total records:  {len(examples)}")
    print(f"  Kept:           {kept} ({100*kept/len(examples):.1f}%)")
    print(f"  Filtered:       {filtered}")
    print(f"  Output:         {jsonl_path}")


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
