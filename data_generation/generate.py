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

    batch_config = config.get('batch', {})
    completion_batch_size = batch_config.get('completion_batch_size', 32)
    prefill_batch_size = batch_config.get('prefill_batch_size', 8)
    prefill_step_size = batch_config.get('prefill_step_size', 2048)
    max_kv_size = batch_config.get('max_kv_size')

    filter_percent = config.get('post_process', {}).get('filter_shortest_percent', 10.0)

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{dataset_config} (split: {dataset_split})")
    print(f"Output: {output_dir}")
    print(f"Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, "
          f"min_p={min_p}, rep_penalty={repetition_penalty}, max_tokens={max_tokens}")
    print(f"MLX batch: completion={completion_batch_size}, prefill={prefill_batch_size}, "
          f"prefill_step={prefill_step_size}, max_kv_size={max_kv_size}")
    print("-" * 60)

    # Load templates
    stdin_template, function_template = load_templates(template_dir)
    print("Templates loaded.")

    # ===== STAGE 0: Load and Format Data =====
    print("\n" + "=" * 60)
    print("STAGE 0: Load data + format prompts")
    stage0_start = time.time()

    examples = load_hf_dataset(dataset_name, dataset_config, split=dataset_split)
    if limit > 0:
        examples = examples[:limit]
        print(f"Limited to {len(examples)} examples")
    print(f"Total examples: {len(examples)}")

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
    )

    prompts = [ex['prompt'] for ex in examples]
    outputs = generator.generate(prompts, generation_config, verbose=True)

    for ex, output in zip(examples, outputs):
        ex['output'] = output

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
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
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

    # Resolve template directory (templates/ next to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "templates")

    generate(config, template_dir, limit=args.limit)


if __name__ == "__main__":
    main()
