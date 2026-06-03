#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Evaluation entry point for LiveCodeBench v6 using MLX-LM."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on LiveCodeBench v6 using MLX-LM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output_path", type=str, default="./results", help="Results directory")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum generation length")
    parser.add_argument("--n_repeat", type=int, default=20, help="Samples per problem for pass@k")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of problems (0 = all)")
    parser.add_argument(
        "--sampling_params",
        type=str,
        default="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0",
        help="Generation params as key=value pairs (e.g., 'temperature=0.6,top_p=0.95,top_k=20,min_p=0.0')",
    )
    parser.add_argument("--seed", type=str, default="0,1234,1234,1234", help="Random seeds (comma-separated)")
    parser.add_argument("--completion_batch_size", type=int, default=32, help="MLX-LM completion batch size")
    parser.add_argument("--prefill_batch_size", type=int, default=8, help="MLX-LM prefill batch size")
    parser.add_argument("--prefill_step_size", type=int, default=2048, help="MLX-LM prompt prefill step size")
    parser.add_argument("--max_kv_size", type=int, default=None, help="Optional MLX-LM rotating KV cache size")
    parser.add_argument(
        "--lcb_data_files",
        type=str,
        default="",
        help="Optional comma-separated LiveCodeBench JSONL files for smoke tests, e.g. test.jsonl",
    )
    parser.add_argument(
        "--lcb_allow_any_date",
        action="store_true",
        help="Smoke-test option: skip the LiveCodeBench v6 contest-date filter",
    )
    parser.add_argument(
        "--lcb_preprocess_num_proc",
        type=int,
        default=0,
        help="Number of processes for LiveCodeBench preprocessing (0 = auto, distributed defaults to 1)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote tokenizer code",
    )
    parser.add_argument(
        "--mlx_distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MLX JACCL tensor-parallel model loading under mlx.launch",
    )
    parser.add_argument(
        "--mlx_distributed_backend",
        choices=("jaccl",),
        default="jaccl",
        help="MLX distributed backend for tensor-parallel inference",
    )
    return parser.parse_args()


def parse_sampling_params(sampling_params_str: str) -> Dict[str, Any]:
    """Parse sampling parameters from 'key=value,key=value' format."""
    float_keys = {"temperature", "top_p", "min_p"}
    int_keys = {"top_k"}
    valid_keys = float_keys | int_keys

    result = {}
    for pair in sampling_params_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid format: '{pair}'. Expected 'key=value'")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in valid_keys:
            raise ValueError(f"Unknown sampling parameter: '{key}'. Valid: {sorted(valid_keys)}")
        if key in float_keys:
            result[key] = float(value)
        elif key in int_keys:
            result[key] = int(value)
    return result


def save_results(results: Dict, config: Dict, output_path: str, model_name: str):
    """Save evaluation results to JSON."""
    path = Path(output_path) / model_name.replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    result_file = path / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    result_file.write_text(
        json.dumps({"results": results, "config": config, "date": time.time()}, indent=2, default=str)
    )
    logger.info(f"Results saved to {result_file}")


def main():
    args = parse_args()

    from evaluation.benchmark import LiveCodeBenchV6
    from evaluation.mlx_generation import (
        MLXTextGenerator,
        distributed_barrier,
        init_mlx_distributed_group,
    )

    if args.mlx_distributed and args.mlx_distributed_backend != "jaccl":
        raise ValueError("--mlx_distributed_backend must be jaccl")

    distributed_group = (
        init_mlx_distributed_group(args.mlx_distributed_backend)
        if args.mlx_distributed
        else None
    )
    distributed_rank = distributed_group.rank() if distributed_group is not None else 0
    distributed_world_size = distributed_group.size() if distributed_group is not None else 1
    distributed_is_rank0 = distributed_rank == 0

    logger.info(f"Loading model: {args.model}")
    if args.mlx_distributed:
        logger.info(
            "MLX distributed enabled: backend=%s rank=%s/%s",
            args.mlx_distributed_backend,
            distributed_rank,
            distributed_world_size,
        )
    generator = MLXTextGenerator(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_mlx_distributed=args.mlx_distributed,
        distributed_backend=args.mlx_distributed_backend,
        distributed_group=distributed_group,
    )
    tokenizer = generator.tokenizer

    sampling_params = parse_sampling_params(args.sampling_params)
    seed = [int(s) for s in args.seed.split(",")]
    lcb_data_files = [
        file_name.strip()
        for file_name in args.lcb_data_files.split(",")
        if file_name.strip()
    ] or None
    lcb_preprocess_num_proc = args.lcb_preprocess_num_proc
    if lcb_preprocess_num_proc < 0:
        raise ValueError("--lcb_preprocess_num_proc must be >= 0")
    if lcb_preprocess_num_proc == 0:
        lcb_preprocess_num_proc = 1 if args.mlx_distributed else None

    benchmark = LiveCodeBenchV6(
        generator=generator,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        n_repeat=args.n_repeat,
        limit=args.limit,
        sampling_params=sampling_params,
        seed=seed,
        completion_batch_size=args.completion_batch_size,
        prefill_batch_size=args.prefill_batch_size,
        prefill_step_size=args.prefill_step_size,
        max_kv_size=args.max_kv_size,
        lcb_data_files=lcb_data_files,
        lcb_filter_contest_date=not args.lcb_allow_any_date,
        lcb_preprocess_num_proc=lcb_preprocess_num_proc,
    )

    logger.info("Starting evaluation...")
    start_time = time.time()
    results = benchmark.run(
        run_evaluation=distributed_is_rank0,
        before_generate=lambda: distributed_barrier(distributed_group),
        after_generate=lambda: distributed_barrier(distributed_group),
    )
    elapsed = time.time() - start_time

    if args.mlx_distributed and not distributed_is_rank0:
        logger.info(
            "MLX distributed rank %s completed generation; rank 0 will evaluate and save results.",
            distributed_rank,
        )
        return

    save_results(results, vars(args), args.output_path, args.model)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    for k in [1, 5, 10, 16, 20, 32]:
        key = f"pass@{k}"
        if key in results and isinstance(results[key], float):
            print(f"{key}: {results[key]:.2%}")

    for k in [1, 5, 10, 16, 20, 32]:
        for key in sorted(results.keys()):
            if key.startswith(f"pass@{k}_"):
                diff = key[len(f"pass@{k}_"):]
                print(f"  {key}: {results[key]:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
