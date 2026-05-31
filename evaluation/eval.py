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
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote tokenizer code",
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
    from evaluation.mlx_generation import MLXTextGenerator

    logger.info(f"Loading model: {args.model}")
    generator = MLXTextGenerator(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = generator.tokenizer

    sampling_params = parse_sampling_params(args.sampling_params)
    seed = [int(s) for s in args.seed.split(",")]

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
    )

    logger.info("Starting evaluation...")
    start_time = time.time()
    results = benchmark.run()
    elapsed = time.time() - start_time

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
