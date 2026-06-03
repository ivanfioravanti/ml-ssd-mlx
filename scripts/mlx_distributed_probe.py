#!/usr/bin/env python
"""Smoke-test MLX distributed collectives and optional sharded model loading."""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.mlx_generation import MLXGenerationConfig, MLXTextGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe MLX distributed transport and optional tensor-parallel MLX-LM generation."
    )
    parser.add_argument(
        "--backend",
        default="jaccl",
        choices=("jaccl",),
        help="MLX distributed backend to initialize.",
    )
    parser.add_argument(
        "--payload-mb",
        type=int,
        default=16,
        help="Payload size for the all_sum bandwidth smoke test.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of all_sum timing rounds.",
    )
    parser.add_argument(
        "--model",
        help="Optional MLX model id/path to load with mlx_lm.utils.sharded_load.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that returns the nth Fibonacci number.",
        help="Prompt used when --model is provided.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Generated tokens for the optional model probe.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to the tokenizer loader.",
    )
    return parser.parse_args()


def log(rank: int, message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or rank == 0:
        print(f"[rank {rank}] {message}", flush=True)


def barrier(group: mx.distributed.Group) -> None:
    token = mx.distributed.all_sum(mx.array(1, dtype=mx.int32), group=group)
    mx.eval(token)


def collective_probe(group: mx.distributed.Group, payload_mb: int, rounds: int) -> None:
    rank = group.rank()
    size = group.size()

    rank_value = mx.array([rank + 1], dtype=mx.float32)
    summed = mx.distributed.all_sum(rank_value, group=group)
    gathered = mx.distributed.all_gather(mx.array([rank], dtype=mx.int32), group=group)
    mx.eval(summed, gathered)
    log(
        rank,
        f"collectives ok: world_size={size}, all_sum={summed.tolist()}, "
        f"all_gather={gathered.tolist()}",
    )

    if payload_mb <= 0 or rounds <= 0:
        return

    element_count = payload_mb * 1024 * 1024 // 4
    payload = mx.ones((element_count,), dtype=mx.float32)
    barrier(group)

    start = time.perf_counter()
    for _ in range(rounds):
        reduced = mx.distributed.all_sum(payload, group=group)
        mx.eval(reduced)
    barrier(group)
    elapsed = time.perf_counter() - start

    gb = payload.nbytes * rounds / 1e9
    log(
        rank,
        f"all_sum timing: payload={payload_mb} MiB, rounds={rounds}, "
        f"elapsed={elapsed:.3f}s, local_payload_rate={gb / elapsed:.3f} GB/s",
    )


def model_probe(args: argparse.Namespace, group: mx.distributed.Group) -> None:
    rank = group.rank()
    log(
        rank,
        f"loading sharded model: model={args.model}, backend={args.backend}, "
        f"world_size={group.size()}",
    )

    generator = MLXTextGenerator(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_mlx_distributed=True,
        distributed_backend=args.backend,
        distributed_group=group,
    )
    config = MLXGenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.max_tokens,
        completion_batch_size=1,
        prefill_batch_size=1,
    )
    outputs = generator.generate([args.prompt], config, verbose=(rank == 0))
    if rank == 0:
        print("\n=== rank 0 generated sample ===", flush=True)
        print(outputs[0], flush=True)
        print("=== end sample ===", flush=True)


def main() -> None:
    args = parse_args()
    group = mx.distributed.init(strict=True, backend=args.backend)
    rank = group.rank()
    log(rank, f"initialized backend={args.backend}, world_size={group.size()}", all_ranks=True)

    collective_probe(group, args.payload_mb, args.rounds)
    if args.model:
        model_probe(args, group)
    barrier(group)
    log(rank, "probe complete")


if __name__ == "__main__":
    main()
