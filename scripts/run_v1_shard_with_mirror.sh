#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <shard-index>" >&2
    exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/Users/ifioravanti/github/ml-ssd-mlx}"
CONFIG="${CONFIG:-$PROJECT_DIR/data_generation/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output}"
RUN_ID="${RUN_ID:-qwen3-4b-ssd-full-10k-b12-v1x2}"
NUM_SHARDS="${NUM_SHARDS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
LIMIT="${LIMIT:-0}"
MIRROR_HOST="${MIRROR_HOST:-}"
MIRROR_INTERVAL="${MIRROR_INTERVAL:-60}"

SHARD_INDEX="$1"
SHARD_NAME="$(printf "shard-%03d" "$SHARD_INDEX")"
SHARD_DIR="$OUTPUT_ROOT/$RUN_ID/shards/$SHARD_NAME"
LOG_PATH="${LOG_PATH:-$SHARD_DIR/generation.log}"

mkdir -p "$SHARD_DIR"

sync_once() {
    [[ -n "$MIRROR_HOST" ]] || return 0
    ssh "$MIRROR_HOST" "mkdir -p '$SHARD_DIR'"
    rsync -az \
        --exclude '*.tmp' \
        --exclude '.DS_Store' \
        "$SHARD_DIR/" \
        "$MIRROR_HOST:$SHARD_DIR/"
}

mirror_pid=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$mirror_pid" ]]; then
        kill "$mirror_pid" 2>/dev/null || true
        wait "$mirror_pid" 2>/dev/null || true
        sync_once || true
    fi
    exit "$status"
}

if [[ -n "$MIRROR_HOST" ]]; then
    sync_once
    (
        while true; do
            sleep "$MIRROR_INTERVAL"
            sync_once || true
        done
    ) &
    mirror_pid="$!"
    trap cleanup EXIT INT TERM
fi

cd "$PROJECT_DIR"

generate_args=(
    --config "$CONFIG"
    --distributed-output-root "$OUTPUT_ROOT"
    --distributed-run-id "$RUN_ID"
    --distributed-num-shards "$NUM_SHARDS"
    --distributed-shard-index "$SHARD_INDEX"
    --resume
    --checkpoint-every "$CHECKPOINT_EVERY"
)

if [[ "$LIMIT" != "0" ]]; then
    generate_args+=(--limit "$LIMIT")
fi

set +e
PYTHONUNBUFFERED=1 uv run python data_generation/generate.py \
    "${generate_args[@]}" \
    2>&1 | tee -a "$LOG_PATH"
status="${PIPESTATUS[0]}"
set -e

sync_once || true
exit "$status"
