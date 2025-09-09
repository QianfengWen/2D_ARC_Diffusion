#!/usr/bin/env bash

# Wrapper to build offline 3+1 episodic shards from ARC synth JSONs.
#
# Examples:
#   ./run_make_episodes.sh                                       # build with defaults
#   ./run_make_episodes.sh GLOB='arc_synth_400/synth_*.json' GRID_SIZE=10 OUT_DIR=episodes_3shot_g10
#   ./run_make_episodes.sh GLOB='arc_synth_400/synth_*.json' TRAIN_PER_TASK=400 TEST_PER_TASK=100 SHARD_SIZE=5000
#
# Notes:
# - Quote or escape the glob so the shell doesn't expand it: 'arc_synth_400/synth_*.json'
# - Uses ./.venv/bin/python unless $PYTHON is set.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-./.venv/bin/python}

# Defaults
# GLOB=${GLOB:-'arc_synth_400/synth_*.json'}
GLOB=${GLOB:-'arc_synth_400/synth_d4a91cb9.json'}
GRID_SIZE=${GRID_SIZE:-10}
CTX_POLICY=${CTX_POLICY:-random}        # first3|random|sliding
TRAIN_PER_TASK=${TRAIN_PER_TASK:-5000}
TEST_PER_TASK=${TEST_PER_TASK:-100}
SHARD_SIZE=${SHARD_SIZE:-5000}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-episodes_3shot_g10_d4a91cb9_5000}

echo "[EPISODES] Building shards from: $GLOB"
"$PY" "$ROOT_DIR/make_arc_episodes_offline.py" \
  --glob "$GLOB" \
  --grid_size "$GRID_SIZE" \
  --ctx_policy "$CTX_POLICY" \
  --train_per_task "$TRAIN_PER_TASK" \
  --test_per_task "$TEST_PER_TASK" \
  --shard_size "$SHARD_SIZE" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR" \
  "$@"

echo "Done. Shards and meta.json written to: $OUT_DIR"


