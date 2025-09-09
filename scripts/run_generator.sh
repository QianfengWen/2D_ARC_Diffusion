#!/usr/bin/env bash

# Simple wrapper to generate all ARC synthetic tasks with uniqueness.
#
# Usage examples:
#   ./run_generator.sh                           # default 5k/500 per task
#   ./run_generator.sh --seed 123 --out_dir out  # override defaults
#   ./run_generator.sh --n_train 100 --n_test 20 --attempts_per_example 80
#
# Any extra flags are passed directly to the Python CLI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-./.venv/bin/python}

DEFAULT_N_TRAIN=400
DEFAULT_N_TEST=100
DEFAULT_SEED=42
DEFAULT_ATTEMPTS=60
DEFAULT_OUT_DIR="arc_synth_400"

$PY "$ROOT_DIR/arc_synth_generators.py" make_all \
  --n_train "${DEFAULT_N_TRAIN}" \
  --n_test "${DEFAULT_N_TEST}" \
  --seed "${DEFAULT_SEED}" \
  --attempts_per_example "${DEFAULT_ATTEMPTS}" \
  --out_dir "${DEFAULT_OUT_DIR}" \
  "$@"

echo "Generation complete. Files written to: ${DEFAULT_OUT_DIR}"