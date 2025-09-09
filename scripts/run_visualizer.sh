#!/usr/bin/env bash

# Wrapper to visualize/export PNGs from:
# - ARC JSON datasets (train/test pairs)
# - Offline predictions JSON (test_episodes_outputs)
# - Episodic shard .pt files (contexts + query GT)
#
# Modes:
#   show       - Show pairs from a single JSON (opens a window)
#   save       - Save PNGs for a single JSON (both splits by default)
#   save_all   - Save PNGs for all JSONs matching DATA_GLOB (default)
#
# Examples:
#   # Dataset JSONs
#   ./run_visualizer.sh                                     # save_all dataset pairs with defaults
#   ./run_visualizer.sh show arc_synth_400/synth_bb43febb.json --max 10 --diff
#   ./run_visualizer.sh save arc_synth_400/synth_bb43febb.json
#   ./run_visualizer.sh save_all DATA_GLOB=arc_synth_400/synth_*.json OUT_BASE=viz_out CELL_SIZE=32
#
#   # Offline predictions (from episodes directory)
#   ./run_ddpm.sh predict EPISODES_DIR=episodes_3shot_g16 OUT_JSON=predictions_episodes/preds.json
#   ./run_visualizer.sh save_offline predictions_episodes/preds.json --out viz_out/preds_offline --max 50
#
#   # Episodic shard preview (contexts + query GT)
#   ./run_visualizer.sh save_shard episodes_3shot_g16/train_0000.pt --out viz_out/episodes_g16_train_0000 --max 32
#
# Uses ./.venv/bin/python by default; set $PYTHON to override.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-./.venv/bin/python}

# Defaults (dataset JSON modes)
DATA_GLOB=${DATA_GLOB:-arc_synth_400/synth_*.json}
OUT_BASE=${OUT_BASE:-viz_out}
CELL_SIZE=${CELL_SIZE:-32}
DPI=${DPI:-100}
# For dataset visualization leave as is (train/test).
# For offline predictions, arc_visualizer currently expects ARC-style pairs; the
# offline predictions file is a different JSON (test_episodes_outputs). To visualize
# those, you can write a small converter or extend arc_visualizer; for now we keep
# standard split behavior.
SPLITS=${SPLITS:-test}

cmd="save_all"
if [[ $# > 0 && ( "$1" == "show" || "$1" == "save" || "$1" == "save_all" || "$1" == "save_offline" || "$1" == "save_shard" ) ]]; then
  cmd="$1"
  shift
fi

show_usage() {
  echo "Usage: $0 [show|save|save_all] [args...]"
}

# Remove leading 'synth_' from a basename to get a clean task id
task_from_json() {
  local json="$1"
  local stem
  stem="$(basename "$json" .json)"
  echo "${stem#synth_}"
}

if [[ "$cmd" == "show" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Error: show requires a JSON path" >&2
    show_usage; exit 1
  fi
  json="$1"; shift
  "$PY" "$ROOT_DIR/arc_visualizer.py" show "$json" "$@"
  exit 0
fi

if [[ "$cmd" == "save" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Error: save requires a JSON path" >&2
    show_usage; exit 1
  fi
  json="$1"; shift
  task="$(task_from_json "$json")"
  out_dir="$OUT_BASE/$task"
  mkdir -p "$out_dir"
  IFS=',' read -r -a splits_arr <<< "$SPLITS"
  for split in "${splits_arr[@]}"; do
    echo "[viz] Saving $split images for $json → $out_dir"
    "$PY" "$ROOT_DIR/arc_visualizer.py" save \
      "$json" \
      --split "$split" \
      --out "$out_dir" \
      --prefix "$task" \
      --cell "$CELL_SIZE" \
      --dpi "$DPI" \
      "$@"
  done
  echo "Saved images under: $out_dir"
  exit 0
fi

# save_offline (predictions JSON with test_episodes_outputs)
if [[ "$cmd" == "save_offline" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Error: save_offline requires a predictions JSON path" >&2
    show_usage; exit 1
  fi
  pred_json="$1"; shift
  prefix="${PREFIX:-$(basename "${pred_json}" .json)}"
  out_dir="${OUT_OFF:-viz_out/preds_offline}"
  mkdir -p "$out_dir"
  echo "[viz] Saving offline predictions for $pred_json → $out_dir"
  "$PY" "$ROOT_DIR/arc_visualizer.py" save_offline \
    "$pred_json" \
    --out "$out_dir" \
    --prefix "$prefix" \
    --cell "$CELL_SIZE" \
    --dpi "$DPI" \
    "$@"
  echo "Saved offline prediction images under: $out_dir"
  exit 0
fi

# save_shard (episodic shard .pt)
if [[ "$cmd" == "save_shard" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Error: save_shard requires a shard .pt path" >&2
    show_usage; exit 1
  fi
  shard_path="$1"; shift
  out_dir="${OUT_SHARD:-viz_out/${shard_path##*/}}"
  prefix="${PREFIX:-$(basename "${shard_path}" .pt)}"
  mkdir -p "$out_dir"
  echo "[viz] Saving shard previews for $shard_path → $out_dir"
  "$PY" "$ROOT_DIR/arc_visualizer.py" save_shard \
    "$shard_path" \
    --out "$out_dir" \
    --prefix "$prefix" \
    "$@"
  echo "Saved shard images under: $out_dir"
  exit 0
fi

# save_all
shopt -s nullglob
jsons=( $DATA_GLOB )
if [[ ${#jsons[@]} -eq 0 ]]; then
  echo "No files matched DATA_GLOB: $DATA_GLOB" >&2
  exit 1
fi

IFS=',' read -r -a splits_arr <<< "$SPLITS"
for json in "${jsons[@]}"; do
  task="$(task_from_json "$json")"
  out_dir="$OUT_BASE/$task"
  mkdir -p "$out_dir"
  for split in "${splits_arr[@]}"; do
    echo "[viz] Saving $split images for $json → $out_dir"
    "$PY" "$ROOT_DIR/arc_visualizer.py" save \
      "$json" \
      --split "$split" \
      --out "$out_dir" \
      --prefix "$task" \
      --cell "$CELL_SIZE" \
      --dpi "$DPI"
  done
done

echo "All visualizations saved under: $OUT_BASE"


