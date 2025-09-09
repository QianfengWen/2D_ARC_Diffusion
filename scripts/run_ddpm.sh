#!/usr/bin/env bash

# Wrapper to train or predict with 3-shot ddpm_arc.py (offline episodic shards)
#
# Usage:
#   ./run_ddpm.sh                              # train with defaults (offline shards)
#   ./run_ddpm.sh train --epochs 100           # override any args
#   ./run_ddpm.sh predict --ckpt CKPT.pt --episodes_dir episodes_3shot_g16
#
# Notes:
# - Defaults assume datasets generated into ./arc_synth_400 (from run_generator.sh)
# - Uses ./.venv/bin/python by default; set $PYTHON to override.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-./.venv/bin/python}

# Determine subcommand: train (default) or predict

# CMD="train"
CMD="predict"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ $# -gt 0 && ( "$1" == "train" || "$1" == "predict" ) ]]; then
  CMD="$1"
  shift
fi

if [[ "$CMD" == "train" ]]; then
  # Training defaults (offline episodes)
  EPISODES_DIR=${EPISODES_DIR:-episodes_3shot_g10_d4a91cb9_5000}
  BATCH_SIZE=${BATCH_SIZE:-512}
  EPOCHS=${EPOCHS:-200}
  LR=${LR:-2e-4}
  TIMESTEPS=${TIMESTEPS:-400}
  VAL_BATCHES=${VAL_BATCHES:-8}
  SAVE_EVERY=${SAVE_EVERY:-10}
  OUT_DIR=${OUT_DIR:-checkpoints_3shot_offline_g10_d4a91cb9_${TIMESTAMP}}
  mkdir -p "$OUT_DIR"
  echo "[DDPM-3shot OFFLINE] Training on: $EPISODES_DIR"
  "$PY" "$ROOT_DIR/ddpm_arc.py" train \
    --episodes_dir "$EPISODES_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --timesteps "$TIMESTEPS" \
    --val_batches "$VAL_BATCHES" \
    --save_every "$SAVE_EVERY" \
    --out_dir "$OUT_DIR" \
    "$@"

  echo "Training complete. Checkpoints in: $OUT_DIR"

else
  # Prediction defaults (offline episodes)
  CKPT=${CKPT:-checkpoints_3shot_offline_g10_d4a91cb9_20250908_230400/ddpm_3shot_best.pt}
  EPISODES_DIR=${EPISODES_DIR:-episodes_3shot_g10_d4a91cb9_5000}
  OUT_JSON=${OUT_JSON:-predictions_episodes/preds_d4a91cb9_5000.json}

  mkdir -p "$(dirname "$OUT_JSON")"
  echo "[DDPM-3shot OFFLINE] Predicting using: $CKPT on $EPISODES_DIR"

  "$PY" "$ROOT_DIR/ddpm_arc.py" predict \
    --ckpt "$CKPT" \
    --episodes_dir "$EPISODES_DIR" \
    --out_json "$OUT_JSON" \
    "$@"

  echo "Predictions written to: $OUT_JSON"
fi


