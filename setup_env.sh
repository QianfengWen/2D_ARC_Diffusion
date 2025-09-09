#!/usr/bin/env bash

# Create or reuse a Python virtual environment and install requirements
#
# Usage:
#   bash 2D_ARC_Diffusion/setup_env.sh              # uses ./.venv in current dir
#   VENV_DIR=.venv bash 2D_ARC_Diffusion/setup_env.sh
#   PYTHON_BIN=python3.10 bash 2D_ARC_Diffusion/setup_env.sh
#   PIP_EXTRA_ARGS="--extra-index-url https://download.pytorch.org/whl/cu121" \
#     bash 2D_ARC_Diffusion/setup_env.sh
#
# Notes:
# - By default, the venv lives at "$PWD/.venv" so wrapper scripts that use
#   ${PYTHON:-./.venv/bin/python} will find it when run from the repo root.
# - Set VENV_DIR to override the venv location.
# - Set PIP_EXTRA_ARGS to pass extra flags to pip install (e.g., CUDA wheels).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements.txt}"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional first arg overrides VENV_DIR for convenience
if [[ $# -ge 1 ]]; then
  VENV_DIR="$1"
fi

echo "[setup] Using virtualenv at: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating venv with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[setup] Directory exists but missing Python; recreating venv"
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    echo "[setup] Reusing existing venv"
  fi
fi

PY="${VENV_DIR}/bin/python"

echo "[setup] Upgrading pip/setuptools/wheel"
"${PY}" -m pip install --upgrade pip setuptools wheel ${PIP_EXTRA_ARGS:-}

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[setup] ERROR: requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

echo "[setup] Installing requirements from ${REQ_FILE}"
"${PY}" -m pip install -r "${REQ_FILE}" ${PIP_EXTRA_ARGS:-}

echo "[setup] Done. Activate with: source '${VENV_DIR}/bin/activate'"
echo "[setup] Or run wrappers that use \$PYTHON (defaults to ${VENV_DIR}/bin/python)."

