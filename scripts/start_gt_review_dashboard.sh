#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HOST="${GT_REVIEW_HOST:-127.0.0.1}"
PORT="${GT_REVIEW_PORT:-8765}"

# Prefer Homebrew Python 3.11 if available (has cv2, numpy, jsonschema, pytest), else active python3
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "/opt/homebrew/opt/python@3.11/bin/python3.11" ]]; then
  PY="/opt/homebrew/opt/python@3.11/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "ERROR: Python 3 interpreter not found." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "${PY}" -m src.review_dashboard.server --host "${HOST}" --port "${PORT}" "$@"
