#!/usr/bin/env bash
# Bootstrap from repo root (same as scripts/setup_from_clone.py).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/setup_from_clone.py" "$@"
