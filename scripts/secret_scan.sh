#!/usr/bin/env bash
# Compatibility wrapper for the cross-platform Python implementation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/secret_scan.py" "$@"
