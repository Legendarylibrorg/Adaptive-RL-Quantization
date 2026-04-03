#!/usr/bin/env bash
# Run before `git commit` (Linux/macOS). Exits non-zero on failure.
# Interpreter: sources scripts/_resolve_venv_python.sh — set PYTHON_BIN to force (CI uses PYTHON_BIN=python).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
# shellcheck source=scripts/_resolve_venv_python.sh
source "${ROOT_DIR}/scripts/_resolve_venv_python.sh"

echo "== git diff --check (unstaged whitespace / conflict markers) =="
git diff --check
echo "== git diff --cached --check (staged; re-stage after fixes: git add -u) =="
if ! git diff --cached --quiet 2>/dev/null; then
  git diff --cached --check
fi

echo "== Python compile (syntax) =="
"${PYTHON_BIN}" -m compileall -q adaptive_quant analysis
shopt -s nullglob
for f in *.py; do
  "${PYTHON_BIN}" -m py_compile "$f"
done
shopt -u nullglob

echo "== Bash scripts (syntax) =="
for s in scripts/*.sh; do
  bash -n "$s"
done

echo "== unittest =="
"${PYTHON_BIN}" -m unittest discover -s tests -q

echo "OK: pre_commit_check.sh finished successfully."
