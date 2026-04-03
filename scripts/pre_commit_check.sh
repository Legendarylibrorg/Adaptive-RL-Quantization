#!/usr/bin/env bash
# Run before `git commit` (Linux/macOS). Exits non-zero on failure.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

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
