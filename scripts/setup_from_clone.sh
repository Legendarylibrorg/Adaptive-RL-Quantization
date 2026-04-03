#!/usr/bin/env bash
# One-shot path: clone repo -> check git/curl/python -> venv -> pip (curl bootstrap if needed) -> install -> tests -> short RL pipeline.
# Linux-first. From repo root: bash scripts/setup_from_clone.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "install git first — e.g. Debian/Ubuntu: sudo apt install -y git curl python3 python3-venv (see docs/INSTALL.md)"
command -v curl >/dev/null 2>&1 || die "install curl first — used to bootstrap pip in minimal venvs (see docs/INSTALL.md)"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "no '${PYTHON_BIN}' on PATH (need Python 3.11+)"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
[[ -x "${VENV_PY}" ]] || die "venv python missing: ${VENV_PY}"

if ! "${VENV_PY}" -m pip --version >/dev/null 2>&1; then
  echo " pip missing in venv; bootstrapping with curl + get-pip.py ..."
  curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_PY}"
fi

"${VENV_PY}" -m pip install -U pip
"${VENV_PY}" -m pip install -e .

"${VENV_PY}" -m unittest discover -s tests -q
"${VENV_PY}" "${ROOT_DIR}/run_research.py" --config "${ROOT_DIR}/config.e2e_smoke.json"

echo ""
echo "OK: venv, editable install, tests, and reproducible E2E RL smoke (train → eval → benchmarks → analysis) finished."
echo "   Tune RL: edit config.e2e_smoke.json (episodes, seed, run_name) or copy it to my.json and run:"
echo "   source ${VENV_DIR}/bin/activate   # then:"
echo "   python run_research.py --config my.json"
