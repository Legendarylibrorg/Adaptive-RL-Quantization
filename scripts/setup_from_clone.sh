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
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! python3 -m pip --version >/dev/null 2>&1; then
  echo " pip missing in venv; bootstrapping with curl + get-pip.py ..."
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

python3 -m pip install -U pip setuptools
python3 -m pip install -e .

python3 -m unittest discover -s tests -q
python3 run_research.py --config "${ROOT_DIR}/config.e2e_smoke.json"

echo ""
echo "OK: venv, editable install, tests, and end-to-end RL smoke (train → eval → benchmarks) finished."
echo "   Tune RL: edit config.e2e_smoke.json (episodes, run_name) or copy it to my.json and run:"
echo "   python3 run_research.py --config my.json"
