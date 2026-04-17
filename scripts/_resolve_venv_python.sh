# Source-only: do not execute with bash; use from scripts that set ROOT_DIR first.
# Prefer the repo venv interpreter when PYTHON_BIN is unset (Linux / macOS).
# Prerequisites: ROOT_DIR is the repository root. Optional: VENV_DIR.
# shellcheck shell=bash
if [[ -z "${PYTHON_BIN:-}" ]]; then
  _adaptive_rl_venv_py="${VENV_DIR:-${ROOT_DIR}/.venv}/bin/python"
  if [[ -x "${_adaptive_rl_venv_py}" ]]; then
    PYTHON_BIN="${_adaptive_rl_venv_py}"
  else
    PYTHON_BIN="python3"
  fi
  unset _adaptive_rl_venv_py
fi
