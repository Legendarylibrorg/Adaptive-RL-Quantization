#!/usr/bin/env bash
# Linux + NVIDIA: CUDA checks, optional unittest, then run_pytorch.py --preset 4090.
# Uses scripts/_resolve_venv_python.sh when PYTHON_BIN is unset (prefers .venv).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
# shellcheck source=scripts/_resolve_venv_python.sh
source "${ROOT_DIR}/scripts/_resolve_venv_python.sh"
RUN_TESTS="${RUN_TESTS:-1}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from adaptive_quant.torch_install import INSTALL_CUDA_TORCH_SCRIPT, torch_cuda_ready_report, validate_cuda_after_install

report = torch_cuda_ready_report()
if not report.get("torch_installed", False):
    raise SystemExit(
        "PyTorch is not installed. On Linux + NVIDIA run:\n"
        f"  {INSTALL_CUDA_TORCH_SCRIPT}\n"
        "then retry this script."
    )

if not report.get("cuda_available"):
    raise SystemExit(
        "CUDA is not available. Confirm `nvidia-smi` works, then install a CUDA torch wheel:\n"
        f"  {INSTALL_CUDA_TORCH_SCRIPT}"
    )

print(f"CUDA device: {report.get('device_name', 'unknown')}")
print(f"Device capability: {report.get('device_capability', 'unknown')}")
arch_list = report.get("torch_cuda_arch_list") or report.get("arch_list") or []
print(f"Torch CUDA arch list: {', '.join(arch_list) or 'unknown'}")
device_name = str(report.get("device_name", ""))
if "4090" not in device_name.lower():
    print(
        "Warning: active CUDA device does not look like an RTX 4090. "
        "The pipeline will still run with the fixed 4090 preset."
    )

try:
    validate_cuda_after_install("cuda")
except RuntimeError as exc:
    raise SystemExit(str(exc)) from exc
PY

if [[ "${RUN_TESTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m unittest discover -s tests -t . -v
fi

"${PYTHON_BIN}" "${ROOT_DIR}/run_pytorch.py" --preset 4090
