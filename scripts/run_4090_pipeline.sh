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

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is not installed. Install a CUDA-enabled PyTorch build on the 4090 host before running this script."
    ) from exc

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This script is intended for a GPU host.")

device_index = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(device_index)
total_memory_gb = round(torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3), 2)

print(f"CUDA device: {device_name}")
print(f"Device memory (GB): {total_memory_gb}")
if "4090" not in device_name.lower():
    print("Warning: active CUDA device does not look like an RTX 4090. The pipeline will still run with the fixed 4090 preset.")

from adaptive_quant.torch_policy import torch_cuda_diagnostics, validate_cuda_runtime_compatibility

diagnostics = torch_cuda_diagnostics("cuda")
print(f"Device capability: {diagnostics.get('device_capability', 'unknown')}")
print(f"Torch CUDA arch list: {', '.join(diagnostics.get('torch_cuda_arch_list') or ['unknown'])}")
try:
    validate_cuda_runtime_compatibility("cuda")
except RuntimeError as exc:
    raise SystemExit(str(exc)) from exc
PY

if [[ "${RUN_TESTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m unittest discover -s tests -t . -v
fi

"${PYTHON_BIN}" "${ROOT_DIR}/run_pytorch.py" --preset 4090
