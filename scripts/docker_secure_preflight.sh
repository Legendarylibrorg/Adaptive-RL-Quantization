#!/usr/bin/env bash
# Verify Docker / Compose security contract before hardened runs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHECK_GPU=0
for arg in "$@"; do
  case "${arg}" in
    --gpu) CHECK_GPU=1 ;;
    -h | --help)
      echo "Usage: $0 [--gpu]"
      echo "  Checks Docker, Compose hardening, merged GPU overlay, and optionally NVIDIA runtime."
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "docker_secure_preflight: $*" >&2
  exit 1
}

ok() {
  echo "docker_secure_preflight: ok - $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

require_compose_key() {
  local file="$1"
  local needle="$2"
  if ! grep -F -- "${needle}" "${file}" >/dev/null 2>&1; then
    fail "${file} missing required setting: ${needle}"
  fi
}

require_cmd docker
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  fail "docker compose (plugin) or docker-compose required"
fi

BASE_COMPOSE="${ROOT_DIR}/docker-compose.yml"
GPU_COMPOSE="${ROOT_DIR}/docker-compose.gpu.yml"
[[ -f "${BASE_COMPOSE}" ]] || fail "missing ${BASE_COMPOSE}"
[[ -f "${GPU_COMPOSE}" ]] || fail "missing ${GPU_COMPOSE}"
[[ -f "${ROOT_DIR}/scripts/docker_gpu_device_probe.py" ]] || fail "missing scripts/docker_gpu_device_probe.py"

for key in \
  'user: "10001:10001"' \
  'privileged: false' \
  'read_only: true' \
  'cap_drop:' \
  '- ALL' \
  'no-new-privileges:true' \
  'pids_limit:' \
  '/tmp:rw,noexec,nosuid,nodev' \
  'adaptive_outputs:/app/outputs'; do
  require_compose_key "${BASE_COMPOSE}" "${key}"
done

require_compose_key "${GPU_COMPOSE}" 'NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-0}'
require_compose_key "${GPU_COMPOSE}" 'driver: nvidia'
require_compose_key "${GPU_COMPOSE}" 'gpus:'
require_compose_key "${GPU_COMPOSE}" 'config.docker.gpu_smoke.json'

if grep -Eiq 'privileged:\s*true|/var/run/docker\.sock|\$HOME|~/.ssh' "${BASE_COMPOSE}" "${GPU_COMPOSE}"; then
  fail "compose files must not enable privileged mode or mount host secrets"
fi

merged="$(
  "${COMPOSE[@]}" -f "${BASE_COMPOSE}" -f "${GPU_COMPOSE}" config 2>/dev/null
)" || fail "docker compose config failed (is the Docker daemon running?)"

for key in \
  'read_only: true' \
  'cap_drop:' \
  'no-new-privileges:true' \
  'user: "10001:10001"'; do
  if ! grep -F -- "${key}" <<<"${merged}" >/dev/null 2>&1; then
    fail "merged compose config missing ${key} (GPU overlay must not weaken base hardening)"
  fi
done
if grep -Eiq 'privileged:\s*true' <<<"${merged}"; then
  fail "merged compose config must not set privileged: true"
fi

if [[ "${CHECK_GPU}" -eq 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "--gpu: nvidia-smi not found; install the NVIDIA driver in the VM first"
  fi
  nvidia-smi >/dev/null 2>&1 || fail "--gpu: nvidia-smi failed"

  runtime_ok=0
  if command -v nvidia-container-cli >/dev/null 2>&1 && nvidia-container-cli info >/dev/null 2>&1; then
    runtime_ok=1
  elif docker info 2>/dev/null | grep -Eiq 'nvidia|nvidia-container'; then
    runtime_ok=1
  fi

  if [[ "${runtime_ok}" -eq 0 ]]; then
    fail "--gpu: NVIDIA container runtime not detected; install NVIDIA Container Toolkit in the VM"
  fi
  ok "Docker, merged Compose hardening, and NVIDIA container runtime"
else
  ok "Docker, Compose hardening, and merged GPU overlay (pass --gpu inside a GPU VM to validate runtime)"
fi
