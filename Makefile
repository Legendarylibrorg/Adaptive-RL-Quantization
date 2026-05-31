# Research-oriented Makefile (Linux/macOS; on Windows use the Python scripts under scripts/). From repo root: `make help`
# Override interpreter:  PY=python3.12 make test

.PHONY: help
.PHONY: setup setup-quick
.PHONY: install install-dev install-torch
.PHONY: run reproduce smoke run-config moe multiseed multiseed-smoke sweep sweep-smoke
.PHONY: pytorch 3090 4090 4090-universal
.PHONY: online calibrate route-help
.PHONY: test test-quiet lint format check secret-scan doctor
.PHONY: docker-preflight docker-build docker-test docker-smoke docker-no-network-smoke
.PHONY: docker-gpu-preflight docker-gpu-build docker-gpu-smoke docker-gpu-verify docker-gpu-test docker-gpu-pytorch
.PHONY: outputs-clean clean-venv

# After ./setup.sh, prefer the repo venv so `make run` works without activating it.
_VENV_PY := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/Scripts/python.exe))
ifneq ($(_VENV_PY),)
PY ?= $(_VENV_PY)
else
PY ?= python3
endif
PKG := src/adaptive_quant src/analysis tests
SCRIPTS_PY := $(wildcard scripts/*.py)
RUN := $(wildcard run_*.py)
CFG := $(wildcard src/config*.py)

# Multiseed: preset dense|moe ; seeds "a,b,c" or "0-4"
MULTISEED_PRESET ?= dense
MULTISEED_SEEDS ?= 13,17,23

# PyTorch entrypoint preset: gpu | 3090 | 4090 | 4090-universal
PYTORCH_PRESET ?= gpu

help:
	@echo "Adaptive RL Quantization — research Makefile"
	@echo "Override PY=... for a non-default interpreter."
	@echo ""
	@echo "[Bootstrap]"
	@echo "  make setup            venv + editable install + tests + E2E smoke (./setup.sh)"
	@echo "  make setup-quick      venv + editable install only (no tests/smoke)"
	@echo "  make install          pip install -e . (simulator path; assumes venv active)"
	@echo "  make install-dev      + Ruff (lint/format)"
	@echo "  make install-torch    + PyTorch extra (CUDA wheel still your job)"
	@echo ""
	@echo "[Experiments — simulator / stdlib trainer]"
	@echo "  make run              full run: ./run or run_research.py (config.py)"
	@echo "  make reproduce        CI-equivalent smoke: config.e2e_smoke.json (alias: make smoke)"
	@echo "  make run-config       RESEARCH_CONFIG=path.json|toml (required)"
	@echo "  make moe              run_moe_research.py (MoE benchmarks)"
	@echo "  make multiseed        MULTISEED_PRESET=dense|moe MULTISEED_SEEDS=..."
	@echo "  make multiseed-smoke  two seeds, low episodes (quick sanity)"
	@echo "  make sweep            SWEEP_CONFIG=config.sweep.example.json (override as needed)"
	@echo "  make sweep-smoke      two learning rates, low episodes (quick sanity)"
	@echo ""
	@echo "[CUDA — install torch first: make install-torch]"
	@echo "  make pytorch          PYTORCH_PRESET=gpu (default) | 3090 | 4090 | 4090-universal"
	@echo "  make 3090             same as pytorch with preset 3090 (RTX 3090)"
	@echo "  make 4090             same as pytorch with preset 4090"
	@echo "  make 4090-universal   multi-hardware 4090-host preset"
	@echo ""
	@echo "[Other runners]"
	@echo "  make online           supported online adaptation pipeline"
	@echo "  make calibrate        llama.cpp calibration (needs binary+model in config)"
	@echo "  make route-help       GGUF route catalog + contextual bandit help"
	@echo ""
	@echo "[Quality]"
	@echo "  make test | test-quiet | secret-scan | lint | format | check"
	@echo "  make doctor           env summary: Python path, torch/ruff, git, outputs/* counts"
	@echo ""
	@echo "[Secure Docker — prefer disposable Linux VM, then these targets]"
	@echo "  make docker-preflight         verify Docker + Compose hardening contract"
	@echo "  make docker-build             build hardened simulator image"
	@echo "  make docker-test              run unit tests in locked-down container"
	@echo "  make docker-smoke             run E2E smoke in locked-down container"
	@echo "  make docker-no-network-smoke  run smoke with Docker networking disabled"
	@echo "  make docker-gpu-preflight     preflight + NVIDIA container runtime (inside VM)"
	@echo "  make docker-gpu-build         build GPU override image (merge compose files)"
	@echo "  make docker-gpu-smoke         device probe (warn) + CPU torch smoke"
	@echo "  make docker-gpu-verify        requires GPU VM: preflight + strict device probe + smoke"
	@echo "  make docker-gpu-pytorch       full --preset gpu (needs CUDA torch; use VM venv)"
	@echo "  make docker-gpu-test          unit tests in GPU image (CPU torch wheel)"
	@echo ""
	@echo "[Maintenance]"
	@echo "  make outputs-clean CONFIRM=yes   wipe outputs/{benchmarks,logs,...}"
	@echo "  make clean-venv                  remove .ruff-venv scratch dir"

setup:
	$(PY) scripts/setup_from_clone.py

setup-quick:
	$(PY) scripts/setup_from_clone.py --quick

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"
	@echo "Tip: add torch with: pip install -e \".[dev,torch]\""

install-torch:
	$(PY) -m pip install -e ".[torch]"

# --- Experiments ---

run:
	@if [ -x ./run ]; then ./run; else $(PY) run_research.py; fi

reproduce:
	$(PY) run_research.py --config config.e2e_smoke.json

smoke: reproduce

run-config:
	@if [ -z "$(RESEARCH_CONFIG)" ]; then \
		echo "Missing RESEARCH_CONFIG. Example:"; \
		echo "  make run-config RESEARCH_CONFIG=config.e2e_smoke.json"; \
		exit 1; \
	fi
	$(PY) run_research.py --config "$(RESEARCH_CONFIG)"

moe:
	$(PY) run_moe_research.py

multiseed:
	$(PY) run_multiseed.py --preset "$(MULTISEED_PRESET)" --seeds "$(MULTISEED_SEEDS)"

multiseed-smoke:
	$(PY) run_multiseed.py --preset dense --seeds 7,11 --episodes 24

SWEEP_CONFIG ?= config.sweep.example.json

sweep:
	$(PY) run_sweep.py --sweep-config "$(SWEEP_CONFIG)"

sweep-smoke:
	$(PY) run_sweep.py --config config.e2e_smoke.json --run-name test_sweep \
		--vary learning_rate=0.02,0.035 --episodes 24 --quiet

# --- CUDA ---

pytorch:
	$(PY) run_pytorch.py --preset "$(PYTORCH_PRESET)"

3090:
	$(PY) run_pytorch.py --preset 3090

4090:
	$(PY) run_pytorch.py --preset 4090

4090-universal:
	$(PY) run_pytorch.py --preset 4090-universal

# --- Other ---

online:
	$(PY) run_online_learning.py

calibrate:
	$(PY) run_calibrate_llama_cpp.py

route-help:
	$(PY) run_route_learning.py --help

# --- Quality ---

test:
	$(PY) -m unittest discover -s tests -t . -v

test-quiet:
	$(PY) -m unittest discover -s tests -t . -q

secret-scan:
	$(PY) scripts/secret_scan.py

lint:
	$(PY) -m ruff check $(PKG) $(SCRIPTS_PY) $(RUN) $(CFG)

format:
	$(PY) -m ruff format $(PKG) $(SCRIPTS_PY) $(RUN) $(CFG)

check: lint
	$(PY) scripts/pre_commit_check.py

doctor:
	@cd "$(CURDIR)" && PYTHONPATH="$(CURDIR)/src:$(CURDIR):$$PYTHONPATH" $(PY) scripts/env_report.py

# --- Secure Docker (run inside a disposable Linux VM when artifacts are untrusted) ---

COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

docker-preflight:
	bash scripts/docker_secure_preflight.sh

docker-build: docker-preflight
	docker compose build

docker-test: docker-build
	docker compose run --rm adaptive-rl-quant python -m unittest discover -s tests -t . -q

docker-smoke: docker-build
	docker compose run --rm adaptive-rl-quant

docker-no-network-smoke: docker-build
	docker compose run --rm --network none adaptive-rl-quant

docker-gpu-preflight:
	bash scripts/docker_secure_preflight.sh --gpu

docker-gpu-build: docker-preflight
	$(COMPOSE_GPU) build

docker-gpu-smoke: docker-gpu-build
	$(COMPOSE_GPU) run --rm adaptive-rl-quant python scripts/docker_gpu_device_probe.py
	$(COMPOSE_GPU) run --rm adaptive-rl-quant

docker-gpu-verify: docker-gpu-preflight docker-gpu-build
	ADAPTIVE_RL_REQUIRE_CONTAINER_CUDA=1 $(COMPOSE_GPU) run --rm adaptive-rl-quant python scripts/docker_gpu_device_probe.py
	$(COMPOSE_GPU) run --rm adaptive-rl-quant

docker-gpu-pytorch: docker-gpu-preflight docker-gpu-build
	$(COMPOSE_GPU) run --rm adaptive-rl-quant adaptive-rl-quant-pytorch --preset gpu

docker-gpu-test: docker-gpu-build
	$(COMPOSE_GPU) run --rm adaptive-rl-quant python -m unittest discover -s tests -t . -q

# --- Maintenance ---

outputs-clean:
	@if [ "$(CONFIRM)" != "yes" ]; then \
		echo "This removes: outputs/benchmarks outputs/logs outputs/analysis outputs/checkpoints outputs/reports"; \
		echo "To proceed:   make outputs-clean CONFIRM=yes"; \
		exit 1; \
	fi
	rm -rf outputs/benchmarks outputs/logs outputs/analysis outputs/checkpoints outputs/reports
	@echo "Removed those directories under outputs/ (rest of outputs/ left untouched)."

clean-venv:
	rm -rf .ruff-venv
