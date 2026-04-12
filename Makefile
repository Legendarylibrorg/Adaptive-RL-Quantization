# Research-oriented Makefile (Linux/macOS; on Windows use the Python scripts under scripts/). From repo root: `make help`
# Override interpreter:  PY=python3.12 make test

.PHONY: help
.PHONY: install install-dev install-torch
.PHONY: run reproduce smoke run-config moe multiseed multiseed-smoke
.PHONY: pytorch 4090 4090-universal
.PHONY: online calibrate
.PHONY: test test-quiet lint format check secret-scan doctor
.PHONY: outputs-clean clean-venv

PY ?= python3
PKG := adaptive_quant analysis tests
SCRIPTS_PY := $(wildcard scripts/*.py)
RUN := $(wildcard run_*.py)
CFG := $(wildcard config*.py)

# Multiseed: preset dense|moe ; seeds "a,b,c" or "0-4"
MULTISEED_PRESET ?= dense
MULTISEED_SEEDS ?= 13,17,23

# PyTorch entrypoint preset: gpu | 4090 | 4090-universal
PYTORCH_PRESET ?= gpu

help:
	@echo "Adaptive RL Quantization — research Makefile"
	@echo "Override PY=... for a non-default interpreter."
	@echo ""
	@echo "[Bootstrap]"
	@echo "  make install          pip install -e . (simulator path)"
	@echo "  make install-dev      + Ruff (lint/format)"
	@echo "  make install-torch    + PyTorch extra (CUDA wheel still your job)"
	@echo ""
	@echo "[Experiments — simulator / stdlib trainer]"
	@echo "  make run              full run: run_research.py (config.py)"
	@echo "  make reproduce        CI-equivalent smoke: config.e2e_smoke.json (alias: make smoke)"
	@echo "  make run-config       RESEARCH_CONFIG=path.json|toml (required)"
	@echo "  make moe              run_moe_research.py (MoE benchmarks)"
	@echo "  make multiseed        MULTISEED_PRESET=dense|moe MULTISEED_SEEDS=..."
	@echo "  make multiseed-smoke  two seeds, low episodes (quick sanity)"
	@echo ""
	@echo "[CUDA — install torch first: make install-torch]"
	@echo "  make pytorch          PYTORCH_PRESET=gpu (default) | 4090 | 4090-universal"
	@echo "  make 4090             same as pytorch with preset 4090"
	@echo "  make 4090-universal   multi-hardware 4090-host preset"
	@echo ""
	@echo "[Other runners]"
	@echo "  make online           experimental online loop"
	@echo "  make calibrate        llama.cpp calibration (needs binary+model in config)"
	@echo ""
	@echo "[Quality]"
	@echo "  make test | test-quiet | secret-scan | lint | format | check"
	@echo "  make doctor           env summary: Python path, torch/ruff, git, outputs/* counts"
	@echo ""
	@echo "[Maintenance]"
	@echo "  make outputs-clean CONFIRM=yes   wipe outputs/{benchmarks,logs,...}"
	@echo "  make clean-venv                  remove .ruff-venv scratch dir"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"
	@echo "Tip: add torch with: pip install -e \".[dev,torch]\""

install-torch:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[torch]"

# --- Experiments ---

run:
	$(PY) run_research.py

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

# --- CUDA ---

pytorch:
	$(PY) run_pytorch.py --preset "$(PYTORCH_PRESET)"

4090:
	$(PY) run_pytorch.py --preset 4090

4090-universal:
	$(PY) run_pytorch.py --preset 4090-universal

# --- Other ---

online:
	$(PY) run_online_learning.py

calibrate:
	$(PY) run_calibrate_llama_cpp.py

# --- Quality ---

test:
	$(PY) -m unittest discover -s tests -v

test-quiet:
	$(PY) -m unittest discover -s tests -q

secret-scan:
	$(PY) scripts/secret_scan.py

lint:
	$(PY) -m ruff check $(PKG) $(SCRIPTS_PY) $(RUN) $(CFG)

format:
	$(PY) -m ruff format $(PKG) $(SCRIPTS_PY) $(RUN) $(CFG)

check: lint
	$(PY) scripts/pre_commit_check.py

doctor:
	@cd "$(CURDIR)" && PYTHONPATH="$(CURDIR):$$PYTHONPATH" $(PY) scripts/env_report.py

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
