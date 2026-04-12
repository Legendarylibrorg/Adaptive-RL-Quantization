# Usage Guide

## Platform and working directory

- **Simulator path:** supported on **Linux, macOS, and Windows**. Work in the **repo root** (directory with `config.py` and `pyproject.toml`).
- **Windows:** use `py -3.11` or `python` instead of `python3`, and venv `Scripts\activate`.
- **GPU workflows:** remain **Linux-first**.

Install and system packages: [INSTALL.md](INSTALL.md). Command reference: [RUNNING.md](RUNNING.md).

## Basics

1. `python3 -m pip install -e .`
2. `python3 -m unittest discover -s tests -q`
3. `python3 run_research.py` or `python3 run_research.py --config my.json`

Cross-platform shortcuts: `python3 scripts/setup_from_clone.py` and `python3 scripts/pre_commit_check.py` on Unix-like hosts; on Windows use `py -3.11` or `python`.

**Dependencies:** Core package has **no required PyPI deps** ([`pyproject.toml`](../pyproject.toml)). PyTorch is optional (`pip install -e ".[torch]"` or a manual CUDA wheel) for GPU entrypoints and `training_backend="pytorch"`.

## Configuration without editing Python

- Copy [`config.example.json`](../config.example.json) or add a `.toml` file.
- Run: `python3 run_research.py -c ./conf.toml`
- Details and `preset` keys: [CONFIG.md](CONFIG.md).

**API examples:**

```python
from adaptive_quant import load_config, quick_config, build_trainer
from adaptive_quant.research_pipeline import run_pipeline_entrypoint
from config import CONFIG

run_pipeline_entrypoint(CONFIG)

cfg = load_config("my.json")
trainer = build_trainer(cfg)

cfg2 = quick_config(run_name="ablation", training_episodes=500)
```

Or `ResearchPipeline(cfg).run()`, `FrameworkConfig.from_file`, `FrameworkConfig.from_mapping`. Heavy symbols on `import adaptive_quant` (e.g. `Trainer`) load lazily — see [`__init__.py`](../adaptive_quant/__init__.py).

## Outputs

Everything lands under `outputs/`: `logs/`, `benchmarks/` (summaries + optional `*_preflight.json` + `*_recommendation.json`), `analysis/<run_name>/`, `checkpoints/`, `reports/`. Names follow `run_name` and path fields in config.

The recommendation artifact records detected host hardware, the target hardware class used for scoring, adaptive-policy performance on that target, and the best fixed quant candidate discovered from deterministic RL rollouts.

## Re-run analysis (no training)

From repo root, pass **log or history path** then **output dir**:

```bash
python3 analysis/hardware_generalization.py path/to/*_multi_hw.jsonl out/
python3 analysis/input_adaptation.py path/to/*_dynamic.jsonl out/
python3 analysis/quant_function_behavior.py path/to/*_learned.jsonl out/
```

Logic is in [`analysis/analyzers.py`](../analysis/analyzers.py); other files under `analysis/` are thin CLIs.
