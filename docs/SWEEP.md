# Hyperparameter Sweep Guide

`adaptive-rl-quant-sweep` runs a **cartesian grid** or an **explicit trial list** over config overrides, executes the full research pipeline once per trial, and ranks results by a numeric objective extracted from each trial's pipeline summary.

The source-checkout equivalent is `python3 run_sweep.py`.

## When to use a sweep vs multiseed

| Goal | Tool |
| --- | --- |
| Same config, different random seeds (variance / confidence) | `adaptive-rl-quant-multiseed` |
| Different hyperparameters (learning rate, reward weights, torch batch sizes, …) | `adaptive-rl-quant-sweep` |

Multiseed repeats one setting; sweep compares **different settings** under the same base config.

## Main commands

```bash
# Sweep file (recommended for reproducible grids)
adaptive-rl-quant-sweep --sweep-config config.sweep.example.json

# CLI grid (no sweep file)
adaptive-rl-quant-sweep --config config.e2e_smoke.json \
  --vary learning_rate=0.02,0.035 \
  --vary reward_weights.beta_throughput=0.04,0.08

# Quick smoke (two learning rates, low episodes)
make sweep-smoke

# Full example grid from repo config
make sweep
# or: make sweep SWEEP_CONFIG=my_sweep.json
```

Source-checkout equivalents:

```bash
python3 run_sweep.py --sweep-config config.sweep.example.json
python3 run_sweep.py --preset dense --vary learning_rate=0.02,0.035 --episodes 48
```

## What each trial runs

Every trial calls the same **research pipeline** as `adaptive-rl-quant`:

train → evaluate → benchmarks → analysis → optional Markdown report

Trial-specific overrides are applied on top of the base config (from `--config`, `--preset`, or the sweep file's `base_config`). Each trial gets a unique `run_name`:

```text
<base_run_name>_trial001_<suffix>
```

The suffix is derived from overridden keys (for example `learning_rate_0.02`).

## Sweep config file

Copy [`config.sweep.example.json`](../config.sweep.example.json):

```json
{
  "base_config": "config.e2e_smoke.json",
  "run_name": "lr_sweep_smoke",
  "seed": 13,
  "objective": "evaluation.mean_reward",
  "direction": "maximize",
  "grid": {
    "learning_rate": [0.02, 0.035],
    "reward_weights.beta_throughput": [0.04, 0.08]
  }
}
```

This example runs **2 × 2 = 4 trials** (cartesian product of the two grid axes).

### Sweep-specific keys

These keys are **not** part of `FrameworkConfig`. They are stripped before building the base config:

| Key | Role |
| --- | --- |
| `base_config` | Optional path to a base JSON/TOML config loaded before grid trials |
| `grid` | Cartesian product of parameter lists (same dotted keys as `--set` / flat config fields) |
| `trials` | Alternative to `grid`: explicit list of override mappings (**mutually exclusive** with `grid`) |
| `objective` | Dotted metric path used to rank trials (default `evaluation.mean_reward`) |
| `direction` | `maximize` or `minimize` |
| `seed` | Fixed seed applied to every trial unless overridden per trial |

Any **remaining** top-level keys after extracting sweep metadata are merged as base-config overrides (same rules as a normal config file, including optional `preset`).

JSON and TOML sweep files are supported (`.json`, `.toml`).

### Explicit trial list (no cartesian product)

Use `trials` when you want hand-picked combinations instead of a full grid:

```json
{
  "base_config": "config.e2e_smoke.json",
  "run_name": "manual_trials",
  "objective": "evaluation.mean_reward",
  "direction": "maximize",
  "trials": [
    { "learning_rate": 0.02, "reward_weights.beta_throughput": 0.04 },
    { "learning_rate": 0.035, "torch_batch_episodes": 16 }
  ]
}
```

## CLI flags

| Flag | Role |
| --- | --- |
| `--sweep-config PATH` | Load grid/trials/objective from a sweep JSON/TOML file |
| `--config PATH` | Base config when `--sweep-config` is omitted, or fallback when the sweep file has no `base_config` |
| `--preset dense\|moe` | Base preset when no `--config` or sweep `base_config` (default `dense`) |
| `--vary KEY=val1,val2` | Grid axis; repeat for cartesian products |
| `--objective PATH` | Dotted metric path (default `evaluation.mean_reward`) |
| `--direction maximize\|minimize` | Ranking direction (default `maximize`) |
| `--run-name NAME` | Base run name for artifacts (defaults to base config `run_name`) |
| `--seed N` | Fixed seed for every trial (overridden by sweep file `seed` when set) |
| `--episodes N` | Override `training_episodes` for every trial (useful for smoke tests) |
| `--quiet` | Suppress end-of-run CLI banners |

Override keys use the same normalization as startup `--set` flags (dotted paths, nested dict merge). **Privileged keys** (backend, llama.cpp paths, router allowlists, etc.) follow the same policy as other research entrypoints — prefer putting them in a reviewed config file rather than sweep grids.

## Objective metrics

The default objective is `evaluation.mean_reward` from each trial's pipeline summary JSON.

Other common objectives (use dotted paths that exist in the summary):

- `evaluation.mean_latency_ms` — minimize with `--direction minimize`
- `evaluation.mean_throughput_tps`
- `benchmarks.fixed_q4.mean_reward`
- Any numeric field reachable via dotted path in the flattened summary

Ranking uses [`extract_metric`](../src/adaptive_quant/experiment_aggregate.py): it flattens numeric fields from the trial summary and matches the objective path (exact key, suffix, or final path segment).

Inspect a completed trial summary to discover valid objective paths:

```bash
python3 -c "import json; print(json.dumps(json.load(open('outputs/benchmarks/my_run_trial001_lr_0.02_summary.json')), indent=2)[:2000])"
```

## Outputs

For base run name `my_sweep`:

| Artifact | Path |
| --- | --- |
| Sweep aggregate JSON | `outputs/benchmarks/my_sweep_sweep_summary.json` |
| Leaderboard report | `outputs/reports/my_sweep_sweep_report.md` |
| Paper bundle | `outputs/paper_bundles/my_sweep_sweep/` |
| Per-trial summary | `outputs/benchmarks/my_sweep_trialNNN_<suffix>_summary.json` |
| Per-trial report | `outputs/reports/my_sweep_trialNNN_<suffix>_report.md` |

The aggregate JSON includes:

- `leaderboard` — trials ranked by objective
- `trials` — per-trial overrides, objective values, summary paths
- `sweep` — grid/trials metadata and config paths
- `config` — base config snapshot
- `artifacts` — report, paper bundle, and per-trial summary paths

Open per-trial `outputs/reports/*_report.md` files for full benchmark and analysis detail.

## Makefile shortcuts

From repo root (uses `.venv` when present):

```bash
make sweep              # config.sweep.example.json (override with SWEEP_CONFIG=...)
make sweep-smoke        # two learning rates, 24 episodes, --quiet
```

## Implementation map

| File | Role |
| --- | --- |
| [`run_sweep.py`](../run_sweep.py) | Repo-root shim → `adaptive_quant.cli.sweep` |
| [`src/adaptive_quant/cli/sweep.py`](../src/adaptive_quant/cli/sweep.py) | CLI, trial loop, leaderboard report |
| [`src/adaptive_quant/sweep.py`](../src/adaptive_quant/sweep.py) | Grid expansion, trial naming, ranking, sweep file loader |
| [`src/adaptive_quant/experiment_aggregate.py`](../src/adaptive_quant/experiment_aggregate.py) | `extract_metric` for objective extraction |

## Related docs

- [CONFIG.md](CONFIG.md) — sweep file key reference (short form)
- [RUNNING.md](RUNNING.md) — entrypoint table and Makefile notes
- [USAGE.md](USAGE.md) — artifact layout shared with multiseed runs
- [PAPER.md](PAPER.md) — research workflow context
