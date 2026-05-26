# Configuration Guide

## JSON and TOML (file-based configs)

For a **single file** instead of editing Python presets:

1. Use **[`config.e2e_smoke.json`](../config.e2e_smoke.json)** for a **short reproducible E2E RL** run (train → eval → benchmarks → analysis); copy [`config.example.json`](../config.example.json) for a `minimal`-preset example, or [`config.example.pytorch.toml`](../config.example.pytorch.toml) for GPU file-based runs; you can also author your own `.toml` / `.json`.
2. Optional top-level string **`preset`** picks a base profile (merged first, then other keys override):

   | `preset` value | Effect |
   | --- | --- |
   | `default` (or omit `preset` when loading via API without a file base) | Plain `FrameworkConfig()` |
   | `minimal` | Short run: low `training_episodes`, `stability_probe_count=1`, no replay, preflight off |
   | `pytorch` | `training_backend="pytorch"`, `torch_gpu_profile="auto"` |
   | `reproducible` | Same as `FrameworkConfig.reproducible_research()` (sequential env, deterministic train policy & probes, `torch_deterministic`, no compile) |

3. Run from the **repo root** with the installed commands:

   ```bash
   adaptive-rl-quant --config ./my_run.json
   adaptive-rl-quant -c ./my_run.toml
   adaptive-rl-quant-pytorch --config ./cuda_run.toml   # replaces --preset entirely
   adaptive-rl-quant-moe --config moe.json
   adaptive-rl-quant-online --config online.toml
   adaptive-rl-quant-calibrate --config ./paths_only.json
   ```

   Source-checkout equivalents remain `python3 run_research.py`, `python3 run_pytorch.py`, `python3 run_moe_research.py`, `python3 run_online_learning.py`, and `python3 run_calibrate_llama_cpp.py`.

4. **API:** `FrameworkConfig.from_file(path)`, `load_config(path)` (from `adaptive_quant`), or `FrameworkConfig.from_mapping(dict, strict=True)` to reject unknown keys. TOML/JSON files are the **preferred** way to share reproducible runs (version control, CI, no import side effects). Python module presets under `config*.py` remain for local iteration and GPU-specific paths. TOML parsing uses stdlib **`tomllib`** (requires **Python 3.11+** per `pyproject.toml`).

   Untrusted config integers (episode counts, MoE topology, llama.cpp limits, etc.) are capped at load time; see `MAX_*` constants in [`validation.py`](../src/adaptive_quant/configuration/validation.py).

Lists in JSON for tuple fields (`hardware_modes`, `discrete_bit_widths`, `scale_bounds`, …) are coerced automatically. Nested **`reward_weights`** merges onto defaults.

---

## Python module presets

**When to use which:** prefer **JSON/TOML + `--config`** for anything you will rerun, share, or run in CI; use **Python presets** when you need programmatic overrides, machine-specific paths, or quick one-off edits without a separate file.

Configuration also lives in:

- [`config.py`](../src/config.py) — exports `CONFIG`, `CONFIG_MOE`, `CONFIG_ONLINE`, `CONFIG_GPU`, `CONFIG_3090`, `CONFIG_4090`, `CONFIG_4090_UNIVERSAL`
- [`adaptive_quant/presets/`](../src/adaptive_quant/presets/) — preset definitions
- [`adaptive_quant/configuration/`](../src/adaptive_quant/configuration/) (`framework.py`: `FrameworkConfig`)

Use [`config.py`](../src/config.py) as the canonical offline research baseline when you are not using `--config`. It is the simplest preset to reproduce and the best starting point for stable experiments.

---

## Research / reproducibility fields

- **`env_sampling_mode`**: `random` (default) | `sequential` | `forced` — controls how prompts and hardware are chosen on `reset()`; **`sequential`** uses `episode_index` for a fixed schedule (see [trainers passing `episode_index`](../src/adaptive_quant/trainer_utils.py)).
- **`env_forced_prompt_id`**, **`env_forced_hardware`**: used when `env_sampling_mode="forced"` if `reset()` does not pass explicit prompt/hardware.
- **`rl_train_policy_mode`**: `stochastic` (sample π during training) | `deterministic` (greedy / argmax during training rollouts).
- **`stability_probe_sampling`**: `random` | `deterministic` — probe order for the stability penalty term.
- **`torch_deterministic`**: enable CUDNN deterministic mode, cuBLAS workspace, global seeds, and stricter PyTorch algorithms (GPU; slower). **`torch.compile` is skipped** when this is true.
- **`jsonl_integrity_chain`**: each JSONL log line includes `_integrity_prev` / `_integrity_hash` (hash chain over canonical step payloads).
- **`replay_manifest_enabled`**: after training, write `outputs/logs/<run_name>_replay_manifest.json` with per-step `step_sha256` and rolling `chain_sha256`, plus `config_sha256`.
- **`replay_verify_after_run`**: when a manifest is written, re-hash the JSONL and re-execute logged decisions in the simulator to confirm outcomes match.

Factory: **`FrameworkConfig.reproducible_research(...)`** turns on sequential sampling, deterministic train/probes, static hardware profiles (`detect_host_hardware=False`), JSONL integrity chain, and replay manifest + verify.

**Replay CLI:** `adaptive-rl-quant-replay --config <file> --manifest <path> --jsonl <path>` (or `run_replay.py`) verifies JSONL against a manifest and replays logged decisions. Use `--build-manifest` to regenerate a manifest from JSONL; `--verify-jsonl-only` to skip simulator replay.
- **`torch_policy_algorithm`**: `ppo` | `vpg` | `awr` (PyTorch trainer only).
- **`torch_awr_beta`**: temperature for `awr` weights.
- **`reward_weights.eta_token_latency`**: optional extra penalty on `latency_ms / prompt_length` (default `0.0`); increase to reward token-efficient routes; perplexity and stability terms still bound quality.
- **`reward_perplexity_reference`**, **`reward_weights.zeta_perplexity_over_ref`**: hinge penalty when perplexity exceeds a baseline (throughput-focused runs with a quality guard).


## Most important fields

General:

- `training_backend`: `"python"` (stdlib trainer; PyTorch not required) or `"pytorch"` (install PyTorch on the host). PyTorch is only loaded when you run PyTorch-backed code paths (this setting, GPU entrypoints, or imports from `adaptive_quant.torch_*`).
- `backend`: `"simulator"` or `"llama_cpp"` (or a custom name registered with `register_backend` in `adaptive_quant.backends.registry`)
- `training_host_label`: optional label for the machine used to train the policy
- `run_name`: controls output filenames
- `run_name` must be a filename-safe slug (letters/digits plus `._-`, no spaces, no path separators).
- `resume_from_checkpoint`: resume a saved run from a checkpoint
- **Checkpoint format (Python trainer):** saves write `*.json` with the serialized policy state, previous action, and training history.
- **Checkpoint format (PyTorch):** new saves write `*.pt` (weights + optimizer tensors only) plus a sidecar `*.checkpoint.json` (episode counters and history). Loads use `weights_only=True` when your PyTorch version supports it. Single-file pickle `.pt` checkpoints without a sidecar are **refused** (no opt-in).

Adaptive behavior:

- `detect_host_hardware`: when true (default), probe the local host and tune the simulated hardware profiles from detected CPU/RAM/GPU characteristics; disable for strictly static cross-host baselines
- `multi_hardware`
- `dynamic_quant`
- `learned_quant`
- `moe_enabled`
- `router_feature_backend`: `"hash"` (stdlib default) or `"hf"` (optional Transformers/PyTorch embeddings).
- `router_hf_embedding_model`: model id for the HF embedding backend (`org/name` format).
- `router_hf_embedding_revision`: **required** when `router_feature_backend="hf"` — pin a commit hash or tag (never load floating `main` in production).
- `router_hf_local_files_only`: require a pre-warmed local HF cache instead of network access (recommended after vetting a snapshot).
- `router_hf_allowed_models`: **required** non-empty allowlist when using the HF backend; must include `router_hf_embedding_model`.
- `route_hf_allowed_repos`: optional allowlist of Hub `org/name` repos for route-catalog training/downloads (also settable via `ADAPTIVE_RL_HF_ALLOWED_REPOS`, comma-separated).
- `quant_mode`
- `prompt_split_enabled`: if true, sample different prompt subsets for training vs evaluation
- `prompt_split_seed`: RNG seed for the prompt split
- `prompt_train_fraction`: fraction of prompts assigned to the training split

Episode budget:

- `training_episodes`: number of episodes for fixed-horizon training (default: 3,000)
- `evaluation_episodes`: number of episodes for evaluation (default: 400)
- `benchmark_training_episodes`
- `benchmark_evaluation_episodes`
- `recommendation_eval_episodes`: bounded episode budget for the post-train RL quant recommendation pass
- `recommendation_candidate_limit`: max number of RL-generated fixed quant candidates to re-score on the detected target hardware

Continuous learning:

- `continuous_training`: if true, trains up to `max_training_episodes` with periodic eval/checkpoint (default: false)
- `max_training_episodes`: upper bound for continuous training (default: 50,000)
- `eval_interval`: evaluate every N episodes during continuous training (default: 1,000)
- `checkpoint_interval`: save checkpoint every N episodes during continuous training (default: 5,000)

GPU replay buffer (VRAM):

- `replay_buffer_capacity`: number of experiences stored in the replay buffer (default: 20,000; PyTorch path only)
- `replay_buffer_on_gpu`: if true, replay buffer tensors live on CUDA VRAM (default: true)

Safety and reward:

- `stability_probe_count`
- `instability_threshold`
- `safe_default_bits`
- `reward_weights`

llama.cpp integration:

- `llama_cpp_binary`
- `llama_cpp_model`
- `llama_cpp_threads`
- `llama_cpp_context`
- `llama_cpp_timeout_s`: subprocess timeout when invoking the llama.cpp binary (prevents hangs)
- `llama_cpp_max_prompt_chars`: clamp prompt length passed to llama.cpp (reduces argv/resource risk)

Simulator calibration:

- `sim_calibration`: optional per-hardware multipliers applied to simulator metrics
  - keys: `gpu`, `cpu`, `low_resource`
  - fields: `latency_multiplier`, `throughput_multiplier`, `memory_multiplier`

PyTorch and 4090:

- `torch_device`
- `torch_gpu_profile`
- `torch_dtype`
- `torch_compile`
- `torch_amp`
- `torch_tf32`
- `torch_hidden_dim`
- `torch_mlp_depth`
- `torch_learning_rate`
- `torch_batch_episodes`
- `torch_minibatch_size`
- `torch_update_epochs`
- `torch_fused_optimizer`
- `torch_preflight`

Experimental online adaptation:

- `online_learning`
- `online_requests`
- `online_exploration_rate`
- `online_canary_ratio`
- `online_replay_capacity`
- `online_min_replay_size`
- `online_update_interval`
- `online_batch_size`
- `online_reward_guard`
- `online_max_latency_ratio`
- `online_max_memory_ratio`
- `online_max_perplexity_delta`
- `online_drift_window`
- `online_drift_reward_delta`
- `online_safe_mode_cooldown`

Efficiency-related:

- `cache_prompt_features`
- `log_every_n_episodes`
- `write_training_history`
- `write_research_report`

MoE-specific:

- `moe_num_experts`
- `moe_top_k`
- `moe_variant_names`
- `moe_fixed_variant`
- `moe_gpu_resident_experts`
- `moe_swap_penalty`
- `moe_cache_miss_penalty`
- `moe_variant_churn_penalty`
- `moe_max_aggressive_experts`
- `moe_max_swap_cost_ms`

## Recommended presets

Local laptop or quick CI-style validation:

- `training_backend="python"`
- `backend="simulator"` (built-in measurement backends: `simulator` and `llama_cpp`; extensions via `register_backend`)
- small `training_episodes`
- start from [`config.py`](../src/config.py)

Auto-tuned GPU training:

- use `from config import CONFIG_GPU` (or clone [`config.py`](../src/config.py))
- keep `torch_gpu_profile="auto"` unless you want to force a profile
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

RTX 3090 training:

- use `from config import CONFIG_3090` (or `adaptive-rl-quant-pytorch --preset 3090`)
- same hygiene as other CUDA presets: `torch_preflight=True`, `cache_prompt_features=True`

RTX 4090 training:

- use `from config import CONFIG_4090`
- keep `training_backend="pytorch"`
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

4090-host universal policy training:

- use `from config import CONFIG_4090_UNIVERSAL` (or `adaptive-rl-quant-pytorch --preset 4090-universal` from the repo root)
- keep `training_host_label="rtx4090"`
- keep `multi_hardware=True`
- keep `hardware_modes=("gpu", "cpu", "low_resource")`

Canonical MoE research:

- use `from config import CONFIG_MOE`
- keep `moe_enabled=True`
- keep `moe_variant_names=("safe", "balanced", "aggressive")`
- keep `moe_max_aggressive_experts` and `moe_max_swap_cost_ms` enabled for safety

Experimental continual adaptation:

- use `from config import CONFIG_ONLINE` for continual adaptation experiments
- keep `online_learning=True`
- tune `online_exploration_rate` and `online_reward_guard` together
- increase `online_drift_reward_delta` if the loop is too rollback-heavy

Real llama.cpp experiments:

- set `backend="llama_cpp"`
- set `llama_cpp_binary`
- set `llama_cpp_model`

Online routing is controlled separately by `router_enabled` and `router_routes`. It is not a separate `backend` value; routes are evaluated through the configured measurement backend.

### Custom measurement backends (advanced)

Plugins can register an extra backend name before building the environment:

```python
from adaptive_quant.backends.registry import register_backend, build_backend
from adaptive_quant.configuration import FrameworkConfig

register_backend("my_backend", lambda config: MyBackend(config))
config = FrameworkConfig(backend="my_backend", run_name="plugin_smoke")
backend = build_backend(config)
```

Built-in names remain `simulator` and `llama_cpp`. Unknown names produce an error that lists any registered custom backends.

If you set `router_feature_backend="hf"`, install the optional dependencies first:

```bash
python3 -m pip install -e ".[torch,router]"
```

## Common edits

Reduce runtime:

- lower `training_episodes`
- lower `benchmark_training_episodes`
- lower `evaluation_episodes`

Reduce log volume:

- increase `log_every_n_episodes`

Speed up very long runs (at the cost of keeping file handles open):

- set `jsonl_buffered=true`
- optionally increase `jsonl_flush_every` (e.g. 32 or 128)

Reduce GPU memory pressure:

- lower `torch_batch_episodes`
- lower `torch_minibatch_size`
- lower `torch_hidden_dim`
- disable `torch_compile` if compile overhead is not worth it for short runs

Increase benchmark fidelity:

- raise `benchmark_training_episodes`
- raise `benchmark_evaluation_episodes`
- switch `backend` to `llama_cpp`
