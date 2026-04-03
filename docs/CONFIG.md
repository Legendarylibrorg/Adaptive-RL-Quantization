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

3. Run from the **repo root** (Linux / macOS):

   ```bash
   python3 run_research.py --config ./my_run.json
   python3 run_research.py -c ./my_run.toml
   python3 run_pytorch.py --config ./cuda_run.toml   # replaces --preset entirely
   python3 run_moe_research.py --config moe.json
   python3 run_online_learning.py --config online.toml
   python3 run_calibrate_llama_cpp.py --config ./paths_only.json
   ```

4. **API:** `FrameworkConfig.from_file(path)`, `load_config(path)` (from `adaptive_quant`), or `FrameworkConfig.from_mapping(dict, strict=True)` to reject unknown keys.

Lists in JSON for tuple fields (`hardware_modes`, `discrete_bit_widths`, `scale_bounds`, …) are coerced automatically. Nested **`reward_weights`** merges onto defaults.

---

## Python module presets

Configuration also lives in:

- [`config.py`](../config.py)
- [`config_moe.py`](../config_moe.py)
- [`config_online.py`](../config_online.py)
- [`config_gpu.py`](../config_gpu.py)
- [`config_4090.py`](../config_4090.py)
- [`config_4090_universal.py`](../config_4090_universal.py)
- [`adaptive_quant/configuration.py`](../adaptive_quant/configuration.py)

Use [`config.py`](../config.py) as the canonical offline research baseline when you are not using `--config`. It is the simplest preset to reproduce and the best starting point for stable experiments.

---

## Research / reproducibility fields

- **`env_sampling_mode`**: `random` (default) | `sequential` | `forced` — controls how prompts and hardware are chosen on `reset()`; **`sequential`** uses `episode_index` for a fixed schedule (see [`runner_cli` / trainers passing `episode_index`](../adaptive_quant/trainer_utils.py)).
- **`env_forced_prompt_id`**, **`env_forced_hardware`**: used when `env_sampling_mode="forced"` if `reset()` does not pass explicit prompt/hardware.
- **`rl_train_policy_mode`**: `stochastic` (sample π during training) | `deterministic` (greedy / argmax during training rollouts).
- **`stability_probe_sampling`**: `random` | `deterministic` — probe order for the stability penalty term.
- **`torch_deterministic`**: enable CUDNN deterministic mode, cuBLAS workspace, global seeds, and stricter PyTorch algorithms (GPU; slower). **`torch.compile` is skipped** when this is true.
- **`torch_policy_algorithm`**: `ppo` | `vpg` | `awr` (PyTorch trainer only).
- **`torch_awr_beta`**: temperature for `awr` weights.
- **`reward_perplexity_reference`**, **`reward_weights.zeta_perplexity_over_ref`**: hinge penalty when perplexity exceeds a baseline (throughput-focused runs with a quality guard).

Factory: **`FrameworkConfig.reproducible_research(seed=..., run_name=..., **kwargs)`** aligns seeds and turns on the full reproducibility-oriented stack.

## Most important fields

General:

- `training_backend`: `"python"` (stdlib trainer; PyTorch not required) or `"pytorch"` (install PyTorch on the host). PyTorch is only loaded when you run PyTorch-backed code paths (this setting, GPU entrypoints, or imports from `adaptive_quant.torch_*`).
- `backend`: `"simulator"` or `"llama_cpp"`
- `training_host_label`: optional label for the machine used to train the policy
- `run_name`: controls output filenames
- `run_name` must be a filename-safe slug (letters/digits plus `._-`, no spaces, no path separators).
- `resume_from_checkpoint`: resume a PyTorch run from a saved checkpoint
- **Checkpoint format (PyTorch):** new saves write `*.pt` (weights + optimizer tensors only) plus a sidecar `*.checkpoint.json` (episode counters and history). Loads use `weights_only=True` when your PyTorch version supports it.
- `allow_legacy_checkpoint_load`: default **false**. Set **true** only to load older single-file pickle checkpoints that lack the sidecar (trusted files only); then turn it off again.

Adaptive behavior:

- `multi_hardware`
- `dynamic_quant`
- `learned_quant`
- `moe_enabled`
- `quant_mode`
- `prompt_split_enabled`: if true, sample different prompt subsets for training vs evaluation
- `prompt_split_seed`: RNG seed for the prompt split
- `prompt_train_fraction`: fraction of prompts assigned to the training split

Episode budget:

- `training_episodes`: number of episodes for fixed-horizon training (default: 3,000)
- `evaluation_episodes`: number of episodes for evaluation (default: 400)
- `benchmark_training_episodes`
- `benchmark_evaluation_episodes`

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
- `backend="simulator"`
- small `training_episodes`
- start from [`config.py`](../config.py)

Auto-tuned GPU training:

- use [`config_gpu.py`](../config_gpu.py)
- keep `torch_gpu_profile="auto"` unless you want to force a profile
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

RTX 4090 training:

- use [`config_4090.py`](../config_4090.py)
- keep `training_backend="pytorch"`
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

4090-host universal policy training:

- use [`config_4090_universal.py`](../config_4090_universal.py)
- keep `training_host_label="rtx4090"`
- keep `multi_hardware=True`
- keep `hardware_modes=("gpu", "cpu", "low_resource")`

Canonical MoE research:

- use [`config_moe.py`](../config_moe.py)
- keep `moe_enabled=True`
- keep `moe_variant_names=("safe", "balanced", "aggressive")`
- keep `moe_max_aggressive_experts` and `moe_max_swap_cost_ms` enabled for safety

Experimental continual adaptation:

- use [`config_online.py`](../config_online.py) for continual adaptation experiments
- keep `online_learning=True`
- tune `online_exploration_rate` and `online_reward_guard` together
- increase `online_drift_reward_delta` if the loop is too rollback-heavy

Real llama.cpp experiments:

- set `backend="llama_cpp"`
- set `llama_cpp_binary`
- set `llama_cpp_model`

## Common edits

Reduce runtime:

- lower `training_episodes`
- lower `benchmark_training_episodes`
- lower `evaluation_episodes`

Reduce log volume:

- increase `log_every_n_episodes`

Reduce GPU memory pressure:

- lower `torch_batch_episodes`
- lower `torch_minibatch_size`
- lower `torch_hidden_dim`
- disable `torch_compile` if compile overhead is not worth it for short runs

Increase benchmark fidelity:

- raise `benchmark_training_episodes`
- raise `benchmark_evaluation_episodes`
- switch `backend` to `llama_cpp`
