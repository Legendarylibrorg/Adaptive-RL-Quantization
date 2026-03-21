# Configuration Guide

Configuration lives in:

- [config.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config.py)
- [config_online.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_online.py)
- [config_gpu.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_gpu.py)
- [config_4090.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_4090.py)
- [adaptive_quant/configuration.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/adaptive_quant/configuration.py)

## Most important fields

General:

- `training_backend`: `"python"` or `"pytorch"`
- `backend`: `"simulator"` or `"llama_cpp"`
- `run_name`: controls output filenames

Adaptive behavior:

- `multi_hardware`
- `dynamic_quant`
- `learned_quant`
- `quant_mode`

Episode budget:

- `training_episodes`
- `evaluation_episodes`
- `benchmark_training_episodes`
- `benchmark_evaluation_episodes`

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

Online adaptation:

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

## Recommended presets

Local laptop or quick CI-style validation:

- `training_backend="python"`
- `backend="simulator"`
- small `training_episodes`

Auto-tuned GPU training:

- use [config_gpu.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_gpu.py)
- keep `torch_gpu_profile="auto"` unless you want to force a profile
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

RTX 4090 training:

- use [config_4090.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_4090.py)
- keep `training_backend="pytorch"`
- keep `cache_prompt_features=True`
- keep `torch_preflight=True`

- use [config_online.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_online.py) for continual adaptation experiments
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
