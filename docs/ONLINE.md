# Online Adaptation Guide

`run_online_learning.py` turns the framework into a continual adaptation loop.

It does four things:

1. bootstraps a policy with offline simulator training
2. serves a stream of online requests with hardware- and input-aware decisions
3. stores exploratory experiences in a replay buffer
4. applies replay updates while protecting the serving path with canaries and rollback

## Main command

```bash
python3 run_online_learning.py
```

## Main files

- [run_online_learning.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/run_online_learning.py)
- [config_online.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/config_online.py)
- [adaptive_quant/online_learning.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/adaptive_quant/online_learning.py)
- [analysis/online_learning.py](/Users/devcomputer/Downloads/Adaptive-RL-Quantization/analysis/online_learning.py)

## How the loop works

Each online request goes through this flow:

1. build a state from hardware, prompt features, sensitivity, and previous action
2. choose a deterministic baseline action
3. optionally sample an exploratory action
4. on canary traffic, compare candidate and baseline under guardrails
5. serve the accepted decision
6. add exploratory outcomes to replay
7. update the policy after enough replay accumulates
8. rollback to the best recent snapshot if served reward drifts down too far

## Key safety controls

- `online_reward_guard`: how much worse a candidate can be than the baseline reward
- `online_max_latency_ratio`: maximum allowed latency inflation versus baseline
- `online_max_memory_ratio`: maximum allowed memory inflation versus baseline
- `online_max_perplexity_delta`: maximum allowed perplexity increase versus baseline
- `online_drift_window`: size of the reward window used for drift detection
- `online_drift_reward_delta`: how far reward can fall before rollback
- `online_safe_mode_cooldown`: how long to pause exploration after rollback

## Outputs

Logs:

- `outputs/logs/*_online_telemetry.jsonl`
- `outputs/logs/*_online_replay.jsonl`

Summaries:

- `outputs/benchmarks/*_online_summary.json`
- `outputs/benchmarks/*_summary.json`

Analysis:

- `outputs/analysis/<run_name>/online`

## Tuning advice

If the loop never updates:

- raise `online_exploration_rate`
- lower `online_min_replay_size`
- lower `online_update_interval`

If the loop keeps rolling back:

- increase `online_drift_reward_delta`
- reduce `online_exploration_rate`
- loosen or retune the canary guardrails only if that matches your risk tolerance
