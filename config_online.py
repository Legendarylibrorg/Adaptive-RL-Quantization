from config import CONFIG

CONFIG_ONLINE = CONFIG.clone(
    training_episodes=160,
    evaluation_episodes=48,
    run_name="adaptive_online_policy",
    online_learning=True,
    online_requests=192,
    online_exploration_rate=0.30,
    online_canary_ratio=0.75,
    online_replay_capacity=1024,
    online_min_replay_size=16,
    online_update_interval=8,
    online_batch_size=32,
    online_reward_guard=1.10,
    online_max_latency_ratio=1.20,
    online_max_memory_ratio=1.12,
    online_max_perplexity_delta=1.10,
    online_drift_window=48,
    online_drift_reward_delta=4.00,
    online_safe_mode_cooldown=8,
)
