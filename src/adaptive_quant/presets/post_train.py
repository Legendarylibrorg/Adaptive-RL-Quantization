from adaptive_quant.presets.gpu import CONFIG_GPU

# Long-horizon RL post-training for open-weight models:
# - continuous training with periodic eval/checkpoints
# - sequential prompt curriculum from prompts/post_train_library.json
# - learned route selection across model/quant candidates (hash router by default)
# Point backend=llama_cpp and router_routes at your GGUF paths for real measurements.
CONFIG_POST_TRAIN = CONFIG_GPU.clone(
    run_name="oss_llm_post_train_rl",
    training_host_label="post_train",
    continuous_training=True,
    max_training_episodes=50_000,
    eval_interval=2_000,
    checkpoint_interval=10_000,
    training_episodes=8_192,
    evaluation_episodes=512,
    benchmark_training_episodes=2_048,
    benchmark_evaluation_episodes=256,
    prompt_library_path="prompts/post_train_library.json",
    env_sampling_mode="sequential",
    prompt_split_enabled=True,
    prompt_train_fraction=0.85,
    router_enabled=True,
    router_feature_backend="hash",
    router_routes=(
        "hf:openai-community/gpt2@q4",
        "hf:openai-community/gpt2@q8",
        "hf:openai-community/gpt2@q2",
    ),
    router_exploration=0.12,
    replay_buffer_capacity=100_000,
    replay_buffer_on_gpu=True,
    log_every_n_episodes=32,
    jsonl_buffered=True,
    jsonl_flush_every=64,
    cache_prompt_features=True,
    torch_compile=True,
    torch_batch_episodes=512,
    torch_minibatch_size=256,
    torch_update_epochs=4,
    torch_preflight=True,
)
