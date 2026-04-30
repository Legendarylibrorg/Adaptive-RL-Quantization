from config_gpu import make_rtx_torch_preset

CONFIG_4090 = make_rtx_torch_preset(
    training_host_label="rtx4090",
    benchmark_training_episodes=768,
    benchmark_evaluation_episodes=96,
    run_name="adaptive_universal_policy_torch4090",
    torch_gpu_profile="rtx4090",
    torch_hidden_dim=768,
    torch_mlp_depth=3,
    torch_batch_episodes=1536,
    torch_minibatch_size=768,
    torch_update_epochs=4,
    torch_entropy_coef=0.008,
    torch_preflight_batch_size=12288,
    torch_preflight_min_free_memory_gb=10.0,
)
