from adaptive_quant.presets.gpu import make_rtx_torch_preset

CONFIG_3090 = make_rtx_torch_preset(
    training_host_label="rtx3090",
    benchmark_training_episodes=640,
    benchmark_evaluation_episodes=80,
    run_name="adaptive_universal_policy_torch3090",
    torch_gpu_profile="rtx3090",
    torch_hidden_dim=640,
    torch_mlp_depth=3,
    torch_batch_episodes=768,
    torch_minibatch_size=384,
    torch_update_epochs=4,
    torch_entropy_coef=0.008,
    torch_preflight_batch_size=6144,
    torch_preflight_min_free_memory_gb=9.0,
)
