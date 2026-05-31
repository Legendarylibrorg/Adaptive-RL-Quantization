from adaptive_quant.gpu_profiles import GPU_PROFILES
from adaptive_quant.presets.baseline import CONFIG

CONFIG_GPU = CONFIG.clone(
    training_backend="pytorch",
    continuous_training=False,
    training_episodes=4096,
    evaluation_episodes=256,
    benchmark_training_episodes=1024,
    benchmark_evaluation_episodes=128,
    run_name="adaptive_universal_policy_torch_gpu",
    cache_prompt_features=True,
    log_every_n_episodes=8,
    torch_device="cuda",
    torch_gpu_profile="auto",
    torch_dtype="bfloat16",
    torch_compile=True,
    torch_amp=True,
    torch_tf32=True,
    torch_hidden_dim=768,
    torch_mlp_depth=3,
    torch_learning_rate=3e-4,
    torch_weight_decay=1e-4,
    torch_batch_episodes=1024,
    torch_minibatch_size=512,
    torch_update_epochs=4,
    torch_ppo_clip=0.2,
    torch_value_coef=0.5,
    torch_entropy_coef=0.008,
    torch_max_grad_norm=1.0,
    torch_fused_optimizer=True,
    torch_preflight=True,
    torch_preflight_batch_size=8192,
    torch_preflight_warmup_steps=12,
    torch_preflight_steps=48,
    torch_preflight_min_free_memory_gb=10.0,
)


def make_rtx_torch_preset(
    *,
    training_host_label: str,
    run_name: str,
    torch_gpu_profile: str,
    benchmark_training_episodes: int | None = None,
    benchmark_evaluation_episodes: int | None = None,
    torch_hidden_dim: int | None = None,
    torch_mlp_depth: int | None = None,
    torch_batch_episodes: int | None = None,
    torch_minibatch_size: int | None = None,
    torch_update_epochs: int | None = None,
    torch_entropy_coef: float | None = None,
    torch_preflight_batch_size: int | None = None,
    torch_preflight_min_free_memory_gb: float | None = None,
):
    profile_overrides = dict(GPU_PROFILES[torch_gpu_profile].overrides)
    explicit_overrides = {
        "benchmark_training_episodes": benchmark_training_episodes,
        "benchmark_evaluation_episodes": benchmark_evaluation_episodes,
        "torch_hidden_dim": torch_hidden_dim,
        "torch_mlp_depth": torch_mlp_depth,
        "torch_batch_episodes": torch_batch_episodes,
        "torch_minibatch_size": torch_minibatch_size,
        "torch_update_epochs": torch_update_epochs,
        "torch_entropy_coef": torch_entropy_coef,
        "torch_preflight_batch_size": torch_preflight_batch_size,
        "torch_preflight_min_free_memory_gb": torch_preflight_min_free_memory_gb,
    }
    profile_overrides.update(
        {key: value for key, value in explicit_overrides.items() if value is not None}
    )
    return CONFIG_GPU.clone(
        training_host_label=training_host_label,
        run_name=run_name,
        torch_gpu_profile=torch_gpu_profile,
        **profile_overrides,
    )
