from adaptive_quant.configuration import FrameworkConfig

# Easier overrides: use JSON/TOML — e.g.
#   CONFIG = FrameworkConfig.from_file("config.e2e_smoke.json")
#   CONFIG = FrameworkConfig.from_file("config.example.json")
# or: from adaptive_quant import load_config; CONFIG = load_config("my.toml")

# Canonical offline research baseline used by the simplest reproducible workflow.
# Default: fixed horizon (completes in reasonable time). For long runs set
# continuous_training=True and max_training_episodes; for CUDA + VRAM use
# training_backend="pytorch" (see config_gpu.py).
CONFIG = FrameworkConfig(
    multi_hardware=True,
    dynamic_quant=True,
    learned_quant=True,
    quant_mode="hybrid",
    hardware_modes=("gpu", "cpu", "low_resource"),
    training_episodes=3_000,
    evaluation_episodes=400,
    continuous_training=False,
    eval_interval=1_000,
    checkpoint_interval=5_000,
    max_training_episodes=50_000,
    replay_buffer_capacity=20_000,
    replay_buffer_on_gpu=True,
    run_name="adaptive_universal_policy",
)
