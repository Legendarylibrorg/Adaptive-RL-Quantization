from adaptive_quant.configuration import FrameworkConfig


# Canonical offline research baseline used by the simplest reproducible workflow.
# Use training_backend="pytorch" on a CUDA host for GPU/VRAM utilization.
CONFIG = FrameworkConfig(
    multi_hardware=True,
    dynamic_quant=True,
    learned_quant=True,
    quant_mode="hybrid",
    hardware_modes=("gpu", "cpu", "low_resource"),
    training_episodes=10_000,
    evaluation_episodes=500,
    continuous_training=True,
    eval_interval=1_000,
    checkpoint_interval=5_000,
    max_training_episodes=100_000,
    replay_buffer_capacity=50_000,
    replay_buffer_on_gpu=True,
    run_name="adaptive_universal_policy",
)
