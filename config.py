from adaptive_quant.configuration import FrameworkConfig


CONFIG = FrameworkConfig(
    multi_hardware=True,
    dynamic_quant=True,
    learned_quant=True,
    quant_mode="hybrid",
    hardware_modes=("gpu", "cpu", "low_resource"),
    training_episodes=240,
    evaluation_episodes=60,
    run_name="adaptive_universal_policy",
)

