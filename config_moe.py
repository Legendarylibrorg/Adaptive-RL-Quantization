from adaptive_quant.configuration import FrameworkConfig


CONFIG_MOE = FrameworkConfig(
    multi_hardware=True,
    dynamic_quant=True,
    learned_quant=True,
    moe_enabled=True,
    quant_mode="hybrid",
    hardware_modes=("gpu", "cpu", "low_resource"),
    moe_num_experts=16,
    moe_top_k=2,
    moe_variant_names=("safe", "balanced", "aggressive"),
    moe_max_aggressive_experts=1,
    moe_max_swap_cost_ms=7.5,
    training_episodes=320,
    evaluation_episodes=72,
    benchmark_training_episodes=96,
    benchmark_evaluation_episodes=24,
    run_name="adaptive_moe_policy",
)
