from adaptive_quant.presets.rtx4090 import CONFIG_4090

CONFIG_4090_UNIVERSAL = CONFIG_4090.clone(
    run_name="adaptive_universal_policy_host4090",
    training_host_label="rtx4090",
    multi_hardware=True,
    hardware_modes=("gpu", "cpu", "low_resource"),
)
