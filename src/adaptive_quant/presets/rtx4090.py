from adaptive_quant.presets.gpu import make_rtx_torch_preset

CONFIG_4090 = make_rtx_torch_preset(
    training_host_label="rtx4090",
    run_name="adaptive_universal_policy_torch4090",
    torch_gpu_profile="rtx4090",
)
