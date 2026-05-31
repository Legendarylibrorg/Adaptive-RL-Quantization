from adaptive_quant.presets.gpu import make_rtx_torch_preset

CONFIG_3090 = make_rtx_torch_preset(
    training_host_label="rtx3090",
    run_name="adaptive_universal_policy_torch3090",
    torch_gpu_profile="rtx3090",
)
