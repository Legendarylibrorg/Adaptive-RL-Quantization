"""Curated :class:`FrameworkConfig` presets for CLI entrypoints and examples."""

from adaptive_quant.presets.baseline import CONFIG
from adaptive_quant.presets.gpu import CONFIG_GPU, make_rtx_torch_preset
from adaptive_quant.presets.moe import CONFIG_MOE
from adaptive_quant.presets.online import CONFIG_ONLINE
from adaptive_quant.presets.post_train import CONFIG_POST_TRAIN
from adaptive_quant.presets.rtx3090 import CONFIG_3090
from adaptive_quant.presets.rtx4090 import CONFIG_4090
from adaptive_quant.presets.rtx4090_universal import CONFIG_4090_UNIVERSAL

__all__ = [
    "CONFIG",
    "CONFIG_3090",
    "CONFIG_4090",
    "CONFIG_4090_UNIVERSAL",
    "CONFIG_GPU",
    "CONFIG_MOE",
    "CONFIG_ONLINE",
    "CONFIG_POST_TRAIN",
    "make_rtx_torch_preset",
]
