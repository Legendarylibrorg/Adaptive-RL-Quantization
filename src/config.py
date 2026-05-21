"""Top-level preset exports for editable installs (``from config import CONFIG``).

Preset definitions live in :mod:`adaptive_quant.presets`. Import named constants from
this module or from ``adaptive_quant.presets`` directly; legacy per-preset ``config_*.py``
shim modules were removed in favor of this single entry point.
"""

from adaptive_quant.presets import (
    CONFIG,
    CONFIG_3090,
    CONFIG_4090,
    CONFIG_4090_UNIVERSAL,
    CONFIG_GPU,
    CONFIG_MOE,
    CONFIG_ONLINE,
    make_rtx_torch_preset,
)

__all__ = [
    "CONFIG",
    "CONFIG_3090",
    "CONFIG_4090",
    "CONFIG_4090_UNIVERSAL",
    "CONFIG_GPU",
    "CONFIG_MOE",
    "CONFIG_ONLINE",
    "make_rtx_torch_preset",
]
