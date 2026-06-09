"""Direct Preference Optimization (DPO) alignment for LLM post-training.

Experimental alignment path, separate from the RL quantization policy trainers.
Requires optional deps: ``pip install -e ".[alignment]"``.
"""

from __future__ import annotations

import importlib
from typing import Any

from adaptive_quant.llm_alignment.config import DPOSettings

_EAGER_EXPORTS = ("DPOSettings",)

_LAZY: dict[str, tuple[str, str]] = {
    "DPODataCollator": ("adaptive_quant.llm_alignment.data_collator", "DPODataCollator"),
    "DPOMetrics": ("adaptive_quant.llm_alignment.dpo_loss", "DPOMetrics"),
    "DPOTrainer": ("adaptive_quant.llm_alignment.dpo_trainer", "DPOTrainer"),
    "clone_reference_from_policy": (
        "adaptive_quant.llm_alignment.model_loading",
        "clone_reference_from_policy",
    ),
    "compute_dpo_loss": ("adaptive_quant.llm_alignment.dpo_loss", "compute_dpo_loss"),
    "get_batch_logps": ("adaptive_quant.llm_alignment.dpo_loss", "get_batch_logps"),
    "load_policy_and_reference": (
        "adaptive_quant.llm_alignment.model_loading",
        "load_policy_and_reference",
    ),
    "load_preference_dataset": (
        "adaptive_quant.llm_alignment.preference_data",
        "load_preference_dataset",
    ),
}

__all__ = sorted((*_EAGER_EXPORTS, *_LAZY))


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    return getattr(importlib.import_module(module_name), attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
