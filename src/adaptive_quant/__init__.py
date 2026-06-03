"""Train an RL policy to choose quantization (and optional MoE variants) using prompt + hardware context.

The default path is a **stdlib simulator**; optional **PyTorch** accelerates training and optional **llama.cpp**
replaces simulated metrics when you configure a binary and GGUF. Public symbols listed in ``__all__`` are
lazy-imported where useful so importing ``adaptive_quant`` does not pull optional CUDA stacks until needed.

Entrypoints: ``run_research.py`` (Python trainer), ``run_pytorch.py`` (CUDA trainer),
``run_online_learning.py`` (continual adaptation), ``ResearchPipeline`` / ``run_pipeline_entrypoint``,
and ``run_online_pipeline`` for programmatic runs. Configure via ``FrameworkConfig`` or JSON/TOML
(``load_config`` / ``FrameworkConfig.from_file``).
"""

from __future__ import annotations

import importlib
from typing import Any

from adaptive_quant.configuration import FrameworkConfig as FrameworkConfig
from adaptive_quant.gpu_profiles import apply_gpu_profile as apply_gpu_profile
from adaptive_quant.gpu_profiles import available_gpu_profiles as available_gpu_profiles
from adaptive_quant.types import OnlineRequest as OnlineRequest

_EAGER_EXPORTS = (
    "FrameworkConfig",
    "OnlineRequest",
    "apply_gpu_profile",
    "available_gpu_profiles",
)

_LAZY: dict[str, tuple[str, str]] = {
    "BenchmarkSuite": ("adaptive_quant.benchmark", "BenchmarkSuite"),
    "AdaptiveQuantizationEnv": ("adaptive_quant.environment", "AdaptiveQuantizationEnv"),
    "OnlineLearningLoop": ("adaptive_quant.online_learning", "OnlineLearningLoop"),
    "Trainer": ("adaptive_quant.trainer", "Trainer"),
    "UniversalQuantizationPolicy": ("adaptive_quant.policy", "UniversalQuantizationPolicy"),
    "build_request_stream": ("adaptive_quant.online_learning", "build_request_stream"),
    "build_trainer": ("adaptive_quant.trainer", "build_trainer"),
    "detect_host_hardware": ("adaptive_quant.hardware", "detect_host_hardware"),
    "load_config": ("adaptive_quant.easy_config", "load_config"),
    "quick_config": ("adaptive_quant.easy_config", "quick_config"),
    "recommend_quantization": ("adaptive_quant.recommendation", "recommend_quantization"),
    "run_online_pipeline": ("adaptive_quant.online_pipeline", "run_online_pipeline"),
    "ResearchPipeline": ("adaptive_quant.research_pipeline", "ResearchPipeline"),
    "run_pipeline_entrypoint": ("adaptive_quant.research_pipeline", "run_pipeline_entrypoint"),
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
