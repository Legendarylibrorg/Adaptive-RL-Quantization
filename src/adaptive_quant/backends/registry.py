from __future__ import annotations

from collections.abc import Callable

from adaptive_quant.backends.llama_cpp import LlamaCppBackend
from adaptive_quant.backends.protocol import Backend
from adaptive_quant.backends.simulator import SimulatorBackend
from adaptive_quant.configuration import FrameworkConfig

BackendBuilder = Callable[[FrameworkConfig], Backend]

_EXTRA_BUILDERS: dict[str, BackendBuilder] = {}


def register_backend(name: str, builder: BackendBuilder) -> None:
    """Register a custom backend builder (lowercased key). Built-ins: simulator, llama_cpp."""
    _EXTRA_BUILDERS[name.strip().lower()] = builder


def build_backend(config: FrameworkConfig) -> Backend:
    """Build the measurement backend configured for an experiment."""
    backend = config.backend.strip().lower()
    custom = _EXTRA_BUILDERS.get(backend)
    if custom is not None:
        return custom(config)
    if backend == "simulator":
        return SimulatorBackend(config)
    if backend == "llama_cpp":
        return LlamaCppBackend(config)
    raise ValueError(f"Unsupported backend {config.backend!r}; expected 'simulator' or 'llama_cpp'.")
