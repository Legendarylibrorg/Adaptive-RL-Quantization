"""Model + quantization **route** catalog.

A *route* is a tuple of ``(model repository, quantized file, target hardware hints)``. Each
route names something a user could fetch with ``huggingface-cli`` (or ``hf``) and feed into a
``llama.cpp``-compatible runner. The bandit learner in :mod:`adaptive_quant.route_policy`
treats each route as an arm and learns which arm wins for which task / hardware bucket.

The catalog is persisted as a JSON document so multiple runs can share / extend it. Loading
is strict: unknown fields raise so typos in registry contributions fail fast.

Quantization labels are mapped to **effective bits per weight** via :data:`QUANT_BITS`. The
estimates here come from llama.cpp's published k-quant breakdowns (and the F16/F32 baselines)
and are deliberately conservative — the bandit only needs *relative* differences to learn a
ranking, so small absolute errors do not bias the learned ordering.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from adaptive_quant.configuration.validation import (
    validate_hf_filename,
    validate_hf_model_id,
    validate_hf_revision,
    validate_optional_filesystem_path,
)
from adaptive_quant.logging_utils import read_json, write_json

# llama.cpp / GGUF effective bits per weight. K-quant family figures from llama.cpp release notes
# (e.g. "Q4_K_M ~= 4.83 bpw"). Plain Q4_0/Q4_1 are slightly lower because they lack the K-quant
# super-block overhead; F16/F32 are exact baselines.
QUANT_BITS: dict[str, float] = {
    # 2-bit
    "Q2_K": 2.625,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ2_M": 2.70,
    # 3-bit
    "Q3_K_S": 3.50,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "IQ3_XXS": 3.06,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    # 4-bit
    "Q4_0": 4.55,
    "Q4_1": 4.75,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.83,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.50,
    # 5-bit
    "Q5_0": 5.54,
    "Q5_1": 5.74,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    # 6-bit
    "Q6_K": 6.56,
    # 8-bit
    "Q8_0": 8.50,
    # baselines
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

# Acceptable hardware affinity labels — used as soft hints for the bandit.
_HARDWARE_HINTS: frozenset[str] = frozenset({"gpu", "cpu", "low_resource", "any"})

_ROUTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class QuantSpec:
    """Description of a quantization label.

    ``effective_bits`` is the *bits per weight* used by reward shaping; it is independent of
    the underlying GGUF metadata. ``family`` is currently always ``"gguf"`` but is plumbed
    through so AWQ / GPTQ / EXL2 routes can be added without breaking the schema.
    """

    label: str
    effective_bits: float
    family: str = "gguf"

    @classmethod
    def from_label(cls, label: str, *, family: str = "gguf") -> "QuantSpec":
        normalized = label.strip().upper()
        bits = QUANT_BITS.get(normalized)
        if bits is None:
            raise KeyError(
                f"Unknown quant label {label!r}. Known: {sorted(QUANT_BITS)}. "
                "Pass effective_bits explicitly via ModelRoute(effective_bits=...) for novel quants."
            )
        return cls(label=normalized, effective_bits=float(bits), family=family)


@dataclass
class ModelRoute:
    """A single route the bandit can choose from."""

    route_id: str
    repo_id: str
    quant_label: str
    filename: str | None = None
    revision: str | None = None
    family: str = "gguf"
    parameters_b: float | None = None
    size_mb: float | None = None
    effective_bits: float | None = None
    hardware_hints: tuple[str, ...] = ("any",)
    domain_hints: tuple[str, ...] = ()
    notes: str = ""
    local_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not _ROUTE_ID_RE.match(self.route_id):
            raise ValueError(
                f"Invalid route_id {self.route_id!r}: expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/."
            )
        if self.local_path is not None:
            validate_optional_filesystem_path("local_path", self.local_path)
        validate_hf_model_id("repo_id", self.repo_id, require_hub_namespace=True)
        if self.filename is not None:
            validate_hf_filename("filename", self.filename)
            if self.family.strip().lower() == "gguf" and not self.filename.lower().endswith(
                ".gguf"
            ):
                raise ValueError(
                    f"GGUF route filename must end with '.gguf', got {self.filename!r}"
                )
        if self.revision is not None:
            validate_hf_revision("revision", self.revision)
        normalized_quant = (
            self.quant_label.strip().upper() if isinstance(self.quant_label, str) else ""
        )
        if not normalized_quant:
            raise ValueError("quant_label is required for ModelRoute")
        object.__setattr__(self, "quant_label", normalized_quant)

        if self.effective_bits is None:
            spec = QuantSpec.from_label(normalized_quant, family=self.family)
            object.__setattr__(self, "effective_bits", float(spec.effective_bits))
        else:
            object.__setattr__(self, "effective_bits", float(self.effective_bits))

        normalized_hints = tuple(hint.strip().lower() for hint in self.hardware_hints if hint)
        if not normalized_hints:
            normalized_hints = ("any",)
        unknown = [hint for hint in normalized_hints if hint not in _HARDWARE_HINTS]
        if unknown:
            raise ValueError(
                f"Unknown hardware_hints {unknown!r}; allowed: {sorted(_HARDWARE_HINTS)}"
            )
        object.__setattr__(self, "hardware_hints", normalized_hints)

        normalized_domains = tuple(domain.strip().lower() for domain in self.domain_hints if domain)
        object.__setattr__(self, "domain_hints", normalized_domains)

        if self.parameters_b is not None and float(self.parameters_b) <= 0:
            raise ValueError("parameters_b must be > 0 when set")
        if self.size_mb is not None and float(self.size_mb) <= 0:
            raise ValueError("size_mb must be > 0 when set")

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            label=self.quant_label, effective_bits=self.effective_bits or 0.0, family=self.family
        )

    def matches_hardware(self, hardware_value: str) -> bool:
        return "any" in self.hardware_hints or hardware_value in self.hardware_hints

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hardware_hints"] = list(self.hardware_hints)
        payload["domain_hints"] = list(self.domain_hints)
        return payload


@dataclass
class RouteCatalog:
    """JSON-backed registry of :class:`ModelRoute` instances."""

    routes: list[ModelRoute] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for route in self.routes:
            if route.route_id in seen:
                raise ValueError(f"Duplicate route_id in catalog: {route.route_id!r}")
            seen.add(route.route_id)

    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self):
        return iter(self.routes)

    def add(self, route: ModelRoute, *, replace_existing: bool = False) -> None:
        for index, existing in enumerate(self.routes):
            if existing.route_id == route.route_id:
                if not replace_existing:
                    raise ValueError(
                        f"Route already registered: {route.route_id!r} (use replace_existing=True to overwrite)"
                    )
                self.routes[index] = route
                return
        self.routes.append(route)

    def remove(self, route_id: str) -> bool:
        for index, existing in enumerate(self.routes):
            if existing.route_id == route_id:
                self.routes.pop(index)
                return True
        return False

    def by_id(self, route_id: str) -> ModelRoute:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(f"Unknown route_id: {route_id!r}")

    def filter(
        self,
        *,
        hardware: str | None = None,
        domain: str | None = None,
        max_effective_bits: float | None = None,
        max_size_mb: float | None = None,
    ) -> list[ModelRoute]:
        result: list[ModelRoute] = []
        for route in self.routes:
            if hardware is not None and not route.matches_hardware(hardware):
                continue
            if (
                domain is not None
                and route.domain_hints
                and domain.lower() not in route.domain_hints
            ):
                continue
            if (
                max_effective_bits is not None
                and (route.effective_bits or 0.0) > max_effective_bits
            ):
                continue
            if (
                max_size_mb is not None
                and route.size_mb is not None
                and route.size_mb > max_size_mb
            ):
                continue
            result.append(route)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteCatalog":
        if not isinstance(data, Mapping):
            raise TypeError(f"RouteCatalog payload must be a mapping, got {type(data).__name__}")
        raw_routes = data.get("routes", [])
        if not isinstance(raw_routes, list):
            raise TypeError("RouteCatalog 'routes' must be a list")
        valid_keys = {f.name for f in fields(ModelRoute)}
        routes: list[ModelRoute] = []
        for index, payload in enumerate(raw_routes):
            if not isinstance(payload, Mapping):
                raise TypeError(f"routes[{index}] must be an object")
            unknown = set(payload) - valid_keys
            if unknown:
                raise ValueError(f"Unknown ModelRoute keys at routes[{index}]: {sorted(unknown)}")
            kwargs = dict(payload)
            for key in ("hardware_hints", "domain_hints"):
                if key in kwargs and isinstance(kwargs[key], list):
                    kwargs[key] = tuple(str(value) for value in kwargs[key])
            routes.append(ModelRoute(**kwargs))
        return cls(routes=routes)

    def to_dict(self) -> dict[str, Any]:
        return {"routes": [route.to_dict() for route in self.routes]}

    @classmethod
    def from_file(cls, path: str | Path) -> "RouteCatalog":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Route catalog not found: {target}")
        return cls.from_dict(read_json(target, label="Route catalog"))

    def save(self, path: str | Path) -> str:
        target = Path(path)
        write_json(str(target), self.to_dict())
        return str(target)

    def update_local_path(self, route_id: str, local_path: str | None) -> ModelRoute:
        validate_optional_filesystem_path("local_path", local_path)
        for index, existing in enumerate(self.routes):
            if existing.route_id == route_id:
                updated = replace(existing, local_path=local_path)
                self.routes[index] = updated
                return updated
        raise KeyError(f"Unknown route_id: {route_id!r}")


def default_route_catalog() -> RouteCatalog:
    """A small curated default registry covering popular open GGUF quants.

    These entries reference well-known community quantizations on the Hub. Users are expected
    to extend / replace this list for their own deployments via ``adaptive-rl-quant-route register``
    (see :mod:`run_route_learning`). The defaults are intentionally diverse along three axes —
    parameter count (1B-8B), quantization tier (2-bit through 8-bit), and hardware affinity —
    so the bandit has signal to learn from on day one.
    """
    return RouteCatalog(
        routes=[
            ModelRoute(
                route_id="llama31-8b-q4km",
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                quant_label="Q4_K_M",
                parameters_b=8.0,
                size_mb=4920.0,
                hardware_hints=("gpu", "cpu"),
                notes="Balanced general-purpose route; common community default.",
            ),
            ModelRoute(
                route_id="llama31-8b-q5km",
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                quant_label="Q5_K_M",
                parameters_b=8.0,
                size_mb=5730.0,
                hardware_hints=("gpu",),
                notes="Higher quality 8B route; prefers GPU memory budgets.",
            ),
            ModelRoute(
                route_id="llama31-8b-q8",
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                quant_label="Q8_0",
                parameters_b=8.0,
                size_mb=8540.0,
                hardware_hints=("gpu",),
                notes="High-fidelity 8-bit route; best on >=12 GB VRAM.",
            ),
            ModelRoute(
                route_id="llama31-8b-q3km",
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf",
                quant_label="Q3_K_M",
                parameters_b=8.0,
                size_mb=4020.0,
                hardware_hints=("cpu", "low_resource"),
                notes="Memory-efficient 8B variant for CPU / small GPU.",
            ),
            ModelRoute(
                route_id="qwen25-7b-q4km",
                repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
                filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                quant_label="Q4_K_M",
                parameters_b=7.0,
                size_mb=4540.0,
                hardware_hints=("gpu", "cpu"),
                domain_hints=("code", "reasoning"),
                notes="Strong code / reasoning baseline at 7B.",
            ),
            ModelRoute(
                route_id="qwen25-7b-q6k",
                repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
                filename="Qwen2.5-7B-Instruct-Q6_K.gguf",
                quant_label="Q6_K",
                parameters_b=7.0,
                size_mb=6260.0,
                hardware_hints=("gpu",),
                domain_hints=("code", "reasoning"),
                notes="Higher quality Qwen 7B for GPU.",
            ),
            ModelRoute(
                route_id="phi35-mini-q4km",
                repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
                filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
                quant_label="Q4_K_M",
                parameters_b=3.8,
                size_mb=2390.0,
                hardware_hints=("cpu", "low_resource", "gpu"),
                notes="Compact instruction model; great fit on small CPU/edge.",
            ),
            ModelRoute(
                route_id="llama32-1b-q8",
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
                quant_label="Q8_0",
                parameters_b=1.24,
                size_mb=1320.0,
                hardware_hints=("low_resource", "cpu"),
                notes="Smallest viable route for low-resource devices; fast and high-quality at this size.",
            ),
        ]
    )


__all__ = [
    "ModelRoute",
    "QUANT_BITS",
    "QuantSpec",
    "RouteCatalog",
    "default_route_catalog",
]
