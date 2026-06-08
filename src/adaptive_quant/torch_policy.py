from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.math_utils import discrete_precision_level
from adaptive_quant.trainer_utils import zero_previous_action
from adaptive_quant.types import QuantizationDecision, QuantMode

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Normal
except Exception as exc:  # pragma: no cover - exercised only when torch is unavailable
    torch = None
    nn = None
    Categorical = None
    Normal = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

TORCH_BACKEND_REQUIRED_MESSAGE = (
    'PyTorch is required for `training_backend="pytorch"`. '
    "Install PyTorch (CUDA build recommended for GPU; CPU-only installs run with device fallback)."
)


def torch_cuda_diagnostics(requested_device: str = "cuda") -> dict[str, Any]:
    """Return a small, JSON-friendly report for the active Torch/CUDA stack."""
    if torch is None:
        return {
            "torch_installed": False,
            "torch_import_error": repr(TORCH_IMPORT_ERROR),
            "cuda_available": False,
        }

    report: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": requested_device,
    }
    if not torch.cuda.is_available():
        return report

    device = torch.device(requested_device or "cuda")
    index = (
        device.index
        if device.type == "cuda" and device.index is not None
        else torch.cuda.current_device()
    )
    props = torch.cuda.get_device_properties(index)
    major, minor = int(props.major), int(props.minor)
    arch = f"sm_{major}{minor}"
    arch_list = _safe_cuda_arch_list()
    report.update(
        {
            "device_index": int(index),
            "device_name": str(props.name),
            "device_capability": f"{major}.{minor}",
            "required_arch": arch,
            "torch_cuda_arch_list": arch_list,
            "cuda_arch_supported": _cuda_arch_supported(major, minor, arch_list),
        }
    )
    if arch_list and not report["cuda_arch_supported"]:
        from adaptive_quant.torch_install import INSTALL_CUDA_TORCH_SCRIPT

        report["cuda_arch_warning"] = (
            f"Active CUDA device {props.name!s} requires {arch} support, but this "
            f"PyTorch build reports architectures: {', '.join(arch_list)}."
        )
        report["install_hint"] = (
            "Install a CUDA-enabled PyTorch wheel that includes this GPU architecture: "
            f"{INSTALL_CUDA_TORCH_SCRIPT}"
        )
    elif "4090" in str(props.name).lower() and arch != "sm_89":
        report["cuda_device_warning"] = (
            f"Expected an RTX 4090-class compute capability of sm_89, got {arch}."
        )
    return report


def validate_cuda_runtime_compatibility(requested_device: str = "cuda") -> None:
    """Fail early when a CUDA device is visible but the active Torch build cannot run it."""
    diagnostics = torch_cuda_diagnostics(requested_device)
    if not diagnostics.get("torch_installed", False):
        raise ImportError(TORCH_BACKEND_REQUIRED_MESSAGE) from TORCH_IMPORT_ERROR
    if not diagnostics.get("cuda_available", False):
        return
    arch_list = diagnostics.get("torch_cuda_arch_list") or []
    if arch_list and not diagnostics.get("cuda_arch_supported", True):
        hint = diagnostics.get("install_hint") or (
            "Install a CUDA-enabled PyTorch wheel that includes this GPU architecture."
        )
        raise RuntimeError(f"{diagnostics['cuda_arch_warning']} {hint}")


def resolve_training_device(
    requested: str, *, require_cuda: bool = False
) -> tuple["torch.device", str | None]:
    """
    Map config.torch_device to a device that exists on this machine.
    Falls back to CPU when CUDA/MPS was requested but is not available,
    unless ``require_cuda`` is set (GPU presets default to True).
    """
    if torch is None:
        raise RuntimeError("resolve_training_device requires torch")
    raw = (requested or "cpu").strip()
    device = torch.device(raw)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            extra = ""
            try:
                from adaptive_quant.hardware import detect_cuda_device

                smi_cuda = detect_cuda_device(prefer_torch=False)
            except Exception:
                smi_cuda = None
            if smi_cuda is not None:
                extra = (
                    f" nvidia-smi sees {smi_cuda.name} ({smi_cuda.total_memory_gb:.2f} GB), "
                    "so the active PyTorch install is probably CPU-only or linked to an "
                    "incompatible CUDA runtime."
                )
            if require_cuda:
                from adaptive_quant.torch_install import cuda_torch_install_instructions

                raise RuntimeError(
                    f"torch_device={requested!r} but CUDA is not available.{extra}\n\n"
                    f"{cuda_torch_install_instructions()}"
                )
            return torch.device(
                "cpu"
            ), f"torch_device={requested!r} but CUDA is not available; using CPU.{extra}"
        validate_cuda_runtime_compatibility(raw)
        return device, None
    if device.type == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            return torch.device(
                "cpu"
            ), f"torch_device={requested!r} but MPS is not available; using CPU."
        return device, None
    return device, None


def configure_global_torch_reproducibility(config: FrameworkConfig) -> None:
    """
    Best-effort bit-level reproducibility for CUDA matmuls and PyTorch ops.
    Call once before building CUDA models or running GPU kernels for training.

    Sets CUBLAS_WORKSPACE_CONFIG when unset (required for deterministic cuBLAS).
    """
    if torch is None or not config.torch_deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


if torch is not None:

    class TorchActorCritic(nn.Module):
        def __init__(self, config: FrameworkConfig) -> None:
            super().__init__()
            self.config = config
            self.supported_modes = config.supported_modes()
            self.fixed_mode = config.resolved_quant_mode()
            self.num_bits = len(config.discrete_bit_widths)
            self.state_dim = config.state_vector_dim()

            layers: list[nn.Module] = []
            input_dim = self.state_dim
            for _ in range(config.torch_mlp_depth):
                layers.extend(
                    [
                        nn.Linear(input_dim, config.torch_hidden_dim),
                        nn.LayerNorm(config.torch_hidden_dim),
                        nn.SiLU(),
                    ]
                )
                input_dim = config.torch_hidden_dim
            self.encoder = nn.Sequential(*layers)

            self.mode_head = (
                nn.Linear(config.torch_hidden_dim, len(self.supported_modes))
                if self.fixed_mode == QuantMode.HYBRID
                else None
            )
            self.discrete_head = nn.Linear(config.torch_hidden_dim, self.num_bits)
            self.group_head = nn.Linear(config.torch_hidden_dim, config.num_groups * self.num_bits)
            self.layer_head = nn.Linear(config.torch_hidden_dim, config.num_layers * self.num_bits)
            self.moe_head = (
                nn.Linear(config.torch_hidden_dim, config.moe_top_k * config.moe_variant_count())
                if config.moe_enabled
                else None
            )
            self.learned_mean = nn.Linear(config.torch_hidden_dim, 3)
            self.learned_log_std = nn.Parameter(
                torch.tensor([-0.35, -0.35, -0.55], dtype=torch.float32)
            )
            self.value_head = nn.Linear(config.torch_hidden_dim, 1)

            self.register_buffer(
                "learned_lower",
                torch.tensor(
                    [config.scale_bounds[0], config.clip_bounds[0], config.precision_bounds[0]],
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "learned_upper",
                torch.tensor(
                    [config.scale_bounds[1], config.clip_bounds[1], config.precision_bounds[1]],
                    dtype=torch.float32,
                ),
            )
            self._initialize_parameters()

        def _initialize_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.8)
                    nn.init.zeros_(module.bias)
            nn.init.constant_(self.learned_mean.bias, -0.25)

        def forward(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
            features = self.encoder(states.float())
            outputs = {
                "features": features,
                "discrete_logits": self.discrete_head(features),
                "group_logits": self.group_head(features).view(
                    -1, self.config.num_groups, self.num_bits
                ),
                "layer_logits": self.layer_head(features).view(
                    -1, self.config.num_layers, self.num_bits
                ),
                "learned_mean": self.learned_mean(features),
                "learned_std": self.learned_log_std.exp()
                .unsqueeze(0)
                .expand(features.shape[0], -1),
                "value": self.value_head(features).squeeze(-1),
            }
            if self.mode_head is not None:
                outputs["mode_logits"] = self.mode_head(features)
            if self.moe_head is not None:
                outputs["moe_logits"] = self.moe_head(features).view(
                    -1, self.config.moe_top_k, self.config.moe_variant_count()
                )
            return outputs

        def map_learned_parameters(self, raw_parameters: torch.Tensor) -> torch.Tensor:
            bounded = torch.sigmoid(raw_parameters)
            return self.learned_lower + bounded * (self.learned_upper - self.learned_lower)


class TorchPolicyAdapter:
    def __init__(self, config: FrameworkConfig) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is not installed in this environment."
            ) from TORCH_IMPORT_ERROR
        self.config = config
        configure_global_torch_reproducibility(config)
        self.supported_modes = config.supported_modes()
        self.mode_to_index = {mode: index for index, mode in enumerate(self.supported_modes)}
        self.device, device_note = resolve_training_device(
            config.torch_device,
            require_cuda=config.torch_require_cuda,
        )
        if device_note:
            warnings.warn(device_note, UserWarning, stacklevel=2)
        self.compute_dtype = _resolve_autocast_dtype(config.torch_dtype)
        self.model = TorchActorCritic(config).to(self.device)
        self._configure_cuda()
        if config.torch_compile and not config.torch_deterministic and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception as exc:
                warnings.warn(
                    f"torch.compile failed; continuing with eager execution: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
        elif config.torch_compile and config.torch_deterministic:
            warnings.warn(
                "torch_compile is disabled when torch_deterministic=True (compile is not reproducibility-safe).",
                UserWarning,
                stacklevel=2,
            )

    def _configure_cuda(self) -> None:
        if self.device.type != "cuda":
            return
        if self.config.torch_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.config.torch_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def autocast_context(self):
        if self.device.type != "cuda" or not self.config.torch_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.compute_dtype)

    def state_tensor(self, state_vectors: list[list[float]]) -> torch.Tensor:
        return torch.tensor(state_vectors, dtype=torch.float32, device=self.device)

    def act(
        self,
        state_vector: list[float],
        deterministic: bool = False,
        *,
        moe_context: Any | None = None,
    ) -> tuple[QuantizationDecision, dict[str, Any]]:
        state = self.state_tensor([state_vector])
        with torch.inference_mode():
            with self.autocast_context():
                outputs = self.model(state)

        selected_mode, mode_index, log_prob, entropy = self._sample_mode(
            outputs, deterministic=deterministic
        )
        record: dict[str, Any] = {
            "mode_index": mode_index,
            "selected_mode": selected_mode.value,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": float(outputs["value"][0].item()),
            "discrete_index": None,
            "group_indices": [],
            "layer_indices": [],
            "moe_indices": [],
            "moe_active": False,
            "learned_raw": [],
        }

        if selected_mode in {QuantMode.DISCRETE, QuantMode.DYNAMIC}:
            logits = outputs["discrete_logits"][0]
            distribution = Categorical(logits=logits)
            index = (
                int(torch.argmax(logits).item())
                if deterministic
                else int(distribution.sample().item())
            )
            record["discrete_index"] = index
            record["log_prob"] += float(
                distribution.log_prob(torch.tensor(index, device=self.device)).item()
            )
            record["entropy"] += float(distribution.entropy().item())
            bit_width = self.config.discrete_bit_widths[index]
            decision = QuantizationDecision(
                mode=selected_mode,
                base_bit_width=bit_width,
                scale_factor=1.0,
                clipping_range=1.0,
                precision_level=discrete_precision_level(
                    bit_width, self.config.discrete_bit_widths
                ),
                metadata={"head": "torch_discrete"},
            )
            self._attach_moe_selection(
                outputs, decision, record, deterministic, moe_context=moe_context
            )
            return decision, record

        if selected_mode == QuantMode.GROUPED:
            group_indices: list[int] = []
            group_bits: list[int] = []
            for slot in range(self.config.num_groups):
                logits = outputs["group_logits"][0, slot]
                distribution = Categorical(logits=logits)
                index = (
                    int(torch.argmax(logits).item())
                    if deterministic
                    else int(distribution.sample().item())
                )
                record["log_prob"] += float(
                    distribution.log_prob(torch.tensor(index, device=self.device)).item()
                )
                record["entropy"] += float(distribution.entropy().item())
                group_indices.append(index)
                group_bits.append(self.config.discrete_bit_widths[index])
            record["group_indices"] = group_indices
            decision = QuantizationDecision(
                mode=selected_mode, group_bit_widths=group_bits, metadata={"head": "torch_grouped"}
            )
            self._attach_moe_selection(
                outputs, decision, record, deterministic, moe_context=moe_context
            )
            return decision, record

        if selected_mode == QuantMode.PER_LAYER:
            layer_indices: list[int] = []
            layer_bits: list[int] = []
            for slot in range(self.config.num_layers):
                logits = outputs["layer_logits"][0, slot]
                distribution = Categorical(logits=logits)
                index = (
                    int(torch.argmax(logits).item())
                    if deterministic
                    else int(distribution.sample().item())
                )
                record["log_prob"] += float(
                    distribution.log_prob(torch.tensor(index, device=self.device)).item()
                )
                record["entropy"] += float(distribution.entropy().item())
                layer_indices.append(index)
                layer_bits.append(self.config.discrete_bit_widths[index])
            record["layer_indices"] = layer_indices
            decision = QuantizationDecision(
                mode=selected_mode,
                layer_bit_widths=layer_bits,
                metadata={"head": "torch_per_layer"},
            )
            self._attach_moe_selection(
                outputs, decision, record, deterministic, moe_context=moe_context
            )
            return decision, record

        raw_mean = outputs["learned_mean"][0]
        raw_std = outputs["learned_std"][0]
        distribution = Normal(raw_mean, raw_std)
        raw_sample = raw_mean if deterministic else distribution.sample()
        bounded = self.model.map_learned_parameters(raw_sample.unsqueeze(0))[0]
        record["learned_raw"] = [float(value) for value in raw_sample.detach().cpu().tolist()]
        record["log_prob"] += float(distribution.log_prob(raw_sample).sum().item())
        record["entropy"] += float(distribution.entropy().sum().item())
        decision = QuantizationDecision(
            mode=selected_mode,
            scale_factor=float(bounded[0].item()),
            clipping_range=float(bounded[1].item()),
            precision_level=float(bounded[2].item()),
            metadata={"head": "torch_learned"},
        )
        self._attach_moe_selection(
            outputs, decision, record, deterministic, moe_context=moe_context
        )
        return decision, record

    def _attach_moe_selection(
        self,
        outputs: dict[str, torch.Tensor],
        decision: QuantizationDecision,
        record: dict[str, Any],
        deterministic: bool,
        *,
        moe_context: Any | None,
    ) -> None:
        if moe_context is None or self.model.moe_head is None or "moe_logits" not in outputs:
            return
        moe_indices: list[int] = []
        moe_names: list[str] = []
        for slot in range(self.config.moe_top_k):
            logits = outputs["moe_logits"][0, slot]
            distribution = Categorical(logits=logits)
            index = (
                int(torch.argmax(logits).item())
                if deterministic
                else int(distribution.sample().item())
            )
            record["log_prob"] += float(
                distribution.log_prob(torch.tensor(index, device=self.device)).item()
            )
            record["entropy"] += float(distribution.entropy().item())
            moe_indices.append(index)
            moe_names.append(self.config.moe_variant_names[index])
        record["moe_indices"] = moe_indices
        decision.moe_variant_indices = moe_indices
        decision.moe_variant_names = moe_names
        decision.metadata["moe_head"] = "torch_packed_expert_bank"
        record["moe_active"] = True

    def evaluate_actions(
        self, states: torch.Tensor, records: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.autocast_context():
            outputs = self.model(states)
        batch_size = len(records)
        log_probs = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        entropies = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        mode_codes = torch.tensor(
            [_mode_code(record["selected_mode"]) for record in records],
            dtype=torch.long,
            device=self.device,
        )

        if self.model.mode_head is not None:
            mode_indices = torch.tensor(
                [int(record["mode_index"]) for record in records],
                dtype=torch.long,
                device=self.device,
            )
            mode_log_probs = torch.log_softmax(outputs["mode_logits"].float(), dim=-1)
            mode_probs = torch.softmax(outputs["mode_logits"].float(), dim=-1)
            valid_mask = mode_indices >= 0
            if valid_mask.any():
                batch_indices = torch.arange(batch_size, device=self.device)[valid_mask]
                log_probs[valid_mask] += mode_log_probs[batch_indices, mode_indices[valid_mask]]
                entropies[valid_mask] += -(mode_probs[valid_mask] * mode_log_probs[valid_mask]).sum(
                    dim=-1
                )

        discrete_mask = (mode_codes == _mode_code(QuantMode.DISCRETE.value)) | (
            mode_codes == _mode_code(QuantMode.DYNAMIC.value)
        )
        if discrete_mask.any():
            discrete_indices = torch.tensor(
                [
                    int(record["discrete_index"]) if record["discrete_index"] is not None else 0
                    for record in records
                ],
                dtype=torch.long,
                device=self.device,
            )
            logits = outputs["discrete_logits"].float()
            log_distribution = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)[discrete_mask]
            log_probs[discrete_mask] += log_distribution[
                batch_indices, discrete_indices[discrete_mask]
            ]
            entropies[discrete_mask] += -(
                probabilities[discrete_mask] * log_distribution[discrete_mask]
            ).sum(dim=-1)

        grouped_mask = mode_codes == _mode_code(QuantMode.GROUPED.value)
        if grouped_mask.any():
            group_indices = torch.tensor(
                [
                    record["group_indices"]
                    if record["group_indices"]
                    else [0] * self.config.num_groups
                    for record in records
                ],
                dtype=torch.long,
                device=self.device,
            )
            logits = outputs["group_logits"].float()
            log_distribution = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            selected = (
                torch.gather(log_distribution, dim=-1, index=group_indices.unsqueeze(-1))
                .squeeze(-1)
                .sum(dim=-1)
            )
            entropy = -(probabilities * log_distribution).sum(dim=-1).sum(dim=-1)
            log_probs[grouped_mask] += selected[grouped_mask]
            entropies[grouped_mask] += entropy[grouped_mask]

        layer_mask = mode_codes == _mode_code(QuantMode.PER_LAYER.value)
        if layer_mask.any():
            layer_indices = torch.tensor(
                [
                    record["layer_indices"]
                    if record["layer_indices"]
                    else [0] * self.config.num_layers
                    for record in records
                ],
                dtype=torch.long,
                device=self.device,
            )
            logits = outputs["layer_logits"].float()
            log_distribution = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            selected = (
                torch.gather(log_distribution, dim=-1, index=layer_indices.unsqueeze(-1))
                .squeeze(-1)
                .sum(dim=-1)
            )
            entropy = -(probabilities * log_distribution).sum(dim=-1).sum(dim=-1)
            log_probs[layer_mask] += selected[layer_mask]
            entropies[layer_mask] += entropy[layer_mask]

        learned_mask = mode_codes == _mode_code(QuantMode.LEARNED.value)
        if learned_mask.any():
            fallback_raw = zero_previous_action()
            raw_actions = torch.tensor(
                [
                    record["learned_raw"] if record["learned_raw"] else fallback_raw
                    for record in records
                ],
                dtype=torch.float32,
                device=self.device,
            )
            distribution = Normal(outputs["learned_mean"].float(), outputs["learned_std"].float())
            log_probs[learned_mask] += distribution.log_prob(raw_actions).sum(dim=-1)[learned_mask]
            entropies[learned_mask] += distribution.entropy().sum(dim=-1)[learned_mask]

        if self.model.moe_head is not None and "moe_logits" in outputs:
            moe_active = torch.tensor(
                [bool(record.get("moe_active")) for record in records],
                dtype=torch.bool,
                device=self.device,
            )
            if moe_active.any():
                moe_indices = torch.tensor(
                    [
                        record["moe_indices"]
                        if record["moe_indices"]
                        else [self.config.default_moe_variant_index()] * self.config.moe_top_k
                        for record in records
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = outputs["moe_logits"].float()
                log_distribution = torch.log_softmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                selected = (
                    torch.gather(log_distribution, dim=-1, index=moe_indices.unsqueeze(-1))
                    .squeeze(-1)
                    .sum(dim=-1)
                )
                entropy = -(probabilities * log_distribution).sum(dim=-1).sum(dim=-1)
                active_f = moe_active.float()
                log_probs = log_probs + selected * active_f
                entropies = entropies + entropy * active_f

        return log_probs, entropies, outputs["value"].float()

    def _sample_mode(
        self, outputs: dict[str, torch.Tensor], deterministic: bool
    ) -> tuple[QuantMode, int, float, float]:
        fixed_mode = self.config.resolved_quant_mode()
        if fixed_mode != QuantMode.HYBRID:
            return fixed_mode, -1, 0.0, 0.0

        logits = outputs["mode_logits"][0]
        distribution = Categorical(logits=logits)
        index = (
            int(torch.argmax(logits).item()) if deterministic else int(distribution.sample().item())
        )
        return (
            self.supported_modes[index],
            index,
            float(distribution.log_prob(torch.tensor(index, device=self.device)).item()),
            float(distribution.entropy().item()),
        )

    def snapshot(self):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed in this environment."
            ) from TORCH_IMPORT_ERROR
        return {
            name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()
        }

    def restore(self, snapshot) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is not installed in this environment."
            ) from TORCH_IMPORT_ERROR
        self.model.load_state_dict(snapshot)


def _resolve_autocast_dtype(dtype_name: str):
    if torch is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


def _safe_cuda_arch_list() -> list[str]:
    if torch is None or not hasattr(torch.cuda, "get_arch_list"):
        return []
    try:
        return [str(item) for item in torch.cuda.get_arch_list()]
    except Exception:
        return []


def _cuda_arch_supported(major: int, minor: int, arch_list: list[str]) -> bool:
    if not arch_list:
        return True
    exact = {f"sm_{major}{minor}", f"compute_{major}{minor}"}
    if exact.intersection(arch_list):
        return True
    # PTX for the same major and newer minor can generally JIT down within a generation.
    for item in arch_list:
        if not item.startswith("compute_"):
            continue
        try:
            encoded = item.removeprefix("compute_")
            item_major = int(encoded[0])
            item_minor = int(encoded[1:])
        except (IndexError, ValueError):
            continue
        if item_major == major and item_minor >= minor:
            return True
    return False


def _mode_code(mode: str) -> int:
    order = {
        QuantMode.DISCRETE.value: 0,
        QuantMode.GROUPED.value: 1,
        QuantMode.PER_LAYER.value: 2,
        QuantMode.DYNAMIC.value: 3,
        QuantMode.LEARNED.value: 4,
    }
    return order[mode]
