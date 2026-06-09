"""Load SFT policy and frozen reference models for DPO."""

from __future__ import annotations

from typing import Any

from adaptive_quant.llm_alignment.config import DPOSettings


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "DPO alignment requires transformers. Install with: pip install -e '.[alignment]'"
        ) from exc
    return transformers


def _build_quantization_config(settings: DPOSettings) -> Any | None:
    if not settings.use_qlora:
        return None

    _require_transformers()
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(__import__("torch"), settings.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _build_lora_config(settings: DPOSettings) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "LoRA/QLoRA requires peft. Install with: pip install -e '.[alignment]'"
        ) from exc

    return LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def _freeze_model(model: Any) -> Any:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def clone_reference_from_policy(policy_model: Any) -> Any:
    """Clone a causal LM's weights into a frozen reference network.

    Uses ``state_dict`` copy (one in-memory replica) instead of a second
    ``from_pretrained`` disk load. For PEFT-wrapped policies, call on the
    underlying base model or use adapter-disabled reference forwards.
    """
    torch = __import__("torch")
    base = policy_model
    if hasattr(policy_model, "get_base_model"):
        base = policy_model.get_base_model()

    if getattr(base, "hf_device_map", None):
        raise ValueError(
            "Cannot clone a device-mapped quantized model in memory. "
            "Use LoRA/QLoRA with adapter-disabled reference forwards instead."
        )

    reference_model = type(base)(base.config)
    reference_model.load_state_dict(base.state_dict())
    if hasattr(base, "dtype"):
        reference_model = reference_model.to(dtype=base.dtype)
    device = getattr(base, "device", None)
    if device is not None:
        reference_model = reference_model.to(device)
    return _freeze_model(reference_model)


def load_policy_and_reference(
    settings: DPOSettings,
) -> tuple[Any, Any | None, Any, bool]:
    """Load trainable policy and frozen reference for DPO.

    Returns:
        ``(policy_model, reference_model, tokenizer, reference_uses_adapter_disable)``

    When ``reference_uses_adapter_disable`` is ``True`` (LoRA/QLoRA), ``reference_model``
    is ``None`` and the trainer must run reference forwards with
    ``policy_model.disable_adapter()``.
    """
    _require_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not settings.sft_model_path:
        raise ValueError("DPOSettings.sft_model_path must point to an SFT checkpoint.")

    quantization_config = _build_quantization_config(settings)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif settings.bf16:
        model_kwargs["torch_dtype"] = __import__("torch").bfloat16

    tokenizer = AutoTokenizer.from_pretrained(settings.sft_model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.sft_model_path,
        **model_kwargs,
    )

    use_peft = settings.use_lora or settings.use_qlora
    if use_peft:
        from peft import get_peft_model

        policy_model = get_peft_model(base_model, _build_lora_config(settings))
        reference_model = None
        reference_uses_adapter_disable = True
    else:
        policy_model = base_model
        reference_model = clone_reference_from_policy(base_model)
        reference_uses_adapter_disable = False

    if settings.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()

    return policy_model, reference_model, tokenizer, reference_uses_adapter_disable
