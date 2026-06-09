"""Configuration for DPO alignment runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DPOSettings:
    """Hyperparameters and paths for a DPO training run."""

    # Model paths
    sft_model_path: str = ""
    output_dir: str = "outputs/alignment"

    # DPO objective
    beta: float = 0.1
    average_log_prob: bool = False

    # Optimization
    learning_rate: float = 5e-7
    num_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # Data
    preference_dataset_path: str = ""
    use_chat_template: bool = False

    # Sequence length
    max_prompt_length: int = 512
    max_length: int = 1024

    # Optional PEFT / QLoRA (applied to policy only; reference stays frozen base weights)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    use_qlora: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Runtime
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    run_name: str = "dpo_run"
