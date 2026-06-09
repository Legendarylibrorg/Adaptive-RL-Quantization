"""DPO training loop with metric tracking and gradient clipping."""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from adaptive_quant.llm_alignment.config import DPOSettings
from adaptive_quant.llm_alignment.data_collator import DPODataCollator
from adaptive_quant.llm_alignment.dpo_loss import DPOMetrics, compute_dpo_loss, get_batch_logps
from adaptive_quant.llm_alignment.model_loading import load_policy_and_reference
from adaptive_quant.logging_utils import JsonlLogger, write_json


def _average_metrics(batch_metrics: list[DPOMetrics]) -> DPOMetrics:
    count = len(batch_metrics)
    if count == 0:
        raise ValueError("Cannot average an empty metrics list.")
    return DPOMetrics(
        loss=sum(item.loss for item in batch_metrics) / count,
        implicit_kl=sum(item.implicit_kl for item in batch_metrics) / count,
        chosen_reward=sum(item.chosen_reward for item in batch_metrics) / count,
        rejected_reward=sum(item.rejected_reward for item in batch_metrics) / count,
        reward_margin=sum(item.reward_margin for item in batch_metrics) / count,
        accuracy=sum(item.accuracy for item in batch_metrics) / count,
    )


class DPOTrainer:
    """Production-oriented DPO trainer for Hugging Face causal LMs.

    Usage::

        settings = DPOSettings(sft_model_path="path/to/sft", beta=0.1)
        trainer = DPOTrainer(settings)
        trainer.train(preference_examples)
    """

    def __init__(
        self,
        settings: DPOSettings,
        *,
        policy_model: Any | None = None,
        reference_model: Any | None = None,
        tokenizer: Any | None = None,
        reference_uses_adapter_disable: bool = False,
    ) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if policy_model is None or tokenizer is None:
            (
                policy_model,
                reference_model,
                tokenizer,
                reference_uses_adapter_disable,
            ) = load_policy_and_reference(settings)

        self.reference_uses_adapter_disable = reference_uses_adapter_disable

        if not getattr(policy_model, "hf_device_map", None):
            policy_model = policy_model.to(self.device)
        if reference_model is not None and not getattr(reference_model, "hf_device_map", None):
            reference_model = reference_model.to(self.device)

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer

        self.collator = DPODataCollator(
            tokenizer=self.tokenizer,
            max_prompt_length=settings.max_prompt_length,
            max_length=settings.max_length,
            use_chat_template=settings.use_chat_template,
        )

        log_dir = Path(settings.output_dir) / settings.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.step_logger = JsonlLogger(log_dir / "dpo_steps.jsonl")

        self.optimizer = self._build_optimizer()
        self.global_step = 0
        self.metrics_history: list[dict[str, float | int]] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.policy_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )

    def _set_seed(self) -> None:
        random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings.seed)

    def _forward_logps(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return get_batch_logps(
            outputs.logits,
            labels,
            average_log_prob=self.settings.average_log_prob,
        )

    def _reference_forward_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.reference_uses_adapter_disable:
                disable_adapter = getattr(self.policy_model, "disable_adapter", None)
                if disable_adapter is None:
                    raise RuntimeError(
                        "reference_uses_adapter_disable=True but policy_model has no "
                        "disable_adapter(); load with LoRA/QLoRA or pass an explicit "
                        "reference_model."
                    )
                with disable_adapter():
                    return self._forward_logps(
                        self.policy_model,
                        input_ids,
                        attention_mask,
                        labels,
                    )
            if self.reference_model is None:
                raise RuntimeError("reference_model is required when not using adapter disable.")
            return self._forward_logps(
                self.reference_model,
                input_ids,
                attention_mask,
                labels,
            )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _training_step(self, batch: dict[str, torch.Tensor]) -> DPOMetrics:
        """Single optimization step on one micro-batch."""
        self.policy_model.train()
        if self.reference_model is not None:
            self.reference_model.eval()

        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        policy_chosen_logps = self._forward_logps(
            self.policy_model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
        )
        policy_rejected_logps = self._forward_logps(
            self.policy_model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
        )

        reference_chosen_logps = self._reference_forward_logps(
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
        )
        reference_rejected_logps = self._reference_forward_logps(
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
        )

        loss, metrics = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.settings.beta,
        )

        scaled_loss = loss / self.settings.gradient_accumulation_steps
        scaled_loss.backward()
        return metrics

    def _maybe_clip_and_step(self) -> None:
        if self.settings.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy_model.parameters() if p.requires_grad],
                self.settings.max_grad_norm,
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _log_metrics(self, metrics: DPOMetrics) -> None:
        record = {
            "step": self.global_step,
            **{key: float(value) for key, value in asdict(metrics).items()},
        }
        self.metrics_history.append(record)
        self.step_logger.log(record)

        if self.global_step % self.settings.logging_steps == 0:
            print(
                f"[DPO step {self.global_step}] "
                f"loss={metrics.loss:.4f} "
                f"kl={metrics.implicit_kl:.4f} "
                f"chosen_r={metrics.chosen_reward:.4f} "
                f"rejected_r={metrics.rejected_reward:.4f} "
                f"margin={metrics.reward_margin:.4f} "
                f"acc={metrics.accuracy:.3f}",
                file=sys.stderr,
            )

    def _save_checkpoint(self) -> None:
        output_dir = Path(self.settings.output_dir) / self.settings.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = output_dir / f"checkpoint-{self.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.policy_model, "save_pretrained"):
            self.policy_model.save_pretrained(ckpt_dir)
        else:
            torch.save(self.policy_model.state_dict(), ckpt_dir / "pytorch_model.bin")

        self.tokenizer.save_pretrained(ckpt_dir)
        write_json(
            ckpt_dir / "dpo_metrics.json",
            {"global_step": self.global_step, "history": self.metrics_history[-50:]},
        )

    def train(
        self,
        dataset: Iterable[dict[str, str]],
        *,
        shuffle: bool = True,
    ) -> list[dict[str, float | int]]:
        """Run DPO training over preference examples with prompt/chosen/rejected fields."""
        self._set_seed()

        examples = list(dataset)
        if not examples:
            raise ValueError("DPO dataset is empty.")

        for key in ("prompt", "chosen", "rejected"):
            if key not in examples[0]:
                raise ValueError(f"Each preference example must include '{key}'.")

        dataloader = DataLoader(
            examples,
            batch_size=self.settings.per_device_train_batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
        )

        steps_per_epoch = math.ceil(len(dataloader) / self.settings.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.settings.num_epochs
        warmup_steps = int(total_steps * self.settings.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)),
        )

        self.optimizer.zero_grad(set_to_none=True)
        accum_metrics: list[DPOMetrics] = []
        micro_steps = 0

        try:
            for _epoch in range(self.settings.num_epochs):
                for batch in dataloader:
                    batch = self._move_batch(batch)
                    metrics = self._training_step(batch)
                    accum_metrics.append(metrics)
                    micro_steps += 1

                    if micro_steps % self.settings.gradient_accumulation_steps == 0:
                        self._maybe_clip_and_step()
                        scheduler.step()
                        self.global_step += 1
                        self._log_metrics(_average_metrics(accum_metrics))
                        accum_metrics.clear()

                        if self.global_step % self.settings.save_steps == 0:
                            self._save_checkpoint()

            if accum_metrics:
                self._maybe_clip_and_step()
                scheduler.step()
                self.global_step += 1
                self._log_metrics(_average_metrics(accum_metrics))
        finally:
            self.step_logger.close()

        self._save_checkpoint()
        summary_path = (
            Path(self.settings.output_dir) / self.settings.run_name / "dpo_training_summary.json"
        )
        write_json(
            summary_path,
            {
                "settings": asdict(self.settings),
                "total_steps": self.global_step,
                "reference_uses_adapter_disable": self.reference_uses_adapter_disable,
                "metrics_history": self.metrics_history,
            },
        )
        return self.metrics_history
