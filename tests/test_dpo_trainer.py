"""Tests for DPO trainer helpers and training loop with mock models."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment,misc]

if torch is not None:
    from adaptive_quant.llm_alignment.config import DPOSettings
    from adaptive_quant.llm_alignment.dpo_loss import DPOMetrics
    from adaptive_quant.llm_alignment.dpo_trainer import DPOTrainer, _average_metrics
    from adaptive_quant.llm_alignment.model_loading import clone_reference_from_policy

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int | None = None,
        ) -> dict[str, list[int]]:
            del truncation
            start = 1 if add_special_tokens else 0
            ids = [start + (ord(ch) % 7) for ch in text]
            if max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids}

    class _TinyLM(torch.nn.Module):
        def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"vocab_size": vocab})()
            self.embed = torch.nn.Embedding(vocab, hidden)
            self.proj = torch.nn.Linear(hidden, vocab)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
            del attention_mask, kwargs
            hidden = self.embed(input_ids)
            return type("Out", (), {"logits": self.proj(hidden)})()

    class DpoTrainerTests(unittest.TestCase):
        def test_average_metrics(self) -> None:
            averaged = _average_metrics(
                [
                    DPOMetrics(1.0, 0.1, 0.2, 0.1, 0.1, 0.5),
                    DPOMetrics(3.0, 0.3, 0.4, 0.2, 0.2, 1.0),
                ]
            )
            self.assertEqual(averaged.loss, 2.0)
            self.assertAlmostEqual(averaged.implicit_kl, 0.2)
            self.assertEqual(averaged.accuracy, 0.75)

        def test_clone_reference_from_policy(self) -> None:
            policy = _TinyLM()
            reference = clone_reference_from_policy(policy)
            self.assertTrue(all(not param.requires_grad for param in reference.parameters()))
            for left, right in zip(policy.parameters(), reference.parameters(), strict=True):
                self.assertTrue(torch.allclose(left.detach(), right.detach()))

        def test_trainer_runs_one_optimizer_step(self) -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                tmp_path = Path(temp_dir)
                policy = _TinyLM()
                reference = clone_reference_from_policy(policy)
                tokenizer = _FakeTokenizer()

                settings = DPOSettings(
                    output_dir=str(tmp_path),
                    run_name="unit",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=1,
                    logging_steps=1,
                    save_steps=1000,
                    num_epochs=1,
                )
                trainer = DPOTrainer(
                    settings,
                    policy_model=policy,
                    reference_model=reference,
                    tokenizer=tokenizer,
                )

                examples = [
                    {
                        "prompt": "hi",
                        "chosen": "good",
                        "rejected": "bad",
                    }
                ]
                history = trainer.train(examples, shuffle=False)
                self.assertEqual(trainer.global_step, 1)
                self.assertEqual(len(history), 1)
                self.assertTrue((tmp_path / "unit" / "dpo_training_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
