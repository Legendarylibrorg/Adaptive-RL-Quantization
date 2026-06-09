"""Smoke tests for DPO alignment CLI wiring."""

from __future__ import annotations

import unittest
from unittest import mock

from adaptive_quant.cli import alignment


class AlignmentCliTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            alignment.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_build_settings_from_args(self) -> None:
        import argparse

        args = argparse.Namespace(
            sft_model="model/path",
            dataset="data.jsonl",
            output_dir="outputs/alignment",
            run_name="test_run",
            beta=0.2,
            learning_rate=1e-6,
            epochs=2,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_grad_norm=0.5,
            max_prompt_length=256,
            max_length=512,
            logging_steps=5,
            save_steps=50,
            seed=7,
            lora=True,
            qlora=False,
            chat_template=True,
        )
        settings = alignment._build_settings(args)
        self.assertEqual(settings.sft_model_path, "model/path")
        self.assertTrue(settings.use_lora)
        self.assertTrue(settings.use_chat_template)

    @mock.patch("adaptive_quant.cli.alignment.load_preference_dataset")
    def test_main_runs_trainer(self, load_data) -> None:
        dpo_trainer = unittest.mock.MagicMock()
        trainer_cls = dpo_trainer.DPOTrainer
        import sys

        fake_torch = mock.MagicMock()
        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "adaptive_quant.llm_alignment.dpo_trainer": dpo_trainer,
            },
        ):
            load_data.return_value = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
            trainer = trainer_cls.return_value
            trainer.global_step = 3
            trainer.train.return_value = [{"loss": 0.5, "reward_margin": 0.1}]

            alignment.main(
                [
                    "--sft-model",
                    "sft",
                    "--dataset",
                    "prefs.jsonl",
                    "--run-name",
                    "cli_smoke",
                ]
            )

            trainer_cls.assert_called_once()
            trainer.train.assert_called_once()
