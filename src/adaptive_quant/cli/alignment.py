"""CLI: Direct Preference Optimization (DPO) alignment on an SFT checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import replace

from adaptive_quant.llm_alignment.config import DPOSettings
from adaptive_quant.llm_alignment.preference_data import load_preference_dataset


def _build_settings(args: argparse.Namespace) -> DPOSettings:
    return DPOSettings(
        sft_model_path=args.sft_model,
        preference_dataset_path=args.dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        use_lora=args.lora,
        use_qlora=args.qlora,
        use_chat_template=args.chat_template,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Direct Preference Optimization (DPO) on an existing SFT checkpoint. "
            "Requires: pip install -e '.[alignment]'"
        ),
    )
    parser.add_argument("--sft-model", required=True, help="Path or HF id for the SFT checkpoint.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="JSON or JSONL file with prompt/chosen/rejected fields.",
    )
    parser.add_argument("--output-dir", default="outputs/alignment", help="Artifact root.")
    parser.add_argument("--run-name", default="dpo_run", help="Run subdirectory name.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL regularization strength.")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true", help="Attach LoRA adapters to the policy.")
    parser.add_argument("--qlora", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Format prompts with tokenizer.apply_chat_template.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.lora and args.qlora:
        print("Note: --qlora implies LoRA adapters on the policy.", file=sys.stderr)

    settings = _build_settings(args)
    if args.qlora:
        settings = replace(settings, use_lora=True, use_qlora=True)

    from adaptive_quant.llm_alignment.dpo_trainer import DPOTrainer

    examples = load_preference_dataset(settings.preference_dataset_path)
    print(
        f"[DPO] loaded {len(examples)} preference examples from {settings.preference_dataset_path}",
        file=sys.stderr,
    )

    trainer = DPOTrainer(settings)
    history = trainer.train(examples)
    print(
        f"[DPO] finished {trainer.global_step} optimizer steps; "
        f"artifacts under {settings.output_dir}/{settings.run_name}",
        file=sys.stderr,
    )
    if history:
        last = history[-1]
        print(
            f"[DPO] final loss={last.get('loss', 0):.4f} "
            f"margin={last.get('reward_margin', 0):.4f}",
            file=sys.stderr,
        )
