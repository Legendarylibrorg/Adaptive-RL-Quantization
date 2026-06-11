"""Tests for DPO data collation (mock tokenizer, no HF models)."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment,misc]

if torch is not None:
    from adaptive_quant.llm_alignment.data_collator import DPODataCollator


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


@unittest.skipIf(torch is None, "torch required")
class DpoCollatorTests(unittest.TestCase):
    def test_collator_masks_prompt_labels(self) -> None:
        collator = DPODataCollator(tokenizer=_FakeTokenizer(), max_prompt_length=8, max_length=16)
        batch = collator(
            [
                {
                    "prompt": "ab",
                    "chosen": "cd",
                    "rejected": "ef",
                }
            ]
        )

        chosen_labels = batch["chosen_labels"][0].tolist()
        rejected_labels = batch["rejected_labels"][0].tolist()
        self.assertIn(-100, chosen_labels)
        self.assertIn(-100, rejected_labels)
        self.assertTrue(any(label != -100 for label in chosen_labels))
        self.assertEqual(batch["chosen_attention_mask"].shape, batch["chosen_input_ids"].shape)

    def test_collator_raises_without_pad_token(self) -> None:
        class _NoPadTokenizer(_FakeTokenizer):
            pad_token_id = None
            eos_token_id = None

        collator = DPODataCollator(tokenizer=_NoPadTokenizer())
        with self.assertRaisesRegex(ValueError, "pad_token"):
            collator([{"prompt": "x", "chosen": "y", "rejected": "z"}])


if __name__ == "__main__":
    unittest.main()
