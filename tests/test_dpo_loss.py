"""Unit tests for DPO log-probability extraction and loss (no HF models required)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adaptive_quant.llm_alignment.dpo_loss import compute_dpo_loss, get_batch_logps


def test_get_batch_logps_sums_completion_tokens_only() -> None:
    batch, seq_len, vocab = 2, 5, 8
    logits = torch.zeros(batch, seq_len, vocab)

    # Force deterministic log-probs: high logit on token id == position index.
    for b in range(batch):
        for t in range(seq_len):
            logits[b, t, t % vocab] = 10.0

    labels = torch.tensor(
        [
            [-100, 1, 2, 3, -100],
            [-100, -100, 4, 5, 6],
        ]
    )

    logps = get_batch_logps(logits, labels, average_log_prob=False)

    # Causal shift: completion tokens are labels[:, 1:] where label != -100.
    expected_0 = torch.log_softmax(logits[0, :-1], dim=-1)[torch.arange(4), labels[0, 1:]]
    expected_0 = expected_0[labels[0, 1:] != -100].sum()
    expected_1 = torch.log_softmax(logits[1, :-1], dim=-1)[torch.arange(4), labels[1, 1:]]
    expected_1 = expected_1[labels[1, 1:] != -100].sum()

    assert torch.allclose(logps[0], expected_0, atol=1e-5)
    assert torch.allclose(logps[1], expected_1, atol=1e-5)


def test_get_batch_logps_average_mode() -> None:
    batch, seq_len, vocab = 1, 4, 6
    logits = torch.randn(batch, seq_len, vocab)
    labels = torch.tensor([[-100, 2, 3, 4]])

    total = get_batch_logps(logits, labels, average_log_prob=False)
    avg = get_batch_logps(logits, labels, average_log_prob=True)
    assert torch.allclose(avg, total / 3.0, atol=1e-5)


def test_compute_dpo_loss_prefers_higher_chosen_margin() -> None:
    policy_chosen = torch.tensor([0.0, 1.0])
    policy_rejected = torch.tensor([-1.0, 0.0])
    reference_chosen = torch.tensor([0.0, 0.0])
    reference_rejected = torch.tensor([0.0, 0.0])

    loss, metrics = compute_dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )

    assert loss.item() > 0.0
    assert metrics.chosen_reward > metrics.rejected_reward
    assert metrics.reward_margin > 0.0
    assert metrics.accuracy == 1.0
