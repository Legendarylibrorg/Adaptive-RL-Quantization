"""DPO loss, log-probability extraction, and alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Extract summed log-probabilities of completion tokens from causal-LM logits.

    Args:
        logits: Model output logits, shape ``(batch, seq_len, vocab_size)``.
        labels: Token ids with prompt/padding positions set to ``-100``, shape
            ``(batch, seq_len)``. Positions aligned with ``input_ids``; the
            causal shift is applied inside this helper.
        average_log_prob: When ``True``, divide the token sum by the number of
            non-masked completion tokens (per sequence). When ``False`` (DPO
            default), return the raw sum over completion tokens.

    Returns:
        Per-sequence log-probability tensor, shape ``(batch,)``.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits and labels must share batch/seq dims; got {logits.shape=} {labels.shape=}"
        )

    # Causal LM: token at position t is predicted from logits at t-1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask prompt tokens (-100) and padding; only completion tokens contribute.
    loss_mask = shift_labels != -100

    # ``gather`` requires valid indices; dummy 0 is never used because of the mask.
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    per_token_logps = per_token_logps * loss_mask.to(per_token_logps.dtype)

    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        return per_token_logps.sum(dim=-1) / token_counts

    return per_token_logps.sum(dim=-1)


@dataclass
class DPOMetrics:
    """Per-batch alignment diagnostics."""

    loss: float
    implicit_kl: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, DPOMetrics]:
    """Compute the DPO loss and derived monitoring metrics.

    Loss (per Rafailov et al.):
        ``-log_sigmoid(beta * ((logpi_w - logpi_ref_w) - (logpi_l - logpi_ref_l)))``

    where ``w`` / ``l`` are chosen / rejected completions.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    loss = losses.mean()

    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margin = chosen_reward - rejected_reward
        # Average log-ratio gap vs. reference — implicit KL proxy used in DPO monitoring.
        implicit_kl = (
            (policy_chosen_logps - reference_chosen_logps)
            + (policy_rejected_logps - reference_rejected_logps)
        ) / 2.0
        accuracy = (chosen_reward > rejected_reward).float().mean()

    metrics = DPOMetrics(
        loss=float(loss.item()),
        implicit_kl=float(implicit_kl.mean().item()),
        chosen_reward=float(chosen_reward.mean().item()),
        rejected_reward=float(rejected_reward.mean().item()),
        reward_margin=float(reward_margin.mean().item()),
        accuracy=float(accuracy.item()),
    )
    return loss, metrics
