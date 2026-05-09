from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    alpha_latency: float = 0.020
    beta_throughput: float = 0.060
    gamma_perplexity: float = 0.850
    delta_memory: float = 0.002
    epsilon_instability: float = 1.000
    # Optional: penalize ms per prompt token (default 0 keeps reward aligned with pre-token-efficiency runs).
    eta_token_latency: float = 0.0
    # Hinge on perplexity vs reward_perplexity_reference (0 disables the extra term).
    zeta_perplexity_over_ref: float = 0.0
