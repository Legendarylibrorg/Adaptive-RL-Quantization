from __future__ import annotations

from typing import Any

from adaptive_quant.torch_policy import torch

if torch is not None:

    class GPUReplayBuffer:
        """Fixed-capacity ring buffer stored entirely on a torch device (VRAM when CUDA)."""

        def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
            self.capacity = capacity
            self.device = device
            self.size = 0
            self.cursor = 0

            self.states = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
            self.records: list[dict[str, Any] | None] = [None] * capacity

        def push(self, state_vector: list[float], reward: float, log_prob: float, value: float, record: dict[str, Any]) -> None:
            idx = self.cursor % self.capacity
            self.states[idx] = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
            self.rewards[idx] = reward
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.records[idx] = record
            self.cursor += 1
            self.size = min(self.size + 1, self.capacity)

        def push_batch(self, records_batch: list[dict[str, Any]]) -> None:
            for rec in records_batch:
                self.push(
                    rec["state_vector"],
                    rec["reward"],
                    rec["log_prob"],
                    rec["value"],
                    rec,
                )

        def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
            indices = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.device)
            return (
                self.states[indices],
                self.rewards[indices],
                self.log_probs[indices],
                self.values[indices],
                [self.records[int(i)] for i in indices.cpu().tolist()],
            )

        def all_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
            n = self.size
            return (
                self.states[:n],
                self.rewards[:n],
                self.log_probs[:n],
                self.values[:n],
                [r for r in self.records[:n] if r is not None],
            )

        def vram_bytes(self) -> int:
            total = 0
            for t in (self.states, self.rewards, self.log_probs, self.values):
                total += t.nelement() * t.element_size()
            return total

else:

    class GPUReplayBuffer:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for GPUReplayBuffer.")
