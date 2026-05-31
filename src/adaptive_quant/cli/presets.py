"""Shared dense/moe preset selection for experiment CLIs."""

from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.presets.baseline import CONFIG as CONFIG_DENSE
from adaptive_quant.presets.moe import CONFIG_MOE


def select_dense_moe_preset(name: str) -> FrameworkConfig:
    if name == "dense":
        return CONFIG_DENSE
    if name == "moe":
        return CONFIG_MOE
    raise SystemExit(f"Unknown preset: {name!r} (expected 'dense' or 'moe')")


def apply_short_run_episodes(config: FrameworkConfig, episodes: int) -> FrameworkConfig:
    return config.clone(
        training_episodes=episodes,
        evaluation_episodes=max(1, episodes // 4),
        continuous_training=False,
    )


__all__ = ["apply_short_run_episodes", "select_dense_moe_preset"]
