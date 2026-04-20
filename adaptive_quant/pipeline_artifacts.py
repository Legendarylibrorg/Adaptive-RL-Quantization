from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import write_json


def write_training_history(config: FrameworkConfig, trainer) -> str | None:
    history = getattr(trainer, "training_history", None)
    if not config.write_training_history or history is None:
        return None
    path = config.training_history_path()
    write_json(path, history)
    return path


def maybe_save_final_checkpoint(config: FrameworkConfig, trainer) -> str | None:
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if not callable(save_checkpoint):
        return None
    return save_checkpoint(config.final_checkpoint_path())


__all__ = ["maybe_save_final_checkpoint", "write_training_history"]
