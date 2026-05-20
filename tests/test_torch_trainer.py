from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from adaptive_quant.torch_trainer import (
    _checkpoint_meta_path,
    _crossed_episode_milestone,
)


class TorchTrainerHelperTests(unittest.TestCase):
    def test_crossed_episode_milestone_at_interval_boundary(self) -> None:
        self.assertTrue(_crossed_episode_milestone(9, 10, 10))
        self.assertFalse(_crossed_episode_milestone(10, 10, 10))
        self.assertFalse(_crossed_episode_milestone(11, 12, 10))

    def test_crossed_episode_milestone_non_positive_interval(self) -> None:
        self.assertFalse(_crossed_episode_milestone(0, 1, 0))
        self.assertFalse(_crossed_episode_milestone(0, 1, -5))

    def test_checkpoint_meta_path_sidecar(self) -> None:
        for pt in (Path("/tmp/run/checkpoint.pt"), Path("outputs/checkpoints/foo.pt")):
            expected = pt.with_name(f"{pt.stem}.checkpoint.json")
            self.assertEqual(Path(_checkpoint_meta_path(str(pt))), expected)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class TorchTrainerImportTests(unittest.TestCase):
    def test_torch_trainer_class_importable(self) -> None:
        from adaptive_quant.torch_trainer import TorchTrainer

        self.assertTrue(callable(TorchTrainer))


if __name__ == "__main__":
    unittest.main()
