from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.hardware import nvidia_smi_visible
from adaptive_quant.torch_install import (
    DEFAULT_CUDA_INDEX,
    INSTALL_CUDA_TORCH_SCRIPT,
    TORCH_CUDA_INDEX_CU126,
    cuda_torch_install_instructions,
    cuda_torch_pip_command,
    torch_cuda_ready_report,
)
from adaptive_quant.torch_policy import _cuda_arch_supported, resolve_training_device
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

    def test_cuda_arch_support_accepts_4090_sm89(self) -> None:
        self.assertTrue(_cuda_arch_supported(8, 9, ["sm_80", "sm_86", "sm_89"]))

    def test_cuda_arch_support_accepts_matching_ptx(self) -> None:
        self.assertTrue(_cuda_arch_supported(8, 9, ["sm_80", "compute_89"]))

    def test_cuda_arch_support_rejects_pre_ada_wheel_for_4090(self) -> None:
        self.assertFalse(_cuda_arch_supported(8, 9, ["sm_70", "sm_75", "sm_80", "sm_86"]))

    def test_cuda_arch_support_is_permissive_when_torch_cannot_report_arches(self) -> None:
        self.assertTrue(_cuda_arch_supported(8, 9, []))

    def test_cuda_torch_pip_command_uses_cu130_by_default(self) -> None:
        cmd = cuda_torch_pip_command()
        self.assertIn(DEFAULT_CUDA_INDEX, cmd)
        self.assertNotIn("cu128", cmd)

    def test_cuda_torch_pip_command_accepts_legacy_index(self) -> None:
        cmd = cuda_torch_pip_command(index_url=TORCH_CUDA_INDEX_CU126)
        self.assertIn("cu126", cmd)

    def test_nvidia_smi_visible_returns_bool(self) -> None:
        self.assertIsInstance(nvidia_smi_visible(), bool)

    def test_torch_cuda_ready_report_includes_smi_field(self) -> None:
        report = torch_cuda_ready_report()
        self.assertIn("nvidia_smi_visible", report)

    def test_cuda_install_instructions_reference_install_script(self) -> None:
        instructions = cuda_torch_install_instructions()
        self.assertIn(INSTALL_CUDA_TORCH_SCRIPT, instructions)
        self.assertIn("cu130", cuda_torch_pip_command())


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class ResolveTrainingDeviceTests(unittest.TestCase):
    def test_require_cuda_raises_when_cuda_unavailable(self) -> None:
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_training_device("cuda", require_cuda=True)
        message = str(ctx.exception)
        self.assertIn("CUDA is not available", message)
        self.assertIn("install_cuda_torch", message)

    def test_require_cuda_false_falls_back_to_cpu_with_note(self) -> None:
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            device, note = resolve_training_device("cuda", require_cuda=False)
        self.assertEqual(device.type, "cpu")
        self.assertIn("using CPU", note or "")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class TorchTrainerImportTests(unittest.TestCase):
    def test_torch_trainer_class_importable(self) -> None:
        from adaptive_quant.torch_trainer import TorchTrainer

        self.assertTrue(callable(TorchTrainer))


if __name__ == "__main__":
    unittest.main()
