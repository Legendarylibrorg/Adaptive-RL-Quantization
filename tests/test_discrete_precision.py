from __future__ import annotations

import tempfile
import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.math_utils import discrete_precision_level
from adaptive_quant.policy import UniversalQuantizationPolicy
from adaptive_quant.types import HardwareType


class DiscretePrecisionTests(unittest.TestCase):
    def test_single_bit_width_maps_to_zero(self) -> None:
        self.assertEqual(discrete_precision_level(4, (4,)), 0.0)
        self.assertEqual(discrete_precision_level(8, (8, 8, 8)), 0.0)

    def test_span_normalizes_endpoints(self) -> None:
        self.assertEqual(discrete_precision_level(2, (2, 8)), 0.0)
        self.assertEqual(discrete_precision_level(8, (2, 8)), 1.0)
        self.assertEqual(discrete_precision_level(5, (2, 8)), 0.5)

    def test_policy_discrete_act_does_not_crash_single_width(self) -> None:
        config = FrameworkConfig(
            quant_mode="discrete",
            dynamic_quant=False,
            learned_quant=False,
            discrete_bit_widths=(4,),
            training_episodes=1,
            run_name="single_width_policy",
            stability_probe_count=1,
        )
        env = AdaptiveQuantizationEnv(
            config, log_path=f"{tempfile.gettempdir()}/single_width.jsonl"
        )
        policy = UniversalQuantizationPolicy(config)
        state = env.reset(forced_hardware=HardwareType.GPU, forced_prompt_id="very_complex")
        decision, _trace = policy.act(state, deterministic=True)
        self.assertEqual(decision.base_bit_width, 4)
        self.assertEqual(decision.precision_level, 0.0)


if __name__ == "__main__":
    unittest.main()
