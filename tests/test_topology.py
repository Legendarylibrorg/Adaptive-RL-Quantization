"""Tests for pipeline topology helpers."""

from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.research_contract import SCHEMA_VERSION, build_research_contract
from adaptive_quant.pipeline.topology import build_pipeline_topology, infer_simulator_engine


class TopologyTests(unittest.TestCase):
    def test_default_topology_is_python_orchestrator(self) -> None:
        cfg = FrameworkConfig(run_name="topo", detect_host_hardware=False)
        topo = build_pipeline_topology(cfg)
        self.assertEqual(topo["orchestrator"], "python")
        active = topo["active"]
        assert isinstance(active, dict)
        self.assertEqual(active["backend"], "simulator")
        self.assertEqual(active["simulator_engine"], "python")

    def test_research_contract_includes_topology(self) -> None:
        cfg = FrameworkConfig(run_name="contract", detect_host_hardware=False)
        contract = build_research_contract(cfg, pipeline="offline_research")
        self.assertEqual(contract["schema_version"], SCHEMA_VERSION)
        self.assertIn("topology", contract)
        topo = contract["topology"]
        assert isinstance(topo, dict)
        self.assertIn("layers", topo)

    def test_infer_simulator_engine_none_for_llama(self) -> None:
        cfg = FrameworkConfig(
            run_name="llama",
            backend="llama_cpp",
            detect_host_hardware=False,
        )
        self.assertIsNone(infer_simulator_engine(cfg))


if __name__ == "__main__":
    unittest.main()
