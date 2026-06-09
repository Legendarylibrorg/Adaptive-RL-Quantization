"""Tests for research architecture contract helpers."""

from __future__ import annotations

import unittest

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.pipeline.research_contract import (
    EVIDENCE_LOCAL_LLAMA_CPP,
    EVIDENCE_SIMULATOR,
    build_claims_validation,
    build_research_contract,
    infer_evidence_level,
    research_contract_report_lines,
)


class ResearchContractTests(unittest.TestCase):
    def test_simulator_evidence_level(self) -> None:
        cfg = FrameworkConfig(run_name="sim", detect_host_hardware=False)
        self.assertEqual(infer_evidence_level(cfg), EVIDENCE_SIMULATOR)

    def test_llama_cpp_evidence_level(self) -> None:
        cfg = FrameworkConfig(
            run_name="local",
            backend="llama_cpp",
            llama_cpp_binary="/bin/llama-cli",
            llama_cpp_model="/models/a.gguf",
            detect_host_hardware=False,
        )
        self.assertEqual(infer_evidence_level(cfg), EVIDENCE_LOCAL_LLAMA_CPP)

    def test_contract_states_policy_not_llm_weights(self) -> None:
        cfg = FrameworkConfig(run_name="scope", detect_host_hardware=False)
        contract = build_research_contract(cfg, git_commit="abc", pipeline="offline_research")
        learning = contract["learning_target"]
        assert isinstance(learning, dict)
        self.assertEqual(learning["object"], "quantization_policy")
        does_not = learning["does_not_train"]
        assert isinstance(does_not, list)
        self.assertIn("llm_weights", does_not)
        self.assertIn("gguf_quantization_export", does_not)

    def test_simulator_invalidates_hardware_claims(self) -> None:
        cfg = FrameworkConfig(run_name="sim", detect_host_hardware=False)
        contract = build_research_contract(cfg)
        evidence = contract["evidence"]
        assert isinstance(evidence, dict)
        boundary = evidence["claim_boundary"]
        assert isinstance(boundary, dict)
        invalid = boundary["invalid_claims"]
        assert isinstance(invalid, list)
        self.assertIn("real_hardware_latency_claims", invalid)

    def test_claims_validation_includes_valid_and_invalid(self) -> None:
        cfg = FrameworkConfig(run_name="claims", detect_host_hardware=False)
        claims = build_claims_validation(
            config=cfg,
            summary={"evaluation": {"mean_reward": 1.0}},
            metrics={"evaluation.mean_reward": 1.0},
        )
        self.assertEqual(claims["learning_target"], "quantization_policy")
        self.assertTrue(claims["valid_claims"])
        self.assertTrue(claims["invalid_claims"])

    def test_report_lines_include_learning_target(self) -> None:
        cfg = FrameworkConfig(run_name="report", detect_host_hardware=False)
        contract = build_research_contract(cfg)
        lines = research_contract_report_lines(contract)
        joined = "\n".join(lines)
        self.assertIn("quantization_policy", joined)
        self.assertIn("simulator", joined)


if __name__ == "__main__":
    unittest.main()
