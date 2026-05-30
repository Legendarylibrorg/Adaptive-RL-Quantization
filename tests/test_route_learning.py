from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.huggingface_cli import (
    DownloadResult,
    HuggingFaceCli,
    build_download_command,
    find_huggingface_cli,
    parse_local_path,
    run_download,
)
from adaptive_quant.logging_utils import read_json
from adaptive_quant.model_routes import (
    QUANT_BITS,
    ModelRoute,
    QuantSpec,
    RouteCatalog,
    default_route_catalog,
)
from adaptive_quant.route_pipeline import (
    build_route_decision,
    evaluate_route,
    evaluate_route_bandit,
    load_bandit_artifact,
    make_bandit,
    recommend_route,
    save_bandit_artifacts,
    train_route_bandit,
    validate_local_route_models,
)
from adaptive_quant.route_policy import RouteBandit, RouteContext
from adaptive_quant.types import HardwareType


def _smoke_config(tmpdir: Path, *, run_name: str = "route_test") -> FrameworkConfig:
    return FrameworkConfig(
        run_name=run_name,
        training_episodes=4,
        evaluation_episodes=4,
        stability_probe_count=1,
        outputs_dir=str(tmpdir / "outputs"),
        log_dir=str(tmpdir / "outputs" / "logs"),
        benchmark_dir=str(tmpdir / "outputs" / "benchmarks"),
        analysis_dir=str(tmpdir / "outputs" / "analysis"),
        checkpoint_dir=str(tmpdir / "outputs" / "checkpoints"),
        report_dir=str(tmpdir / "outputs" / "reports"),
        env_sampling_mode="sequential",
        rl_train_policy_mode="deterministic",
        stability_probe_sampling="deterministic",
        detect_host_hardware=False,
        route_hf_allowed_repos=(
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "bartowski/Phi-3.5-mini-instruct-GGUF",
            "bartowski/Llama-3.2-1B-Instruct-GGUF",
        ),
    )


class QuantSpecTests(unittest.TestCase):
    def test_quant_table_covers_common_gguf_labels(self) -> None:
        for label in ("Q2_K", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"):
            self.assertIn(label, QUANT_BITS, f"{label} should be in the built-in quant table")

    def test_quant_spec_from_label_normalizes(self) -> None:
        spec = QuantSpec.from_label("q4_k_m")
        self.assertEqual(spec.label, "Q4_K_M")
        self.assertGreater(spec.effective_bits, 4.0)
        self.assertLess(spec.effective_bits, 5.0)

    def test_quant_spec_unknown_label_raises(self) -> None:
        with self.assertRaises(KeyError):
            QuantSpec.from_label("Q9_NOT_REAL")


class ModelRouteTests(unittest.TestCase):
    def test_route_normalizes_quant_and_hints(self) -> None:
        route = ModelRoute(
            route_id="demo-route",
            repo_id="org/repo",
            quant_label="q4_k_m",
            hardware_hints=("GPU", "cpu"),
        )
        self.assertEqual(route.quant_label, "Q4_K_M")
        self.assertEqual(route.hardware_hints, ("gpu", "cpu"))
        self.assertGreater(route.effective_bits or 0.0, 4.0)

    def test_route_rejects_invalid_repo_id(self) -> None:
        with self.assertRaises(ValueError):
            ModelRoute(route_id="bad-repo", repo_id="no-slash", quant_label="Q4_K_M")

    def test_route_rejects_invalid_route_id(self) -> None:
        with self.assertRaises(ValueError):
            ModelRoute(route_id="../escape", repo_id="org/repo", quant_label="Q4_K_M")

    def test_route_rejects_unknown_hardware_hint(self) -> None:
        with self.assertRaises(ValueError):
            ModelRoute(
                route_id="bad-hint",
                repo_id="org/repo",
                quant_label="Q4_K_M",
                hardware_hints=("toaster",),
            )


class RouteCatalogTests(unittest.TestCase):
    def test_default_catalog_round_trip(self) -> None:
        catalog = default_route_catalog()
        self.assertGreater(len(catalog), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            catalog.save(str(path))
            loaded = RouteCatalog.from_file(path)
            self.assertEqual(len(loaded), len(catalog))
            self.assertEqual(
                {route.route_id for route in loaded.routes},
                {route.route_id for route in catalog.routes},
            )

    def test_catalog_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            RouteCatalog.from_dict(
                {
                    "routes": [
                        {"route_id": "x", "repo_id": "a/b", "quant_label": "Q4_K_M", "junk": True}
                    ]
                }
            )

    def test_catalog_add_replace(self) -> None:
        catalog = RouteCatalog()
        route = ModelRoute(route_id="r1", repo_id="org/repo", quant_label="Q4_K_M")
        catalog.add(route)
        with self.assertRaises(ValueError):
            catalog.add(route)  # duplicate
        catalog.add(
            ModelRoute(route_id="r1", repo_id="org/repo2", quant_label="Q5_K_M"),
            replace_existing=True,
        )
        self.assertEqual(catalog.by_id("r1").repo_id, "org/repo2")

    def test_catalog_filter_hardware(self) -> None:
        catalog = default_route_catalog()
        gpu_only = catalog.filter(hardware="gpu")
        for route in gpu_only:
            self.assertTrue(route.matches_hardware("gpu"))


class RouteBanditTests(unittest.TestCase):
    def test_bandit_pulls_every_arm_before_exploiting(self) -> None:
        catalog = default_route_catalog()
        bandit = RouteBandit(catalog=catalog, ucb_c=1.0, warmup_pulls=1, seed=7)
        context = RouteContext(hardware="gpu", domain="qa", complexity="mid")
        chosen = set()
        for _ in range(len(catalog) * 3):
            selection = bandit.select(context)
            chosen.add(selection.route.route_id)
            bandit.update(selection.route.route_id, context, reward=0.5)
        # At least every GPU-feasible arm should be tried (catalog has gpu/cpu/low_resource entries).
        gpu_ids = {route.route_id for route in catalog if route.matches_hardware("gpu")}
        self.assertTrue(gpu_ids.issubset(chosen))

    def test_bandit_learns_to_prefer_higher_reward(self) -> None:
        catalog = RouteCatalog(
            routes=[
                ModelRoute(route_id="winner", repo_id="org/winner", quant_label="Q4_K_M"),
                ModelRoute(route_id="loser", repo_id="org/loser", quant_label="Q4_K_M"),
            ]
        )
        bandit = RouteBandit(catalog=catalog, ucb_c=0.5, warmup_pulls=1, seed=11)
        context = RouteContext(hardware="gpu", domain="qa", complexity="mid")
        for _ in range(120):
            selection = bandit.select(context)
            reward = 1.0 if selection.route.route_id == "winner" else -1.0
            bandit.update(selection.route.route_id, context, reward)
        recommended = bandit.recommend(context)
        self.assertEqual(recommended.route.route_id, "winner")
        self.assertFalse(recommended.explore)

    def test_bandit_state_round_trip(self) -> None:
        catalog = default_route_catalog()
        bandit = RouteBandit(catalog=catalog, seed=3)
        context = RouteContext(hardware="cpu", domain="code", complexity="high")
        for _ in range(8):
            selection = bandit.select(context)
            bandit.update(selection.route.route_id, context, reward=0.1)
        state = bandit.state_dict()
        clone = RouteBandit(catalog=catalog)
        clone.load_state_dict(state)
        self.assertEqual(clone.total_pulls, bandit.total_pulls)
        self.assertEqual(
            clone.recommend(context).route.route_id,
            bandit.recommend(context).route.route_id,
        )

    def test_route_context_normalizes_unknown_domain(self) -> None:
        ctx = RouteContext.from_features(
            hardware="gpu",
            domain="unknown_domain",
            complexity_score=0.8,
            known_domains=("qa", "code"),
        )
        self.assertEqual(ctx.domain, "other")
        self.assertEqual(ctx.complexity, "high")


class HuggingFaceCliTests(unittest.TestCase):
    def test_build_download_command_modern_dialect(self) -> None:
        cli = HuggingFaceCli(binary="/usr/local/bin/hf", dialect="hf")
        allowed = ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",)
        argv = build_download_command(
            cli,
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            local_dir="/tmp/models",
            allowed_repos=allowed,
        )
        self.assertEqual(argv[0], "/usr/local/bin/hf")
        self.assertIn("download", argv)
        self.assertIn("--local-dir", argv)
        # Repo + filename should be positional, not flag values.
        self.assertEqual(argv[2], "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        self.assertEqual(argv[3], "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    def test_build_download_rejects_dangerous_filename(self) -> None:
        cli = HuggingFaceCli(binary="hf", dialect="hf")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="org/repo", filename="--rm-rf-flag")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="org/repo", filename="../escape.gguf")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="org/repo", revision="../main")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="org/repo", local_dir="../models")

    def test_build_download_rejects_dangerous_repo(self) -> None:
        cli = HuggingFaceCli(binary="hf", dialect="hf")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="not-a-repo")
        with self.assertRaises(ValueError):
            build_download_command(cli, repo_id="org/repo;rm -rf /")

    def test_parse_local_path_extracts_gguf(self) -> None:
        text = "Fetching 1 files...\n/Users/me/cache/llama-Q4_K_M.gguf\nDone.\n"
        parsed = parse_local_path(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(str(parsed).endswith("llama-Q4_K_M.gguf"))

    def test_download_result_ok_property(self) -> None:
        ok = DownloadResult(command=["hf", "download"], returncode=0, stdout="", stderr="")
        bad = DownloadResult(command=["hf", "download"], returncode=1, stdout="", stderr="boom")
        timed = DownloadResult(
            command=["hf", "download"], returncode=0, stdout="", stderr="", timed_out=True
        )
        self.assertTrue(ok.ok)
        self.assertFalse(bad.ok)
        self.assertFalse(timed.ok)

    def test_run_download_reports_timeout(self) -> None:
        cli = HuggingFaceCli(binary="hf", dialect="hf")
        with mock.patch(
            "adaptive_quant.huggingface_cli.subprocess.run",
            side_effect=TimeoutError,
        ):
            # TimeoutError is not subprocess.TimeoutExpired; ensure only the intended timeout
            # exception is converted to a DownloadResult.
            with self.assertRaises(TimeoutError):
                run_download(
                    cli,
                    repo_id="org/repo",
                    timeout_s=1,
                    allowed_repos=("org/repo",),
                )
        with mock.patch(
            "adaptive_quant.huggingface_cli.subprocess.run",
            side_effect=__import__("subprocess").TimeoutExpired(cmd=["hf"], timeout=1),
        ):
            result = run_download(
                cli,
                repo_id="org/repo",
                timeout_s=1,
                allowed_repos=("org/repo",),
            )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.returncode, -1)

    def test_hf_cli_env_override_must_be_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "hf"
            candidate.write_text("#!/bin/sh\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"HF_CLI": str(candidate)}):
                with mock.patch("adaptive_quant.huggingface_cli.shutil.which", return_value=None):
                    self.assertIsNone(find_huggingface_cli())


class RoutePipelineTests(unittest.TestCase):
    def test_build_route_decision_preserves_effective_bits(self) -> None:
        route = ModelRoute(route_id="r", repo_id="org/repo", quant_label="Q5_K_M")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp))
            decision = build_route_decision(route, cfg)
            self.assertEqual(len(decision.effective_layer_bits), cfg.num_layers)
            for bits in decision.effective_layer_bits:
                self.assertAlmostEqual(bits, route.effective_bits, places=4)

    def test_build_route_decision_includes_local_llama_model_path(self) -> None:
        route = ModelRoute(
            route_id="r",
            repo_id="org/repo",
            quant_label="Q4_K_M",
            local_path="/models/r.gguf",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp))
            decision = build_route_decision(route, cfg)
        self.assertEqual(decision.metadata["llama_cpp_model_path"], "/models/r.gguf")

    def test_validate_local_route_models_requires_existing_files(self) -> None:
        catalog = RouteCatalog(
            routes=[ModelRoute(route_id="r", repo_id="org/repo", quant_label="Q4_K_M")]
        )
        with self.assertRaises(FileNotFoundError):
            validate_local_route_models(catalog)

    def test_train_and_save_route_bandit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = _smoke_config(tmpdir, run_name="route_train_smoke")
            catalog = default_route_catalog()
            bandit, summary = train_route_bandit(cfg, catalog=catalog, iterations=24)
            self.assertEqual(summary.pulls, 24)
            self.assertGreater(bandit.total_pulls, 0)
            artifacts = save_bandit_artifacts(
                config=cfg,
                catalog=catalog,
                bandit=bandit,
                training_summary=summary,
                evaluation=None,
            )
            self.assertTrue(Path(artifacts["bandit"]).is_file())
            self.assertTrue(Path(artifacts["summary"]).is_file())

            payload = read_json(Path(artifacts["summary"]), label="Route summary (test)")
            self.assertEqual(payload["catalog_size"], len(catalog))
            self.assertIn("bandit_report", payload)

            reloaded_catalog, reloaded_bandit = load_bandit_artifact(artifacts["bandit"])
            self.assertEqual(len(reloaded_catalog), len(catalog))
            self.assertEqual(reloaded_bandit.total_pulls, bandit.total_pulls)

    def test_evaluate_route_bandit_emits_per_prompt_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp), run_name="route_eval_smoke")
            catalog = default_route_catalog()
            bandit = make_bandit(catalog, cfg)
            # Seed the bandit with some pulls so recommendations are not all cold starts.
            train_route_bandit(cfg, catalog=catalog, iterations=12, bandit=bandit)
            results = evaluate_route_bandit(cfg, catalog=catalog, bandit=bandit, sweeps=1)
            self.assertGreater(results["samples"], 0)
            for row in results["rows"]:
                self.assertIn(row["hardware"], cfg.hardware_modes)
                self.assertIn(row["route_id"], {r.route_id for r in catalog.routes})

    def test_recommend_route_returns_feasible_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp), run_name="route_recommend_smoke")
            catalog = default_route_catalog()
            bandit = make_bandit(catalog, cfg)
            train_route_bandit(cfg, catalog=catalog, iterations=18, bandit=bandit)
            selection = recommend_route(
                config=cfg,
                bandit=bandit,
                prompt_text="Implement a binary search tree in Python.",
                domain="code",
                hardware=HardwareType.GPU,
            )
            self.assertTrue(selection.feasible or len(catalog) == 0)
            self.assertIn(selection.route.route_id, {r.route_id for r in catalog.routes})

    def test_recommend_route_sanitizes_adhoc_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp), run_name="route_recommend_sanitize")
            catalog = default_route_catalog()
            bandit = make_bandit(catalog, cfg)
            selection = recommend_route(
                config=cfg,
                bandit=bandit,
                prompt_text="hello\u200bworld",
                domain="code",
                hardware=HardwareType.GPU,
            )
            self.assertIn(selection.route.route_id, {r.route_id for r in catalog.routes})

    def test_evaluate_route_uses_size_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _smoke_config(Path(tmp), run_name="route_size_penalty")
            small = ModelRoute(
                route_id="tiny",
                repo_id="org/tiny",
                quant_label="Q4_K_M",
                size_mb=400.0,
                hardware_hints=("low_resource",),
            )
            huge = ModelRoute(
                route_id="huge",
                repo_id="org/huge",
                quant_label="Q4_K_M",
                size_mb=64_000.0,
                hardware_hints=("low_resource",),
            )
            catalog = RouteCatalog(routes=[small, huge])
            bandit = make_bandit(catalog, cfg)

            from adaptive_quant.backend import SimulatorBackend
            from adaptive_quant.environment import AdaptiveQuantizationEnv

            env = AdaptiveQuantizationEnv(cfg, enable_logging=False)
            backend = SimulatorBackend(cfg)
            state = env.reset(forced_hardware=HardwareType.LOW_RESOURCE, episode_index=0)
            try:
                _, reward_small = evaluate_route(
                    env=env, backend=backend, state=state, route=small, config=cfg
                )
                _, reward_huge = evaluate_route(
                    env=env, backend=backend, state=state, route=huge, config=cfg
                )
            finally:
                env.logger.close()
            self.assertGreater(reward_small, reward_huge)
            # Bandit reference (unused below, but ensures factory constructs cleanly):
            self.assertEqual(len(bandit.catalog), 2)


class RouteCliTests(unittest.TestCase):
    def test_cli_seed_and_list(self) -> None:
        from io import StringIO

        from adaptive_quant.cli.route_learning import main as route_main

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.json"
            with mock.patch("sys.stdout", new_callable=StringIO) as stdout:
                route_main(["--catalog", str(catalog_path), "seed"])
                seed_output = stdout.getvalue()
            self.assertIn("Wrote default catalog", seed_output)
            self.assertTrue(catalog_path.is_file())

            with mock.patch("sys.stdout", new_callable=StringIO) as stdout:
                route_main(["--catalog", str(catalog_path), "list", "--format", "json"])
                listing = stdout.getvalue()
            payload = json.loads(listing)
            self.assertGreater(len(payload["routes"]), 0)

    def test_cli_register_accepts_local_path(self) -> None:
        from adaptive_quant.cli.route_learning import main as route_main

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            catalog_path = tmpdir / "catalog.json"
            model_path = tmpdir / "model.gguf"
            model_path.write_text("fake", encoding="utf-8")

            route_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "register",
                    "--route-id",
                    "local-q4",
                    "--repo",
                    "local/model",
                    "--quant",
                    "Q4_K_M",
                    "--local-path",
                    str(model_path),
                ]
            )
            catalog = RouteCatalog.from_file(catalog_path)
            self.assertEqual(catalog.by_id("local-q4").local_path, str(model_path))

    def test_pyproject_exposes_route_console_script(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["project"]["scripts"]["adaptive-rl-quant-route"],
            "adaptive_quant.cli.route_learning:main",
        )
        self.assertEqual(payload["tool"]["setuptools"]["py-modules"], ["config"])


if __name__ == "__main__":
    unittest.main()
