from __future__ import annotations

from adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from adaptive_quant.logging_utils import write_json
from adaptive_quant.paper_bundle import create_pipeline_paper_bundle
from adaptive_quant.pipeline.benchmark_warn import warn_if_benchmarks_are_large
from adaptive_quant.pipeline.report_markdown import write_research_report_markdown
from adaptive_quant.pipeline.research_contract import build_research_contract
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.replay_trace import finalize_replay_artifacts
from adaptive_quant.security_audit import build_security_audit_record
from adaptive_quant.security_bypass import enforce_security_bypass_policy


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


def run_research_analysis(config: FrameworkConfig, history_path: str | None) -> dict[str, object]:
    from adaptive_quant.pipeline.output_summary import resolve_analysis_log_path
    from analysis.analyzers import (
        analyze_hardware,
        analyze_inputs,
        analyze_moe_cache,
        analyze_moe_experts,
        analyze_quant,
        analyze_training_dynamics,
    )

    analysis_root = f"{config.analysis_dir}/{config.run_name}"
    primary_log = config.primary_log_path()
    analysis: dict[str, object] = {
        "hardware": analyze_hardware(
            resolve_analysis_log_path(config, "multi_hw"), f"{analysis_root}/hardware"
        ),
        "input": analyze_inputs(
            resolve_analysis_log_path(config, "dynamic"), f"{analysis_root}/inputs"
        ),
        "quant_function": analyze_quant(
            resolve_analysis_log_path(config, "learned"), f"{analysis_root}/quant"
        ),
    }
    if config.moe_enabled:
        analysis["moe_experts"] = analyze_moe_experts(primary_log, f"{analysis_root}/moe_experts")
        analysis["moe_cache"] = analyze_moe_cache(primary_log, f"{analysis_root}/moe_cache")
    if history_path is not None:
        analysis["training_dynamics"] = analyze_training_dynamics(
            history_path, f"{analysis_root}/training"
        )
    return analysis


class ResearchPipeline:
    """Single orchestrated experiment: train policy → evaluate → benchmark suite → analysis → ``*_summary.json``."""

    def __init__(
        self,
        config: FrameworkConfig,
        requested_profile: str | None = None,
        *,
        cli_startup_overrides: dict[str, object] | None = None,
    ) -> None:
        self.original_config = config
        self.requested_profile = requested_profile
        self.cli_startup_overrides = cli_startup_overrides

    def run(self) -> dict[str, object]:
        enforce_security_bypass_policy(context="research pipeline")
        config, gpu_profile_report = self._resolve_config()
        commit = git_commit_hash()
        trainer = None
        preflight_report = None
        train_summary: dict[str, object] = {}
        eval_summary: dict[str, object] = {}
        recommendation_summary: dict[str, object] | None = None
        gguf_export_summary: dict[str, object] = {}
        vram_report: dict[str, object] | None = None
        history_path: str | None = None
        checkpoint_path: str | None = None
        recommendation_path: str | None = None
        replay_report: dict[str, object] | None = None
        pipeline_error: Exception | None = None
        try:
            trainer = self._build_trainer(config)
            if config.training_backend == "pytorch" and config.torch_preflight:
                from adaptive_quant.torch_preflight import run_torch_preflight

                preflight_report = run_torch_preflight(config, trainer.policy)
                preflight_report["gpu_profile"] = gpu_profile_report
                write_json(
                    f"{config.benchmark_dir}/{config.run_name}_preflight.json", preflight_report
                )

            train_summary = trainer.train()
            vram_report = self._collect_vram_report(trainer)
            eval_summary = trainer.evaluate()
            log_path = getattr(trainer.env.logger, "path", None)
            if log_path is not None:
                flush = getattr(trainer.env.logger, "flush", None)
                if callable(flush):
                    flush()
                replay_report = finalize_replay_artifacts(config, log_path, git_commit=commit)
            recommendation_summary = self._recommend_quantization(config, trainer)
            recommendation_path = config.recommendation_path()
            write_json(recommendation_path, recommendation_summary)
            from adaptive_quant.pipeline.gguf_export import maybe_export_gguf

            gguf_export_summary = maybe_export_gguf(config, recommendation_summary)
            history_path = write_training_history(config, trainer)
            checkpoint_path = maybe_save_final_checkpoint(config, trainer)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            pipeline_error = exc
        finally:
            if trainer is not None:
                trainer.close()
        if pipeline_error is not None:
            raise pipeline_error

        from adaptive_quant.benchmark import BenchmarkSuite

        warn_if_benchmarks_are_large(config)
        benchmark_summary = BenchmarkSuite(config).run()
        analysis = run_research_analysis(config, history_path)
        phases = [
            "train",
            "evaluate",
            "recommendation",
            "benchmark",
            "analysis",
            "report",
            "paper_bundle",
        ]
        if config.llama_cpp_gguf_export_enabled:
            phases.insert(3, "gguf_export")
        research = build_research_contract(
            config,
            git_commit=commit,
            pipeline="offline_research",
            phases=phases,
        )
        artifact_payload: dict[str, object] = {
            "training_history": history_path,
            "final_checkpoint": checkpoint_path,
            "recommendation": recommendation_path,
            "report": None,
            "replay_manifest": (replay_report or {}).get("manifest_path"),
        }
        exported_gguf = gguf_export_summary.get("output_path")
        if exported_gguf:
            artifact_payload["exported_gguf"] = exported_gguf
        from adaptive_quant.pipeline.output_summary import (
            build_research_artifact_index,
            slim_analysis_for_summary,
        )

        provisional_index = build_research_artifact_index(config, artifact_payload)
        report_path = write_research_report_markdown(
            config,
            git_commit=commit,
            train_summary=train_summary,
            eval_summary=eval_summary,
            benchmark_summary=benchmark_summary,
            gpu_profile_report=gpu_profile_report,
            preflight_report=preflight_report,
            vram_report=vram_report,
            analysis=analysis,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
            recommendation_summary=recommendation_summary,
            gguf_export_summary=gguf_export_summary,
            research=research,
            artifact_index=provisional_index,
        )
        artifact_payload["report"] = report_path
        summary = {
            "config": config_to_flat_dict(config),
            "git_commit": commit,
            "research": research,
            "security_audit": build_security_audit_record(
                config,
                cli_startup_overrides=self.cli_startup_overrides,
            ),
            "gpu_profile": gpu_profile_report,
            "preflight": preflight_report,
            "vram": vram_report,
            "train": train_summary,
            "evaluation": eval_summary,
            "recommendation": recommendation_summary,
            "gguf_export": gguf_export_summary,
            "benchmarks": benchmark_summary,
            "analysis": analysis,
            "replay": replay_report,
            "artifacts": artifact_payload,
        }
        summary["analysis"] = slim_analysis_for_summary(analysis, config)
        paper_bundle = create_pipeline_paper_bundle(config=config, summary=summary)
        summary["artifacts"]["paper_bundle"] = paper_bundle
        summary["artifact_index"] = build_research_artifact_index(config, summary["artifacts"])
        write_json(config.summary_path(), summary)
        return summary

    def _resolve_config(self) -> tuple[FrameworkConfig, dict[str, object] | None]:
        if self.original_config.training_backend != "pytorch":
            return self.original_config, None
        from adaptive_quant.gpu_profiles import apply_gpu_profile
        from adaptive_quant.hardware import detect_cuda_device
        from adaptive_quant.torch_policy import (
            TORCH_BACKEND_REQUIRED_MESSAGE,
            TORCH_IMPORT_ERROR,
            torch,
        )

        try:
            requested = self.requested_profile or self.original_config.torch_gpu_profile
            if torch is None:
                raise ImportError(TORCH_BACKEND_REQUIRED_MESSAGE) from TORCH_IMPORT_ERROR
            cuda = detect_cuda_device()
            device_name = cuda.name if cuda is not None else None
            total_memory_gb = cuda.total_memory_gb if cuda is not None else None
            return apply_gpu_profile(
                self.original_config,
                requested_profile=requested,
                device_name=device_name,
                total_memory_gb=total_memory_gb,
            )
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

    def _build_trainer(self, config: FrameworkConfig):
        from adaptive_quant.trainer import build_trainer

        try:
            return build_trainer(config)
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        except RuntimeError as exc:
            message = str(exc)
            cuda_related = (
                "CUDA is not available",
                "torch_device=",
                "PyTorch build reports architectures",
                "Install a CUDA-enabled PyTorch wheel",
            )
            if any(marker in message for marker in cuda_related):
                raise SystemExit(message) from exc
            raise

    def _collect_vram_report(self, trainer) -> dict[str, object] | None:
        vram_fn = getattr(trainer, "_vram_stats", None)
        if not callable(vram_fn):
            return None
        stats = vram_fn()
        if not stats:
            return None
        replay = getattr(trainer, "replay_buffer", None)
        if replay is not None:
            stats["replay_buffer_entries"] = float(replay.size)
        return stats

    def _recommend_quantization(self, config: FrameworkConfig, trainer) -> dict[str, object]:
        from adaptive_quant.recommendation import recommend_quantization

        return recommend_quantization(trainer, config)


def run_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    requested_profile: str | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
    show_gpu_profile: bool = False,
    show_training_host: bool = False,
    show_target_hardware: bool = False,
    footer_mode: str = "full",
) -> dict[str, object]:
    from adaptive_quant.run_footer import print_pipeline_footer

    summary = ResearchPipeline(
        config,
        requested_profile=requested_profile,
        cli_startup_overrides=cli_startup_overrides,
    ).run()
    if show_training_host and config.training_host_label:
        print("Training host:", config.training_host_label)
    if show_target_hardware:
        print("Target hardware modes:", ", ".join(config.hardware_modes))
    if show_gpu_profile:
        print("GPU profile:", summary["gpu_profile"])
    print_pipeline_footer(config, summary, mode=footer_mode)
    return summary


__all__ = [
    "ResearchPipeline",
    "maybe_save_final_checkpoint",
    "run_pipeline_entrypoint",
    "run_research_analysis",
    "write_training_history",
]
