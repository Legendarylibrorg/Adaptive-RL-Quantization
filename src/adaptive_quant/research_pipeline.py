from __future__ import annotations

from dataclasses import asdict

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import write_json
from adaptive_quant.paper_bundle import create_pipeline_paper_bundle
from adaptive_quant.pipeline.analysis_runner import run_research_analysis
from adaptive_quant.pipeline.benchmark_warn import warn_if_benchmarks_are_large
from adaptive_quant.pipeline.report_markdown import write_research_report_markdown
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.pipeline_artifacts import maybe_save_final_checkpoint, write_training_history


class ResearchPipeline:
    """Single orchestrated experiment: train policy → evaluate → benchmark suite → analysis → ``*_summary.json``."""

    def __init__(self, config: FrameworkConfig, requested_profile: str | None = None) -> None:
        self.original_config = config
        self.requested_profile = requested_profile

    def run(self) -> dict[str, object]:
        config, gpu_profile_report = self._resolve_config()
        commit = git_commit_hash()
        trainer = None
        preflight_report = None
        train_summary: dict[str, object] = {}
        eval_summary: dict[str, object] = {}
        recommendation_summary: dict[str, object] | None = None
        vram_report: dict[str, object] | None = None
        history_path: str | None = None
        checkpoint_path: str | None = None
        recommendation_path: str | None = None
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
            recommendation_summary = self._recommend_quantization(config, trainer)
            recommendation_path = config.recommendation_path()
            write_json(recommendation_path, recommendation_summary)
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
        )
        summary = {
            "config": asdict(config),
            "git_commit": commit,
            "gpu_profile": gpu_profile_report,
            "preflight": preflight_report,
            "vram": vram_report,
            "train": train_summary,
            "evaluation": eval_summary,
            "recommendation": recommendation_summary,
            "benchmarks": benchmark_summary,
            "analysis": analysis,
            "artifacts": {
                "training_history": history_path,
                "final_checkpoint": checkpoint_path,
                "recommendation": recommendation_path,
                "report": report_path,
            },
        }
        paper_bundle = create_pipeline_paper_bundle(config=config, summary=summary)
        summary["artifacts"]["paper_bundle"] = paper_bundle
        write_json(config.summary_path(), summary)
        return summary

    def _resolve_config(self) -> tuple[FrameworkConfig, dict[str, object] | None]:
        if self.original_config.training_backend != "pytorch":
            return self.original_config, None
        from adaptive_quant.gpu_profiles import apply_gpu_profile
        from adaptive_quant.torch_policy import (
            TORCH_BACKEND_REQUIRED_MESSAGE,
            TORCH_IMPORT_ERROR,
            torch,
        )

        try:
            requested = self.requested_profile or self.original_config.torch_gpu_profile
            device_name = None
            total_memory_gb = None
            if torch is None:
                raise ImportError(TORCH_BACKEND_REQUIRED_MESSAGE) from TORCH_IMPORT_ERROR
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                device_name = properties.name
                total_memory_gb = round(properties.total_memory / (1024**3), 2)
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
    show_gpu_profile: bool = False,
    show_training_host: bool = False,
    show_target_hardware: bool = False,
    footer_mode: str = "full",
) -> dict[str, object]:
    from adaptive_quant.run_footer import print_pipeline_footer

    summary = ResearchPipeline(config, requested_profile=requested_profile).run()
    if show_training_host and config.training_host_label:
        print("Training host:", config.training_host_label)
    if show_target_hardware:
        print("Target hardware modes:", ", ".join(config.hardware_modes))
    if show_gpu_profile:
        print("GPU profile:", summary["gpu_profile"])
    print_pipeline_footer(config, summary, mode=footer_mode)
    return summary


__all__ = ["ResearchPipeline", "git_commit_hash", "run_pipeline_entrypoint"]
