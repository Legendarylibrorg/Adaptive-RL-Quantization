from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.logging_utils import md_table, write_json, write_text_file
from adaptive_quant.pipeline_artifacts import maybe_save_final_checkpoint, write_training_history


class ResearchPipeline:
    """Single orchestrated experiment: train policy → evaluate → benchmark suite → analysis → ``*_summary.json``.

    PyTorch runs optionally run VRAM preflight first; GPU profile metadata is attached when
    ``training_backend="pytorch"``. Artifacts consistently use ``config.run_name`` under
    ``outputs/`` (see field paths on ``FrameworkConfig``).
    """

    def __init__(self, config: FrameworkConfig, requested_profile: str | None = None) -> None:
        self.original_config = config
        self.requested_profile = requested_profile

    def run(self) -> dict[str, object]:
        config, gpu_profile_report = self._resolve_config()
        git_commit = _git_commit()
        trainer = self._build_trainer(config)
        preflight_report = None
        train_summary: dict[str, object] = {}
        eval_summary: dict[str, object] = {}
        recommendation_summary: dict[str, object] | None = None
        vram_report: dict[str, object] | None = None
        history_path: str | None = None
        checkpoint_path: str | None = None
        recommendation_path: str | None = None
        pipeline_error: BaseException | None = None
        try:
            if config.training_backend == "pytorch" and config.torch_preflight:
                from adaptive_quant.torch_preflight import run_torch_preflight

                preflight_report = run_torch_preflight(config, trainer.policy)
                preflight_report["gpu_profile"] = gpu_profile_report
                write_json(f"{config.benchmark_dir}/{config.run_name}_preflight.json", preflight_report)

            train_summary = trainer.train()
            vram_report = self._collect_vram_report(trainer)
            eval_summary = trainer.evaluate()
            recommendation_summary = self._recommend_quantization(config, trainer)
            recommendation_path = config.recommendation_path()
            write_json(recommendation_path, recommendation_summary)
            history_path = write_training_history(config, trainer)
            checkpoint_path = maybe_save_final_checkpoint(config, trainer)
        except BaseException as exc:
            pipeline_error = exc
        finally:
            trainer.close()
        if pipeline_error is not None:
            raise pipeline_error

        from adaptive_quant.benchmark import BenchmarkSuite

        benchmark_summary = BenchmarkSuite(config).run()
        analysis = self._run_analysis(config, history_path)
        report_path = self._write_report(
            config,
            git_commit=git_commit,
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
            "git_commit": git_commit,
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
                total_memory_gb = round(properties.total_memory / (1024 ** 3), 2)
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

    def _run_analysis(self, config: FrameworkConfig, history_path: str | None) -> dict[str, object]:
        from analysis.analyzers import (
            analyze_hardware,
            analyze_inputs,
            analyze_moe_cache,
            analyze_moe_experts,
            analyze_quant,
            analyze_training_dynamics,
        )

        analysis_root = f"{config.analysis_dir}/{config.run_name}"
        analysis: dict[str, object] = {
            "hardware": analyze_hardware(f"{config.log_dir}/{config.run_name}_multi_hw.jsonl", f"{analysis_root}/hardware"),
            "input": analyze_inputs(f"{config.log_dir}/{config.run_name}_dynamic.jsonl", f"{analysis_root}/inputs"),
            "quant_function": analyze_quant(f"{config.log_dir}/{config.run_name}_learned.jsonl", f"{analysis_root}/quant"),
        }
        if config.moe_enabled:
            analysis["moe_experts"] = analyze_moe_experts(f"{config.log_dir}/{config.run_name}.jsonl", f"{analysis_root}/moe_experts")
            analysis["moe_cache"] = analyze_moe_cache(f"{config.log_dir}/{config.run_name}.jsonl", f"{analysis_root}/moe_cache")
        if history_path is not None:
            analysis["training_dynamics"] = analyze_training_dynamics(history_path, f"{analysis_root}/training")
        return analysis

    def _write_report(
        self,
        config: FrameworkConfig,
        *,
        git_commit: str | None,
        train_summary: dict[str, object],
        eval_summary: dict[str, object],
        benchmark_summary: dict[str, object],
        gpu_profile_report: dict[str, object] | None,
        preflight_report: dict[str, object] | None,
        vram_report: dict[str, object] | None,
        analysis: dict[str, object],
        history_path: str | None,
        checkpoint_path: str | None,
        recommendation_summary: dict[str, object] | None,
    ) -> str | None:
        if not config.write_research_report:
            return None
        report_path = config.report_path()
        target = Path(report_path)
        report_dir = target.parent

        def _md_code_json(obj: object) -> str:
            return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```"

        def _fmt_num(value: object, *, digits: int = 2) -> str:
            if isinstance(value, bool) or value is None:
                return str(value)
            if isinstance(value, (int, float)):
                return f"{value:.{digits}f}"
            return str(value)

        def _maybe_link(rel_path: Path) -> str:
            # Link paths should be relative to the report file.
            abs_path = report_dir / rel_path
            if abs_path.exists():
                return f"[{rel_path.name}]({rel_path.as_posix()})"
            return f"`{rel_path.as_posix()}` (missing)"

        def _analysis_fig_links() -> list[str]:
            # Prefer relative links from the report location.
            analysis_root = Path(config.analysis_dir) / config.run_name
            rel_root = Path("..") / "analysis" / config.run_name
            candidates = [
                ("hardware reward", analysis_root / "hardware" / "hardware_generalization_reward.svg"),
                ("hardware latency", analysis_root / "hardware" / "hardware_generalization_latency.svg"),
                ("input complexity vs bits", analysis_root / "inputs" / "input_complexity_vs_bits.svg"),
                ("input adaptation scatter", analysis_root / "inputs" / "input_adaptation_scatter.svg"),
                ("quant function params", analysis_root / "quant" / "quant_function_parameters.svg"),
                ("training reward curve", analysis_root / "training" / "training_reward_curve.svg"),
            ]
            lines: list[str] = []
            for label, abs_path in candidates:
                rel_path = rel_root / abs_path.relative_to(analysis_root)
                lines.append(f"- {label}: {_maybe_link(rel_path)}")
            return lines

        eval_metrics = [
            ("mean_reward", eval_summary.get("mean_reward")),
            ("mean_latency_ms", eval_summary.get("mean_latency_ms")),
            ("mean_throughput_tps", eval_summary.get("mean_throughput_tps")),
            ("mean_memory_mb", eval_summary.get("mean_memory_mb")),
            ("mean_perplexity", eval_summary.get("mean_perplexity")),
            ("mean_stability_penalty", eval_summary.get("mean_stability_penalty")),
        ]
        eval_rows = [[k, _fmt_num(v)] for k, v in eval_metrics]
        recommendation = recommendation_summary if isinstance(recommendation_summary, dict) else None
        recommended_quant = recommendation.get("recommended_quant") if isinstance(recommendation, dict) else None

        bench = benchmark_summary
        single_vs_multi = bench.get("single_vs_multi") if isinstance(bench, dict) else None
        static_vs_dynamic = bench.get("static_vs_dynamic") if isinstance(bench, dict) else None
        discrete_vs_learned = bench.get("discrete_vs_learned") if isinstance(bench, dict) else None

        key_results_lines: list[str] = []
        if isinstance(single_vs_multi, dict):
            key_results_lines.append(
                "- universal policy gap improvement: "
                f"**{_fmt_num(single_vs_multi.get('generalization_gap_improvement'))}** "
                "(lower gap is better)"
            )
        if isinstance(static_vs_dynamic, dict):
            evaluation = static_vs_dynamic.get("evaluation", {})
            if isinstance(evaluation, dict):
                s = evaluation.get("static", {})
                d = evaluation.get("dynamic", {})
                if isinstance(s, dict) and isinstance(d, dict):
                    key_results_lines.append(
                        "- static → dynamic reward: "
                        f"{_fmt_num(s.get('mean_reward'))} → {_fmt_num(d.get('mean_reward'))}"
                    )
                    key_results_lines.append(
                        "- static → dynamic stability penalty: "
                        f"{_fmt_num(s.get('mean_stability_penalty'))} → {_fmt_num(d.get('mean_stability_penalty'))}"
                    )
        if isinstance(discrete_vs_learned, dict):
            evaluation = discrete_vs_learned.get("evaluation", {})
            if isinstance(evaluation, dict):
                learned_metrics = evaluation.get("learned", {})
                discrete_metrics = evaluation.get("discrete", {})
                if isinstance(discrete_metrics, dict) and isinstance(learned_metrics, dict):
                    key_results_lines.append(
                        "- discrete → learned reward: "
                        f"{_fmt_num(discrete_metrics.get('mean_reward'))} → {_fmt_num(learned_metrics.get('mean_reward'))}"
                    )
                    key_results_lines.append(
                        "- discrete → learned latency (ms): "
                        f"{_fmt_num(discrete_metrics.get('mean_latency_ms'))} → {_fmt_num(learned_metrics.get('mean_latency_ms'))}"
                    )

        lines = [
            f"# {config.run_name}",
            "",
            "## Overview",
            f"- backend: `{config.training_backend}`",
            f"- training_host_label: `{config.training_host_label or 'unspecified'}`",
            f"- quant_mode: `{config.quant_mode}`",
            f"- moe_enabled: `{config.moe_enabled}`",
            f"- hardware_modes: `{', '.join(config.hardware_modes)}`",
            f"- git_commit: `{git_commit or 'unknown'}`",
            "",
            "## Key results (from this run)",
            *(key_results_lines or ["- (no benchmark summary available)"]),
            "",
            "## Training",
            f"- episodes: `{train_summary.get('episodes')}`",
            f"- mean_reward: `{_fmt_num(train_summary.get('mean_reward'))}`",
            f"- best_reward: `{_fmt_num(train_summary.get('best_reward'))}`",
            f"- final_reward: `{_fmt_num(train_summary.get('final_reward'))}`",
            f"- history: `{history_path or 'not written'}`",
            f"- checkpoint: `{checkpoint_path or 'not written'}`",
            "",
            "## Evaluation",
            *md_table(["metric", "value"], eval_rows),
            "",
            "## Recommendation",
            f"- target_hardware: `{recommendation.get('target_hardware') if recommendation else 'n/a'}`",
            f"- detected_hardware: `{recommendation.get('detected_hardware') if recommendation else 'n/a'}`",
            f"- adaptive_policy_reward: `{_fmt_num((recommendation or {}).get('adaptive_policy', {}).get('mean_reward') if recommendation else None)}`",
            f"- recommended_fixed_quant: `{(recommended_quant or {}).get('signature', 'n/a')}`",
            f"- recommended_fixed_reward: `{_fmt_num((recommended_quant or {}).get('evaluation', {}).get('mean_reward') if isinstance(recommended_quant, dict) else None)}`",
            "",
            "## Benchmarks",
            "",
            "### Single-hardware vs multi-hardware",
            _md_code_json(single_vs_multi) if single_vs_multi is not None else "_not run_",
            "",
            "### Static vs dynamic",
            _md_code_json(static_vs_dynamic) if static_vs_dynamic is not None else "_not run_",
            "",
            "### Discrete vs learned",
            _md_code_json(discrete_vs_learned) if discrete_vs_learned is not None else "_not run_",
            "",
            "## GPU / VRAM / Preflight",
            f"- gpu profile: `{gpu_profile_report or 'n/a'}`",
            f"- preflight: `{preflight_report or 'n/a'}`",
            f"- continuous_training: `{config.continuous_training}`",
            f"- max_training_episodes: `{config.max_training_episodes:,}`",
            f"- replay_buffer_capacity: `{config.replay_buffer_capacity:,}`",
            *(
                [
                    f"- vram_allocated_mb: `{vram_report.get('vram_allocated_mb', 'n/a')}`",
                    f"- vram_reserved_mb: `{vram_report.get('vram_reserved_mb', 'n/a')}`",
                    f"- replay_buffer_mb: `{vram_report.get('replay_buffer_mb', 'n/a')}`",
                    f"- replay_buffer_entries: `{vram_report.get('replay_buffer_entries', 'n/a')}`",
                ]
                if vram_report
                else ["- vram: `n/a (cpu backend)`"]
            ),
            "",
            "## Analysis",
            "### Figures",
            *_analysis_fig_links(),
            "",
            "### Analysis summaries",
            _md_code_json(analysis),
        ]
        write_text_file(report_path, "\n".join(lines) + "\n")
        return str(target)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def run_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    requested_profile: str | None = None,
    show_gpu_profile: bool = False,
    show_training_host: bool = False,
    show_target_hardware: bool = False,
    footer_mode: str = "full",
) -> dict[str, object]:
    """Run ``ResearchPipeline`` and print a short human-readable summary (CLI-friendly).

    ``requested_profile`` selects a named GPU preset when ``training_backend="pytorch"`` (see ``gpu_profiles``).
    ``footer_mode``: ``full`` (paths + metrics), ``minimal`` (one line), ``none`` (silent).

    Returns the same dict written to ``{benchmark_dir}/{run_name}_summary.json``.
    """
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


__all__ = ["ResearchPipeline", "run_pipeline_entrypoint"]
