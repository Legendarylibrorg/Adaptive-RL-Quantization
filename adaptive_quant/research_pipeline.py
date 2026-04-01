from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.moe_cache_behavior import analyze as analyze_moe_cache
from analysis.moe_expert_behavior import analyze as analyze_moe_experts
from analysis.quant_function_behavior import analyze as analyze_quant
from analysis.training_dynamics import analyze as analyze_training_dynamics
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.gpu_profiles import apply_gpu_profile
from adaptive_quant.logging_utils import write_json
from adaptive_quant.md_utils import md_table
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, torch
from adaptive_quant.torch_preflight import run_torch_preflight
from adaptive_quant.trainer import build_trainer


class ResearchPipeline:
    def __init__(self, config: FrameworkConfig, requested_profile: str | None = None) -> None:
        self.original_config = config
        self.requested_profile = requested_profile

    def run(self) -> dict[str, object]:
        config, gpu_profile_report = self._resolve_config()
        git_commit = _git_commit()
        trainer = self._build_trainer(config)
        preflight_report = None
        try:
            if config.training_backend == "pytorch" and config.torch_preflight:
                preflight_report = run_torch_preflight(config, trainer.policy)
                preflight_report["gpu_profile"] = gpu_profile_report
                write_json(f"{config.benchmark_dir}/{config.run_name}_preflight.json", preflight_report)

            train_summary = trainer.train()
            vram_report = self._collect_vram_report(trainer)
            eval_summary = trainer.evaluate()
            history_path = self._write_training_history(config, trainer)
            checkpoint_path = self._maybe_save_final_checkpoint(config, trainer)
        finally:
            trainer.close()

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
        )
        summary = {
            "config": asdict(config),
            "git_commit": git_commit,
            "gpu_profile": gpu_profile_report,
            "preflight": preflight_report,
            "vram": vram_report,
            "train": train_summary,
            "evaluation": eval_summary,
            "benchmarks": benchmark_summary,
            "analysis": analysis,
            "artifacts": {
                "training_history": history_path,
                "final_checkpoint": checkpoint_path,
                "report": report_path,
            },
        }
        write_json(f"{config.benchmark_dir}/{config.run_name}_summary.json", summary)
        return summary

    def _resolve_config(self) -> tuple[FrameworkConfig, dict[str, object] | None]:
        if self.original_config.training_backend != "pytorch":
            return self.original_config, None
        try:
            requested = self.requested_profile or self.original_config.torch_gpu_profile
            device_name = None
            total_memory_gb = None
            if torch is None:
                raise ImportError(
                    "PyTorch is required for `training_backend=\"pytorch\"`. "
                    "Install a CUDA-enabled PyTorch build before running a PyTorch entrypoint."
                ) from TORCH_IMPORT_ERROR
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
        try:
            return build_trainer(config)
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

    def _write_training_history(self, config: FrameworkConfig, trainer) -> str | None:
        history = getattr(trainer, "training_history", None)
        if not config.write_training_history or history is None:
            return None
        path = config.training_history_path()
        write_json(path, history)
        return path

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

    def _maybe_save_final_checkpoint(self, config: FrameworkConfig, trainer) -> str | None:
        save_checkpoint = getattr(trainer, "save_checkpoint", None)
        if not callable(save_checkpoint):
            return None
        return save_checkpoint(config.final_checkpoint_path())

    def _run_analysis(self, config: FrameworkConfig, history_path: str | None) -> dict[str, object]:
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
    ) -> str | None:
        if not config.write_research_report:
            return None
        target = Path(config.report_path())
        target.parent.mkdir(parents=True, exist_ok=True)
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
                l = evaluation.get("learned", {})
                d = evaluation.get("discrete", {})
                if isinstance(d, dict) and isinstance(l, dict):
                    key_results_lines.append(
                        "- discrete → learned reward: "
                        f"{_fmt_num(d.get('mean_reward'))} → {_fmt_num(l.get('mean_reward'))}"
                    )
                    key_results_lines.append(
                        "- discrete → learned latency (ms): "
                        f"{_fmt_num(d.get('mean_latency_ms'))} → {_fmt_num(l.get('mean_latency_ms'))}"
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
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(target)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


__all__ = ["ResearchPipeline"]
