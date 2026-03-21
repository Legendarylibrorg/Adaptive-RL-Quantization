from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess

from analysis.hardware_generalization import analyze as analyze_hardware
from analysis.input_adaptation import analyze as analyze_inputs
from analysis.quant_function_behavior import analyze as analyze_quant
from analysis.training_dynamics import analyze as analyze_training_dynamics
from adaptive_quant.benchmark import BenchmarkSuite
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.gpu_profiles import apply_gpu_profile
from adaptive_quant.logging_utils import write_json
from adaptive_quant.torch_policy import TORCH_IMPORT_ERROR, torch
from adaptive_quant.torch_preflight import run_torch_preflight
from adaptive_quant.trainer import build_trainer


class ResearchPipeline:
    def __init__(self, config: FrameworkConfig, requested_profile: str | None = None) -> None:
        self.original_config = config
        self.requested_profile = requested_profile

    def run(self) -> dict[str, object]:
        config, gpu_profile_report = self._resolve_config()
        trainer = self._build_trainer(config)
        preflight_report = None
        try:
            if config.training_backend == "pytorch" and config.torch_preflight:
                preflight_report = run_torch_preflight(config, trainer.policy)
                preflight_report["gpu_profile"] = gpu_profile_report
                write_json(f"{config.benchmark_dir}/{config.run_name}_preflight.json", preflight_report)

            train_summary = trainer.train()
            eval_summary = trainer.evaluate()
            history_path = self._write_training_history(config, trainer)
            checkpoint_path = self._maybe_save_final_checkpoint(config, trainer)
        finally:
            trainer.close()

        benchmark_summary = BenchmarkSuite(config).run()
        analysis = self._run_analysis(config, history_path)
        report_path = self._write_report(
            config,
            train_summary=train_summary,
            eval_summary=eval_summary,
            benchmark_summary=benchmark_summary,
            gpu_profile_report=gpu_profile_report,
            preflight_report=preflight_report,
            analysis=analysis,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
        )
        summary = {
            "config": asdict(config),
            "git_commit": _git_commit(),
            "gpu_profile": gpu_profile_report,
            "preflight": preflight_report,
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
        if history_path is not None:
            analysis["training_dynamics"] = analyze_training_dynamics(history_path, f"{analysis_root}/training")
        return analysis

    def _write_report(
        self,
        config: FrameworkConfig,
        *,
        train_summary: dict[str, object],
        eval_summary: dict[str, object],
        benchmark_summary: dict[str, object],
        gpu_profile_report: dict[str, object] | None,
        preflight_report: dict[str, object] | None,
        analysis: dict[str, object],
        history_path: str | None,
        checkpoint_path: str | None,
    ) -> str | None:
        if not config.write_research_report:
            return None
        target = Path(config.report_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {config.run_name}",
            "",
            "## Overview",
            f"- backend: `{config.training_backend}`",
            f"- quant_mode: `{config.quant_mode}`",
            f"- hardware_modes: `{', '.join(config.hardware_modes)}`",
            f"- git_commit: `{_git_commit() or 'unknown'}`",
            "",
            "## Training",
            f"- train summary: `{train_summary}`",
            f"- history: `{history_path or 'not written'}`",
            f"- checkpoint: `{checkpoint_path or 'not written'}`",
            "",
            "## Evaluation",
            f"- evaluation summary: `{eval_summary}`",
            "",
            "## Benchmarks",
            f"- benchmark summary: `{benchmark_summary}`",
            "",
            "## GPU / Preflight",
            f"- gpu profile: `{gpu_profile_report}`",
            f"- preflight: `{preflight_report}`",
            "",
            "## Analysis",
            f"- analysis outputs: `{analysis}`",
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
