"""Offline (batch) training helpers for learned task routing during RL post-training."""

from __future__ import annotations

from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.prompts import PromptLibrary, load_prompt_library_json
from adaptive_quant.routing import EfficientTaskRouter, RouterTrace
from adaptive_quant.types import EpisodeResult, EpisodeState, QuantizationDecision, QuantMode


def resolve_prompt_library(
    config: FrameworkConfig, prompt_library: PromptLibrary | None = None
) -> PromptLibrary | None:
    """Return a custom prompt library when ``prompt_library_path`` is configured."""
    if prompt_library is not None:
        return prompt_library
    raw_path = getattr(config, "prompt_library_path", None)
    if not raw_path:
        return None
    return load_prompt_library_json(Path(raw_path))


class OfflineRouterTrainer:
    """Applies router decisions during fixed/continuous offline training episodes."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.router = EfficientTaskRouter(config)
        self._baseline_perplexity: dict[str, float] = {}
        self._pending_trace: RouterTrace | None = None

    @classmethod
    def maybe_create(cls, config: FrameworkConfig) -> OfflineRouterTrainer | None:
        if not config.router_enabled:
            return None
        return cls(config)

    def prepare_decision(
        self, policy_decision: QuantizationDecision, state: EpisodeState
    ) -> QuantizationDecision:
        route, trace = self.router.route(
            task_text=state.prompt.text,
            deterministic=self.config.rl_train_deterministic(),
        )
        self._pending_trace = trace
        bits = route.quant_bits or int(self.config.safe_default_bits)
        metadata = dict(policy_decision.metadata or {})
        metadata.update(
            {
                "head": "router",
                "route": route.key,
                "route_backend": route.backend,
                "policy_head": metadata.get("head", "policy"),
            }
        )
        if route.backend == "llama_cpp":
            path = route.llama_cpp_model_path()
            if path is not None:
                metadata["llama_cpp_model_path"] = path
        elif route.backend == "hf":
            metadata["hf_model"] = route.model_id
        return QuantizationDecision(
            mode=QuantMode.DISCRETE,
            base_bit_width=int(bits),
            metadata=metadata,
        )

    def complete_episode(
        self,
        *,
        state: EpisodeState,
        policy_decision: QuantizationDecision,
        routed_result: EpisodeResult,
        env: object,
        episode_index: int,
    ) -> None:
        trace = self._pending_trace
        self._pending_trace = None
        if trace is None:
            return
        evaluate = getattr(env, "evaluate_current", None)
        if not callable(evaluate):
            return
        baseline_result = evaluate(
            policy_decision,
            episode_index=10_000_000 + int(episode_index),
            log_episode=False,
        )
        baseline_ppl = float(baseline_result.metrics.perplexity)
        self._baseline_perplexity[state.prompt.prompt_id] = baseline_ppl
        router_reward = float(
            self.router.reward_from_metrics(
                memory_mb=float(routed_result.metrics.memory_mb),
                perplexity=float(routed_result.metrics.perplexity),
                baseline_perplexity=baseline_ppl,
                latency_ms=float(routed_result.metrics.latency_ms),
            )
        )
        self.router.update(trace, reward=router_reward)


__all__ = [
    "OfflineRouterTrainer",
    "resolve_prompt_library",
]
