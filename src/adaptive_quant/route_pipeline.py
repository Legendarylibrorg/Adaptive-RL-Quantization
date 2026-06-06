"""Train and evaluate the route bandit using the existing simulator / llama.cpp reward stack.

This pipeline ties together three pieces that already live in the framework:

* :class:`adaptive_quant.environment.AdaptiveQuantizationEnv` — generates ``EpisodeState``
  objects (prompt features, hardware profile, sensitivity).
* :class:`adaptive_quant.backend.SimulatorBackend` /
  :class:`adaptive_quant.backend.LlamaCppBackend` — measures latency / throughput / memory /
  perplexity for a chosen quantization.
* :class:`adaptive_quant.route_policy.RouteBandit` — contextual UCB bandit over routes.

Each *training step* is a single bandit pull: sample a prompt + hardware, ask the bandit for
a route, run the existing backend on a quantization decision built from the route's effective
bits, score it with the framework's reward weights (plus a memory-fit penalty driven by the
route's GGUF file size), and feed the reward back into the bandit. The pipeline keeps the
bandit JSON next to the run's other benchmark artifacts so the learned policy can be reloaded
later for inference-time recommendations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_quant.backend import build_backend
from adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from adaptive_quant.configuration.validation import (
    assert_hf_repo_allowed,
    hf_allow_unlisted_from_env,
)
from adaptive_quant.environment import AdaptiveQuantizationEnv
from adaptive_quant.logging_utils import JsonlLogger, read_json, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.model_routes import ModelRoute, RouteCatalog
from adaptive_quant.paper_bundle import create_pipeline_paper_bundle
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.reward import compute_weighted_reward
from adaptive_quant.route_policy import RouteBandit, RouteContext, RouteSelection
from adaptive_quant.types import (
    EpisodeState,
    HardwareType,
    QuantizationDecision,
    QuantMode,
)


def _median_complexity_prompt(library: PromptLibrary):
    from adaptive_quant.features import extract_input_features

    ordered = sorted(library.prompts, key=lambda p: extract_input_features(p).complexity_score)
    return ordered[len(ordered) // 2]


# When a route's GGUF file would not fit the hardware memory budget, scale a heavy linear
# penalty on the reward so the bandit never converges to an OOM-ing arm. The factor is large
# enough to dominate quality differences but small enough that a near-fit route (within 10%)
# still wins over a much smaller, much worse model.
_MEMORY_OVERFLOW_PENALTY = 6.0


@dataclass
class RouteTrainingSummary:
    """Compact summary returned by :func:`train_route_bandit`."""

    pulls: int
    mean_reward: float
    final_recommendation: dict[str, str | float] | None
    per_route_pulls: dict[str, int] = field(default_factory=dict)
    per_bucket_pulls: dict[str, int] = field(default_factory=dict)
    explore_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulls": self.pulls,
            "mean_reward": self.mean_reward,
            "final_recommendation": self.final_recommendation,
            "per_route_pulls": dict(self.per_route_pulls),
            "per_bucket_pulls": dict(self.per_bucket_pulls),
            "explore_rate": self.explore_rate,
        }


def build_route_decision(route: ModelRoute, config: FrameworkConfig) -> QuantizationDecision:
    """Construct a ``QuantizationDecision`` whose effective bits exactly match the route.

    We bypass :func:`adaptive_quant.quantization.finalize_decision` on purpose: that helper
    snaps to ``config.discrete_bit_widths`` (defaults to ``(2, 4, 8)``), which would erase the
    quality difference between, e.g. ``Q4_K_M`` (~4.83 bpw) and ``Q5_K_M`` (~5.69 bpw). The
    backend only reads ``effective_layer_bits`` so we can feed it a custom layer profile and
    still get the usual latency / throughput / perplexity formulas.
    """
    bits = float(route.effective_bits or config.safe_default_bits)
    metadata: dict[str, Any] = {
        "route_id": route.route_id,
        "repo_id": route.repo_id,
        "quant_label": route.quant_label,
        "average_bits": bits,
        "bit_variance": 0.0,
        "route_origin": "model_routes",
    }
    if route.local_path:
        metadata["llama_cpp_model_path"] = route.local_path

    return QuantizationDecision(
        mode=QuantMode.DISCRETE,
        base_bit_width=int(round(bits)),
        effective_layer_bits=[bits] * config.num_layers,
        metadata=metadata,
    )


def validate_local_route_models(catalog: RouteCatalog) -> None:
    """Ensure every route has a usable local GGUF path before real llama.cpp measurements."""
    missing: list[str] = []
    invalid: list[str] = []
    for route in catalog.routes:
        if not route.local_path:
            missing.append(route.route_id)
            continue
        path = Path(route.local_path)
        if not path.is_file():
            invalid.append(f"{route.route_id}: {path}")
    if missing or invalid:
        details: list[str] = []
        if missing:
            details.append(f"missing local_path for routes: {', '.join(sorted(missing))}")
        if invalid:
            details.append(f"local_path is not a file: {'; '.join(sorted(invalid))}")
        raise FileNotFoundError(
            "llama.cpp route research requires local GGUF files; " + " | ".join(details)
        )


def evaluate_route(
    *,
    env: AdaptiveQuantizationEnv,
    backend,
    state: EpisodeState,
    route: ModelRoute,
    config: FrameworkConfig,
) -> tuple[dict[str, float], float]:
    """Score a (state, route) pair using the existing backend + reward weights.

    Returns ``(metrics, reward)``. Memory budget overflow caused by the GGUF file size adds a
    direct, separate penalty on top of the standard reward so the bandit cannot rationalize
    OOM with throughput.
    """
    decision = build_route_decision(route, config)
    metrics = backend.evaluate(state, decision)

    base_reward = compute_weighted_reward(
        reward_weights=config.reward_weights,
        metrics=metrics,
        stability_penalty=0.0,
        perplexity_reference=config.reward_perplexity_reference,
        include_instability=False,
    )

    if route.size_mb is not None:
        budget = max(1.0, float(state.hardware_profile.memory_budget_mb))
        overflow_ratio = max(0.0, float(route.size_mb) - budget) / budget
        if overflow_ratio > 0.0:
            base_reward -= _MEMORY_OVERFLOW_PENALTY * overflow_ratio

    return metrics, float(base_reward)


def make_bandit(
    catalog: RouteCatalog,
    config: FrameworkConfig,
    *,
    ucb_c: float = 1.5,
    prior_weight: float = 4.0,
    warmup_pulls: int = 3,
    known_domains: tuple[str, ...] | None = None,
) -> RouteBandit:
    """Factory that seeds the bandit RNG from the framework seed for reproducibility."""
    return RouteBandit(
        catalog=catalog,
        ucb_c=ucb_c,
        prior_weight=prior_weight,
        warmup_pulls=warmup_pulls,
        seed=int(config.seed) + 7919,
        known_domains=known_domains,
    )


def train_route_bandit(
    config: FrameworkConfig,
    *,
    catalog: RouteCatalog,
    iterations: int = 512,
    bandit: RouteBandit | None = None,
    telemetry_path: str | None = None,
    prompt_library: PromptLibrary | None = None,
) -> tuple[RouteBandit, RouteTrainingSummary]:
    """Run the bandit training loop and persist a JSONL of every pull.

    The function constructs a fresh ``AdaptiveQuantizationEnv`` (logging disabled — the env's
    own logger is for the existing RL trainer, not for routes) and a backend matching
    ``config.backend``. Each step samples (prompt, hardware) deterministically when
    ``config.env_sampling_mode == "sequential"``; otherwise the env's own RNG decides.
    """
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if not catalog.routes:
        raise ValueError("catalog is empty; register at least one route before training")
    if not hf_allow_unlisted_from_env():
        for route in catalog.routes:
            assert_hf_repo_allowed(route.repo_id, config_allowlist=config.route_hf_allowed_repos)

    library = prompt_library or PromptLibrary()
    known_domains = tuple(sorted({prompt.domain for prompt in library.prompts}))
    if bandit is None:
        bandit = make_bandit(catalog, config, known_domains=known_domains)

    env = AdaptiveQuantizationEnv(config, enable_logging=False, prompt_library=library)
    backend = build_backend(config)

    log_path = telemetry_path or f"{config.log_dir}/{config.run_name}_route_telemetry.jsonl"
    logger = JsonlLogger(
        log_path,
        buffered=bool(config.jsonl_buffered),
        flush_every=int(config.jsonl_flush_every),
    )

    rewards: list[float] = []
    explore_count = 0
    per_route_pulls: Counter[str] = Counter()
    per_bucket_pulls: Counter[str] = Counter()

    try:
        for step in range(iterations):
            state = env.reset(phase="train", episode_index=step)
            context = RouteContext.from_features(
                hardware=state.hardware_profile.hardware_type,
                domain=state.prompt.domain,
                complexity_score=state.input_features.complexity_score,
                known_domains=known_domains,
            )
            selection = bandit.select(context, deterministic=False)
            metrics, reward = evaluate_route(
                env=env,
                backend=backend,
                state=state,
                route=selection.route,
                config=config,
            )
            bandit.update(selection.route.route_id, context, reward)
            rewards.append(reward)
            explore_count += int(selection.explore)
            per_route_pulls[selection.route.route_id] += 1
            per_bucket_pulls[context.key()] += 1

            logger.log(
                {
                    "step": step,
                    "context": {
                        "hardware": context.hardware,
                        "domain": context.domain,
                        "complexity": context.complexity,
                    },
                    "prompt_id": state.prompt.prompt_id,
                    "route_id": selection.route.route_id,
                    "repo_id": selection.route.repo_id,
                    "quant_label": selection.route.quant_label,
                    "effective_bits": selection.route.effective_bits,
                    "metrics": metrics,
                    "reward": reward,
                    "explore": selection.explore,
                    "score": selection.score,
                    "reasoning": selection.reasoning,
                }
            )
    finally:
        env.logger.close()
        logger.close()

    final_recommendation = _final_recommendation(bandit, library)
    summary = RouteTrainingSummary(
        pulls=len(rewards),
        mean_reward=mean(rewards),
        final_recommendation=final_recommendation,
        per_route_pulls=dict(per_route_pulls),
        per_bucket_pulls=dict(per_bucket_pulls),
        explore_rate=(explore_count / max(1, len(rewards))),
    )
    return bandit, summary


def evaluate_route_bandit(
    config: FrameworkConfig,
    *,
    catalog: RouteCatalog,
    bandit: RouteBandit,
    sweeps: int = 1,
    prompt_library: PromptLibrary | None = None,
) -> dict[str, Any]:
    """Sweep every (prompt, hardware) pair under greedy bandit recommendations and aggregate."""
    if sweeps <= 0:
        raise ValueError("sweeps must be > 0")
    library = prompt_library or PromptLibrary()
    known_domains = tuple(sorted({prompt.domain for prompt in library.prompts}))
    env = AdaptiveQuantizationEnv(config, enable_logging=False, prompt_library=library)
    backend = build_backend(config)

    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    try:
        for _ in range(sweeps):
            for prompt in library.prompts:
                for hardware_value in config.hardware_modes:
                    hardware = HardwareType(hardware_value)
                    state = env.reset(forced_prompt_id=prompt.prompt_id, forced_hardware=hardware)
                    context = RouteContext.from_features(
                        hardware=hardware,
                        domain=state.prompt.domain,
                        complexity_score=state.input_features.complexity_score,
                        known_domains=known_domains,
                    )
                    selection = bandit.recommend(context)
                    metrics, reward = evaluate_route(
                        env=env,
                        backend=backend,
                        state=state,
                        route=selection.route,
                        config=config,
                    )
                    rewards.append(reward)
                    rows.append(
                        {
                            "prompt_id": prompt.prompt_id,
                            "domain": prompt.domain,
                            "hardware": hardware.value,
                            "complexity_bucket": context.complexity,
                            "route_id": selection.route.route_id,
                            "repo_id": selection.route.repo_id,
                            "quant_label": selection.route.quant_label,
                            "effective_bits": selection.route.effective_bits,
                            "reward": reward,
                            "latency_ms": metrics["latency_ms"],
                            "throughput_tps": metrics["throughput_tps"],
                            "perplexity": metrics["perplexity"],
                            "memory_mb": metrics["memory_mb"],
                            "score": selection.score,
                            "reasoning": selection.reasoning,
                        }
                    )
    finally:
        env.logger.close()

    return {
        "sweeps": sweeps,
        "samples": len(rows),
        "mean_reward": mean(rewards),
        "rows": rows,
    }


def evaluate_routes_for_prompts(
    config: FrameworkConfig,
    *,
    catalog: RouteCatalog,
    prompt_library: PromptLibrary,
    hardware: tuple[HardwareType, ...] | None = None,
    max_reward_regression: float = 0.05,
    max_perplexity_regression: float | None = 0.02,
) -> dict[str, Any]:
    """Evaluate all routes and choose the lowest-memory route with bounded regression.

    For each prompt/hardware pair, the best reward across all candidate routes is treated as
    the quality reference. A route is eligible when its reward is within
    ``max_reward_regression`` of that reference and, when enabled, its perplexity is within
    ``max_perplexity_regression`` of the best observed perplexity. The recommendation is the
    eligible route with the lowest measured ``memory_mb``.
    """
    if not catalog.routes:
        raise ValueError("catalog is empty; register at least one route before evaluation")
    if max_reward_regression < 0.0:
        raise ValueError("max_reward_regression must be >= 0")
    if max_perplexity_regression is not None and max_perplexity_regression < 0.0:
        raise ValueError("max_perplexity_regression must be >= 0")
    hardware_modes = hardware or tuple(HardwareType(value) for value in config.hardware_modes)
    if not hardware_modes:
        raise ValueError("at least one hardware mode is required")
    if not hf_allow_unlisted_from_env():
        for route in catalog.routes:
            assert_hf_repo_allowed(route.repo_id, config_allowlist=config.route_hf_allowed_repos)

    env = AdaptiveQuantizationEnv(config, enable_logging=False, prompt_library=prompt_library)
    backend = build_backend(config)
    rows: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    try:
        for prompt in prompt_library.prompts:
            for hw in hardware_modes:
                state = env.reset(forced_prompt=prompt, forced_hardware=hw, phase="eval")
                group_rows: list[dict[str, Any]] = []
                for route in catalog.filter(hardware=hw.value):
                    metrics, reward = evaluate_route(
                        env=env,
                        backend=backend,
                        state=state,
                        route=route,
                        config=config,
                    )
                    row = {
                        "prompt_id": prompt.prompt_id,
                        "domain": prompt.domain,
                        "hardware": hw.value,
                        "route_id": route.route_id,
                        "repo_id": route.repo_id,
                        "quant_label": route.quant_label,
                        "effective_bits": route.effective_bits,
                        "reward": reward,
                        "latency_ms": metrics["latency_ms"],
                        "throughput_tps": metrics["throughput_tps"],
                        "perplexity": metrics["perplexity"],
                        "memory_mb": metrics["memory_mb"],
                    }
                    rows.append(row)
                    group_rows.append(row)
                recommendations.append(
                    _select_lowest_memory_without_regression(
                        prompt_id=prompt.prompt_id,
                        domain=prompt.domain,
                        hardware=hw.value,
                        rows=group_rows,
                        max_reward_regression=max_reward_regression,
                        max_perplexity_regression=max_perplexity_regression,
                    )
                )
    finally:
        env.logger.close()

    selected = [rec for rec in recommendations if rec.get("route_id") is not None]
    return {
        "samples": len(rows),
        "prompt_count": len(prompt_library.prompts),
        "hardware": [hw.value for hw in hardware_modes],
        "route_count": len(catalog.routes),
        "max_reward_regression": max_reward_regression,
        "max_perplexity_regression": max_perplexity_regression,
        "mean_selected_memory_mb": mean([float(rec["memory_mb"]) for rec in selected]),
        "recommendations": recommendations,
        "rows": rows,
    }


def recommend_route(
    *,
    config: FrameworkConfig,
    bandit: RouteBandit,
    prompt_id: str | None = None,
    prompt_text: str | None = None,
    domain: str = "online",
    hardware: HardwareType | str = HardwareType.GPU,
) -> RouteSelection:
    """One-shot recommendation: returns the bandit's best route for the given context.

    If neither ``prompt_id`` nor ``prompt_text`` is provided the helper picks the prompt with
    the median complexity from the bundled library — a reasonable default for a "generic"
    request. Prompt features are derived from the prompt text via the same path the env uses
    so the complexity bin matches training.
    """
    from adaptive_quant.configuration.validation import validate_router_task_text
    from adaptive_quant.features import extract_input_features

    library = PromptLibrary()
    known_domains = tuple(sorted({prompt.domain for prompt in library.prompts}))

    if prompt_id is not None:
        prompt = library.by_id(prompt_id)
    elif prompt_text is not None:
        from adaptive_quant.types import PromptSample

        sanitized = validate_router_task_text(prompt_text)
        prompt = PromptSample(
            prompt_id=f"adhoc_{abs(hash(sanitized)) % 1_000_000:06d}",
            text=sanitized,
            domain=str(domain),
        )
    else:
        prompt = _median_complexity_prompt(library)

    features = extract_input_features(prompt)
    context = RouteContext.from_features(
        hardware=hardware,
        domain=prompt.domain,
        complexity_score=features.complexity_score,
        known_domains=known_domains,
    )
    return bandit.recommend(context)


def _select_lowest_memory_without_regression(
    *,
    prompt_id: str,
    domain: str,
    hardware: str,
    rows: list[dict[str, Any]],
    max_reward_regression: float,
    max_perplexity_regression: float | None,
) -> dict[str, Any]:
    if not rows:
        return {
            "prompt_id": prompt_id,
            "domain": domain,
            "hardware": hardware,
            "route_id": None,
            "reason": "no feasible routes for hardware",
        }
    best_reward = max(float(row["reward"]) for row in rows)
    best_perplexity = min(float(row["perplexity"]) for row in rows)
    eligible: list[dict[str, Any]] = []
    for row in rows:
        reward_regression = best_reward - float(row["reward"])
        perplexity_regression = (
            (float(row["perplexity"]) - best_perplexity) / max(best_perplexity, 1e-9)
            if max_perplexity_regression is not None
            else 0.0
        )
        if reward_regression > max_reward_regression:
            continue
        if (
            max_perplexity_regression is not None
            and perplexity_regression > max_perplexity_regression
        ):
            continue
        candidate = dict(row)
        candidate["reward_regression"] = reward_regression
        candidate["perplexity_regression"] = perplexity_regression
        eligible.append(candidate)

    if not eligible:
        best = max(rows, key=lambda row: float(row["reward"]))
        return {
            "prompt_id": prompt_id,
            "domain": domain,
            "hardware": hardware,
            "route_id": best["route_id"],
            "repo_id": best["repo_id"],
            "quant_label": best["quant_label"],
            "effective_bits": best["effective_bits"],
            "reward": best["reward"],
            "memory_mb": best["memory_mb"],
            "perplexity": best["perplexity"],
            "latency_ms": best["latency_ms"],
            "throughput_tps": best["throughput_tps"],
            "reward_regression": 0.0,
            "perplexity_regression": 0.0,
            "reason": "no candidate met regression bounds; selected best reward",
        }

    selected = min(
        eligible,
        key=lambda row: (
            float(row["memory_mb"]),
            -float(row["reward"]),
            float(row["perplexity"]),
            str(row["route_id"]),
        ),
    )
    selected["reason"] = "lowest memory within regression bounds"
    return selected


def save_bandit_artifacts(
    *,
    config: FrameworkConfig,
    catalog: RouteCatalog,
    bandit: RouteBandit,
    training_summary: RouteTrainingSummary | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, str]:
    """Persist bandit state + a benchmark JSON that mirrors recommendation artifacts."""
    bandit_path = f"{config.benchmark_dir}/{config.run_name}_route_bandit.json"
    summary_path = f"{config.benchmark_dir}/{config.run_name}_route_summary.json"

    from adaptive_quant.checkpoint_integrity import attach_dict_integrity

    bandit_payload = attach_dict_integrity(
        {
            "catalog": catalog.to_dict(),
            "bandit": bandit.state_dict(),
        }
    )
    write_json(bandit_path, bandit_payload)

    summary_payload: dict[str, Any] = {
        "run_name": config.run_name,
        "config": config_to_flat_dict(config),
        "git_commit": git_commit_hash(),
        "training_backend": config.training_backend,
        "backend": config.backend,
        "catalog_size": len(catalog),
        "bandit_report": bandit.report(),
        "artifacts": {
            "bandit": bandit_path,
            "summary": summary_path,
            "route_telemetry": f"{config.log_dir}/{config.run_name}_route_telemetry.jsonl",
        },
    }
    if training_summary is not None:
        summary_payload["training"] = training_summary.to_dict()
    if evaluation is not None:
        summary_payload["evaluation"] = evaluation
    paper_bundle = create_pipeline_paper_bundle(
        config=config,
        summary=summary_payload,
        telemetry_path=f"{config.log_dir}/{config.run_name}_route_telemetry.jsonl",
    )
    summary_payload["artifacts"]["paper_bundle"] = paper_bundle
    write_json(summary_path, summary_payload)

    return {
        "bandit": bandit_path,
        "summary": summary_path,
        "paper_bundle": paper_bundle["paper_bundle_dir"],
    }


def load_bandit_artifact(path: str | Path) -> tuple[RouteCatalog, RouteBandit]:
    """Inverse of :func:`save_bandit_artifacts` (bandit JSON only)."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Route bandit artifact not found: {target}")
    from adaptive_quant.checkpoint_integrity import verify_dict_integrity

    payload = read_json(target, label="Route bandit artifact")
    verify_dict_integrity(payload, label="Route bandit artifact")
    catalog = RouteCatalog.from_dict(payload.get("catalog", {}))
    bandit_state = payload.get("bandit") or {}
    bandit = RouteBandit(
        catalog=catalog,
        ucb_c=float(bandit_state.get("ucb_c", 1.5)),
        prior_weight=float(bandit_state.get("prior_weight", 4.0)),
        warmup_pulls=int(bandit_state.get("warmup_pulls", 3)),
        seed=int(bandit_state.get("seed", 13)),
    )
    if bandit_state:
        bandit.load_state_dict(bandit_state)
    return catalog, bandit


def _final_recommendation(
    bandit: RouteBandit, library: PromptLibrary
) -> dict[str, str | float] | None:
    if not bandit.catalog.routes:
        return None
    from adaptive_quant.features import extract_input_features

    pivot = _median_complexity_prompt(library)
    pivot_features = extract_input_features(pivot)
    context = RouteContext.from_features(
        hardware=HardwareType.GPU,
        domain=pivot.domain,
        complexity_score=pivot_features.complexity_score,
        known_domains=tuple(sorted({prompt.domain for prompt in library.prompts})),
    )
    selection = bandit.recommend(context)
    return {
        "context_key": context.key(),
        "route_id": selection.route.route_id,
        "repo_id": selection.route.repo_id,
        "quant_label": selection.route.quant_label,
        "score": float(selection.score),
    }


__all__ = [
    "RouteTrainingSummary",
    "build_route_decision",
    "evaluate_route",
    "evaluate_route_bandit",
    "evaluate_routes_for_prompts",
    "load_bandit_artifact",
    "make_bandit",
    "recommend_route",
    "save_bandit_artifacts",
    "train_route_bandit",
]
