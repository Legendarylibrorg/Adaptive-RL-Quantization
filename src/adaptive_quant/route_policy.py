"""Contextual bandit that learns which route wins per (hardware, domain, complexity) bucket.

The route bandit treats every :class:`adaptive_quant.model_routes.ModelRoute` as an arm and
learns from observed rewards. Context is reduced to a small categorical bucket so the policy
remains interpretable and the per-bucket statistics are exportable as a flat JSON table:

* hardware  — ``gpu`` / ``cpu`` / ``low_resource``
* domain    — prompt domain string (e.g. ``code``, ``qa``); unseen domains map to ``other``
* complexity — three-level binning of ``InputFeatures.complexity_score``: ``low``/``mid``/``high``

Within a bucket we use UCB1 with Welford-tracked mean/variance and an explicit infeasibility
mask so routes that violate hardware affinity hints (e.g. an 8-bit, 8 GB GGUF on a 4 GB low-
resource profile) are excluded from selection rather than fighting it through the reward.
A modest global prior is mixed in for cold buckets so we do not pull the same route on every
new context just because it had one lucky pull.

State is JSON-serializable (``state_dict`` / ``load_state_dict``) so it can live next to the
route catalog and survive process restarts.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from adaptive_quant.features import COMPLEXITY_ROUTE_THRESHOLDS, complexity_bucket
from adaptive_quant.math_utils import argmax
from adaptive_quant.model_routes import ModelRoute, RouteCatalog
from adaptive_quant.types import HardwareType

_COMPLEXITY_BINS = ("low", "mid", "high")
_DOMAIN_OTHER = "other"


@dataclass
class RouteContext:
    """Compact, hashable context for the bandit."""

    hardware: str
    domain: str
    complexity: str

    @classmethod
    def from_features(
        cls,
        *,
        hardware: HardwareType | str,
        domain: str,
        complexity_score: float,
        known_domains: Iterable[str] | None = None,
    ) -> "RouteContext":
        hw = hardware.value if isinstance(hardware, HardwareType) else str(hardware)
        hw = hw.strip().lower() or HardwareType.CPU.value

        domain_normalized = (domain or "").strip().lower() or _DOMAIN_OTHER
        if known_domains is not None:
            allow = {str(d).strip().lower() for d in known_domains if d}
            if allow and domain_normalized not in allow:
                domain_normalized = _DOMAIN_OTHER

        return cls(
            hardware=hw,
            domain=domain_normalized,
            complexity=complexity_bucket(
                float(complexity_score),
                thresholds=COMPLEXITY_ROUTE_THRESHOLDS,
                labels=_COMPLEXITY_BINS,
            ),
        )

    def key(self) -> str:
        return f"{self.hardware}|{self.domain}|{self.complexity}"


@dataclass
class _ArmStats:
    """Welford running statistics for a single (route, bucket) pair."""

    pulls: int = 0
    mean_reward: float = 0.0
    m2: float = 0.0
    last_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        delta = reward - self.mean_reward
        self.mean_reward += delta / self.pulls
        self.m2 += delta * (reward - self.mean_reward)
        self.last_reward = float(reward)

    @property
    def variance(self) -> float:
        if self.pulls < 2:
            return 0.0
        return self.m2 / float(self.pulls - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulls": int(self.pulls),
            "mean_reward": float(self.mean_reward),
            "m2": float(self.m2),
            "last_reward": float(self.last_reward),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_ArmStats":
        return cls(
            pulls=int(payload.get("pulls", 0)),
            mean_reward=float(payload.get("mean_reward", 0.0)),
            m2=float(payload.get("m2", 0.0)),
            last_reward=float(payload.get("last_reward", 0.0)),
        )


@dataclass
class RouteSelection:
    """Result of :meth:`RouteBandit.select` / :meth:`RouteBandit.recommend`."""

    route: ModelRoute
    score: float
    explore: bool
    feasible: bool
    bucket_key: str
    bucket_pulls: int
    global_pulls: int
    confidence: float
    reasoning: str


@dataclass
class RouteBandit:
    """Bucketed contextual UCB bandit over a fixed set of :class:`ModelRoute` arms.

    The bucket → arm stats are stored in two layers:

    * ``_global[arm_id]`` — context-free stats; used as a prior for cold buckets and for
      reporting the run-wide best route.
    * ``_buckets[bucket_key][arm_id]`` — per-context stats; what the policy actually optimizes.

    ``ucb_c`` controls exploration aggressiveness (smaller = greedier). ``prior_weight`` is
    the effective sample count of the global prior added to the bucket-local mean when the
    bucket has been pulled fewer than ``warmup_pulls`` times.
    """

    catalog: RouteCatalog
    ucb_c: float = 1.5
    prior_weight: float = 4.0
    warmup_pulls: int = 3
    seed: int = 13
    known_domains: tuple[str, ...] | None = None
    _global: dict[str, _ArmStats] = field(default_factory=dict, init=False, repr=False)
    _buckets: dict[str, dict[str, _ArmStats]] = field(default_factory=dict, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)
    _total_pulls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ucb_c <= 0:
            raise ValueError("ucb_c must be > 0")
        if self.prior_weight < 0:
            raise ValueError("prior_weight must be >= 0")
        if self.warmup_pulls < 0:
            raise ValueError("warmup_pulls must be >= 0")
        self._rng = random.Random(self.seed)
        for route in self.catalog:
            self._global.setdefault(route.route_id, _ArmStats())

    @property
    def total_pulls(self) -> int:
        return int(self._total_pulls)

    def select(self, context: RouteContext, *, deterministic: bool = False) -> RouteSelection:
        """Pick a route under exploration; ties broken by ``self._rng`` for reproducibility."""
        feasible_routes = [r for r in self.catalog if r.matches_hardware(context.hardware)]
        if not feasible_routes:
            feasible_routes = list(self.catalog)
            feasible = False
        else:
            feasible = True
        if not feasible_routes:
            raise RuntimeError("RouteBandit catalog is empty; nothing to select.")

        bucket_key = context.key()
        bucket = self._buckets.get(bucket_key, {})
        bucket_pulls = sum(stats.pulls for stats in bucket.values())

        scored: list[tuple[float, ModelRoute, _ArmStats, _ArmStats]] = []
        for route in feasible_routes:
            arm = bucket.get(route.route_id, _ArmStats())
            global_arm = self._global.get(route.route_id, _ArmStats())
            score = self._ucb_score(
                arm,
                global_arm=global_arm,
                bucket_pulls=bucket_pulls,
                deterministic=deterministic,
            )
            scored.append((score, route, arm, global_arm))

        # Tie-break randomly when several arms share the top score (reproducible via self._rng).
        scores = [item[0] for item in scored]
        best_value = max(scores)
        ties = [index for index, value in enumerate(scores) if math.isclose(value, best_value)]
        winner = ties[0] if deterministic else ties[self._rng.randrange(len(ties))]
        score, route, arm, global_arm = scored[winner]

        explore = (not deterministic) and (
            arm.pulls < self.warmup_pulls or global_arm.pulls < self.warmup_pulls
        )
        confidence = arm.stddev if arm.pulls >= 2 else global_arm.stddev
        reasoning = self._explain_selection(
            route=route,
            arm=arm,
            global_arm=global_arm,
            bucket_pulls=bucket_pulls,
            score=score,
            deterministic=deterministic,
        )

        return RouteSelection(
            route=route,
            score=float(score),
            explore=bool(explore),
            feasible=feasible,
            bucket_key=bucket_key,
            bucket_pulls=int(arm.pulls),
            global_pulls=int(global_arm.pulls),
            confidence=float(confidence),
            reasoning=reasoning,
        )

    def recommend(self, context: RouteContext) -> RouteSelection:
        """Greedy selection (no exploration bonus). Equivalent to ``select(..., deterministic=True)``."""
        return self.select(context, deterministic=True)

    def update(self, route_id: str, context: RouteContext, reward: float) -> None:
        """Observe a reward for a (context, arm) pull and update statistics in place."""
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")
        if route_id not in self._global:
            # Lazily register newly added catalog routes.
            self._global[route_id] = _ArmStats()
        self._global[route_id].update(float(reward))
        bucket = self._buckets.setdefault(context.key(), {})
        arm = bucket.setdefault(route_id, _ArmStats())
        arm.update(float(reward))
        self._total_pulls += 1

    def best_route(self, context: RouteContext) -> ModelRoute:
        return self.recommend(context).route

    def best_route_global(self) -> ModelRoute:
        means = [
            self._global.get(route.route_id, _ArmStats()).mean_reward for route in self.catalog
        ]
        return self.catalog.routes[argmax(means)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "ucb_c": float(self.ucb_c),
            "prior_weight": float(self.prior_weight),
            "warmup_pulls": int(self.warmup_pulls),
            "seed": int(self.seed),
            "total_pulls": int(self._total_pulls),
            "global": {arm_id: stats.to_dict() for arm_id, stats in self._global.items()},
            "buckets": {
                bucket_key: {arm_id: stats.to_dict() for arm_id, stats in arms.items()}
                for bucket_key, arms in self._buckets.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("Unsupported RouteBandit state_dict version")
        self.ucb_c = float(state.get("ucb_c", self.ucb_c))
        self.prior_weight = float(state.get("prior_weight", self.prior_weight))
        self.warmup_pulls = int(state.get("warmup_pulls", self.warmup_pulls))
        self.seed = int(state.get("seed", self.seed))
        self._rng = random.Random(self.seed)
        self._total_pulls = int(state.get("total_pulls", 0))
        self._global = {
            arm_id: _ArmStats.from_dict(payload)
            for arm_id, payload in dict(state.get("global", {})).items()
        }
        for route in self.catalog:
            self._global.setdefault(route.route_id, _ArmStats())
        self._buckets = {
            bucket_key: {
                arm_id: _ArmStats.from_dict(arm_payload) for arm_id, arm_payload in arms.items()
            }
            for bucket_key, arms in dict(state.get("buckets", {})).items()
        }

    def report(self) -> dict[str, Any]:
        """Human-friendly telemetry: top routes per bucket plus global means."""
        bucket_summary: dict[str, list[dict[str, Any]]] = {}
        for bucket_key, arms in self._buckets.items():
            ordered = sorted(
                (
                    {
                        "route_id": arm_id,
                        "pulls": stats.pulls,
                        "mean_reward": stats.mean_reward,
                        "stddev": stats.stddev,
                    }
                    for arm_id, stats in arms.items()
                ),
                key=lambda row: (-row["mean_reward"], -row["pulls"]),
            )
            bucket_summary[bucket_key] = ordered

        global_summary = sorted(
            (
                {
                    "route_id": arm_id,
                    "pulls": stats.pulls,
                    "mean_reward": stats.mean_reward,
                    "stddev": stats.stddev,
                }
                for arm_id, stats in self._global.items()
            ),
            key=lambda row: (-row["mean_reward"], -row["pulls"]),
        )

        return {
            "total_pulls": self._total_pulls,
            "buckets": bucket_summary,
            "global": global_summary,
        }

    def _ucb_score(
        self,
        arm: _ArmStats,
        *,
        global_arm: _ArmStats,
        bucket_pulls: int,
        deterministic: bool,
    ) -> float:
        prior_mean = global_arm.mean_reward if global_arm.pulls > 0 else 0.0
        prior_strength = self.prior_weight if arm.pulls < max(1, self.warmup_pulls) else 0.0
        denom = max(1, arm.pulls + int(prior_strength))
        blended_mean = (arm.mean_reward * arm.pulls + prior_mean * prior_strength) / denom

        if arm.pulls == 0 and not deterministic:
            # Force at least one pull per arm before exploiting; large bonus dominates ties.
            return blended_mean + 10.0
        if deterministic:
            return blended_mean
        ln_total = math.log(max(1, bucket_pulls + 1))
        bonus = self.ucb_c * math.sqrt(ln_total / max(1, arm.pulls))
        return blended_mean + bonus

    def _explain_selection(
        self,
        *,
        route: ModelRoute,
        arm: _ArmStats,
        global_arm: _ArmStats,
        bucket_pulls: int,
        score: float,
        deterministic: bool,
    ) -> str:
        if arm.pulls == 0:
            return (
                f"cold-start: arm {route.route_id!r} unseen in this bucket; "
                f"global pulls={global_arm.pulls}, score={score:.3f}"
            )
        if deterministic or arm.pulls >= self.warmup_pulls:
            return (
                f"exploit: arm {route.route_id!r} mean={arm.mean_reward:.3f} "
                f"(\u00b1{arm.stddev:.3f}, n={arm.pulls}/{bucket_pulls}); score={score:.3f}"
            )
        return (
            f"explore: arm {route.route_id!r} warming up "
            f"(n={arm.pulls}<warmup={self.warmup_pulls}); score={score:.3f}"
        )


__all__ = [
    "RouteBandit",
    "RouteContext",
    "RouteSelection",
]
