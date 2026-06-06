"""Route learning subpackage (model + quantization selection).

Groups the route catalog, contextual bandit, and training/evaluation helpers under a
single namespace without colliding with :mod:`adaptive_quant.routing` (the separate
lightweight per-task online router).
"""

from __future__ import annotations

from adaptive_quant.model_routes import ModelRoute, QuantSpec, RouteCatalog, default_route_catalog
from adaptive_quant.route_pipeline import (
    RouteTrainingSummary,
    build_route_decision,
    evaluate_route,
    evaluate_route_bandit,
    evaluate_routes_for_prompts,
    load_bandit_artifact,
    make_bandit,
    recommend_route,
    save_bandit_artifacts,
    train_route_bandit,
)
from adaptive_quant.route_policy import RouteBandit, RouteContext, RouteSelection

__all__ = [
    "ModelRoute",
    "QuantSpec",
    "RouteBandit",
    "RouteCatalog",
    "RouteContext",
    "RouteSelection",
    "RouteTrainingSummary",
    "build_route_decision",
    "default_route_catalog",
    "evaluate_route",
    "evaluate_route_bandit",
    "evaluate_routes_for_prompts",
    "load_bandit_artifact",
    "make_bandit",
    "recommend_route",
    "save_bandit_artifacts",
    "train_route_bandit",
]
