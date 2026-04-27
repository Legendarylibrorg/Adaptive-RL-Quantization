"""Route learning subpackage (model + quantization selection).

This groups the route catalog, contextual bandit, and training/evaluation helpers under a
single namespace without colliding with the existing :mod:`adaptive_quant.routing` module
which is a separate lightweight per-task router.
"""

from __future__ import annotations

from adaptive_quant.routes.bandit import RouteBandit, RouteContext, RouteSelection
from adaptive_quant.routes.catalog import ModelRoute, QuantSpec, RouteCatalog, default_route_catalog
from adaptive_quant.routes.pipeline import (
    RouteTrainingSummary,
    build_route_decision,
    evaluate_route,
    evaluate_route_bandit,
    load_bandit_artifact,
    make_bandit,
    recommend_route,
    save_bandit_artifacts,
    train_route_bandit,
)

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
    "load_bandit_artifact",
    "make_bandit",
    "recommend_route",
    "save_bandit_artifacts",
    "train_route_bandit",
]

