from __future__ import annotations

import sys
import warnings

from adaptive_quant.configuration import FrameworkConfig


def warn_if_benchmarks_are_large(config: FrameworkConfig) -> None:
    bench_train = (
        config.training_episodes
        if config.benchmark_training_episodes is None
        else config.benchmark_training_episodes
    )
    bench_eval = (
        config.evaluation_episodes
        if config.benchmark_evaluation_episodes is None
        else config.benchmark_evaluation_episodes
    )
    variant_trains = 6 + (3 if config.moe_enabled else 0)
    estimated_train_episodes = int(bench_train) * int(variant_trains)
    if estimated_train_episodes < 25_000 and int(bench_eval) <= 1_000:
        return
    message = (
        "Benchmark suite may be expensive: it trains multiple variants.\n"
        f"- benchmark_training_episodes={bench_train}\n"
        f"- variants_trained≈{variant_trains}\n"
        f"- estimated_train_episodes≈{estimated_train_episodes:,}\n"
        f"- benchmark_evaluation_episodes={bench_eval}\n"
        "To reduce cost, set benchmark_training_episodes / benchmark_evaluation_episodes in your config."
    )
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    except Exception:
        print(message, file=sys.stderr)
