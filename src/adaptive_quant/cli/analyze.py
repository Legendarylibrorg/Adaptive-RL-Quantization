"""CLI shim: post-hoc analysis (`adaptive-rl-quant-analyze` / `python -m analysis`)."""

from __future__ import annotations


def main() -> None:
    from analysis.__main__ import main as analysis_main

    analysis_main()


if __name__ == "__main__":
    main()
