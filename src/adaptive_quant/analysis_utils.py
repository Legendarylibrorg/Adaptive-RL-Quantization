from __future__ import annotations

import html
import math
from collections.abc import Mapping
from pathlib import Path

from adaptive_quant.configuration.validation import validate_cli_path_argument
from adaptive_quant.logging_utils import write_text_file
from adaptive_quant.math_utils import mean


def _svg_text(value: object) -> str:
    """Escape user-controllable text before embedding in SVG to neutralize stored XSS sinks."""
    return html.escape(str(value), quote=True)


def ensure_directory(path: str) -> Path:
    validate_cli_path_argument("output_dir", path)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def grouped_mean(
    records: list[dict], group_key: str, metric_path: tuple[str, ...]
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        group = record.get(group_key, "unknown")
        current = record
        for key in metric_path:
            current = current.get(key, {})
        if isinstance(current, (int, float)):
            buckets.setdefault(group, []).append(float(current))
    return {group: mean(values) for group, values in buckets.items()}


def flatten_numeric(
    obj: object,
    *,
    prefix: str = "",
    max_items: int = 20_000,
    skip_bools: bool = True,
) -> dict[str, float]:
    out: dict[str, float] = {}

    def walk(node: object, path: str) -> None:
        if len(out) >= max_items:
            return
        if isinstance(node, bool):
            if skip_bools:
                return
            out[path] = float(node)
            return
        if isinstance(node, (int, float)):
            value = float(node)
            if math.isfinite(value):
                out[path] = value
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                if isinstance(key, str):
                    walk(value, f"{path}.{key}" if path else key)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(obj, prefix)
    return out


def _svg_canvas(
    *,
    title: str,
    bg_color: str,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int = 780,
    height: int = 420,
    margin: int = 60,
) -> tuple[list[str], int, int, int]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{bg_color}" />',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Georgia">{_svg_text(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" stroke-width="2" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" stroke-width="2" />',
    ]
    if y_label:
        parts.insert(
            3,
            f'<text x="20" y="{margin - 20}" font-size="14" font-family="Georgia">{_svg_text(y_label)}</text>',
        )
    if x_label:
        parts.append(
            f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Georgia">{_svg_text(x_label)}</text>'
        )
        if y_label:
            parts.append(
                f'<text x="16" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Georgia" transform="rotate(-90 16 {height / 2})">{_svg_text(y_label)}</text>'
            )
    return parts, width, height, margin


def write_bar_chart(path: str, title: str, values: dict[str, float], y_label: str) -> None:
    parts, width, height, margin = _svg_canvas(title=title, bg_color="#fbf7ef", y_label=y_label)
    chart_height = height - 2 * margin
    chart_width = width - 2 * margin
    labels = list(values.keys())
    maximum = max(values.values()) if values else 1.0
    scale = chart_height / maximum if maximum else 1.0
    bar_width = chart_width / max(1, len(labels) * 1.4)
    gap = bar_width * 0.4

    for index, label in enumerate(labels):
        value = values[label]
        bar_height = value * scale
        x = margin + index * (bar_width + gap) + gap
        y = height - margin - bar_height
        color = ["#2a6f97", "#c9713d", "#5b8c5a", "#8f5d8f"][index % 4]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="6" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin + 20}" text-anchor="middle" font-size="13" font-family="Georgia">{_svg_text(label)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" font-family="Georgia">{value:.2f}</text>'
        )

    parts.append("</svg>")
    write_text_file(path, "\n".join(parts))


def write_scatter_plot(
    path: str, title: str, points: list[tuple[float, float]], x_label: str, y_label: str
) -> None:
    parts, width, height, margin = _svg_canvas(
        title=title, bg_color="#f6fbff", x_label=x_label, y_label=y_label
    )
    x_values = [point[0] for point in points] or [0.0]
    y_values = [point[1] for point in points] or [0.0]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    x_span = max(max_x - min_x, 1e-6)
    y_span = max(max_y - min_y, 1e-6)

    def scale_x(value: float) -> float:
        return margin + ((value - min_x) / x_span) * (width - 2 * margin)

    def scale_y(value: float) -> float:
        return height - margin - ((value - min_y) / y_span) * (height - 2 * margin)

    for point in points:
        x = scale_x(point[0])
        y = scale_y(point[1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#c9713d" opacity="0.85" />')

    parts.append("</svg>")
    write_text_file(path, "\n".join(parts))


__all__ = [
    "ensure_directory",
    "grouped_mean",
    "write_bar_chart",
    "write_scatter_plot",
]
