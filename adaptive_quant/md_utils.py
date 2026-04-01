from __future__ import annotations


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


__all__ = ["md_table"]

