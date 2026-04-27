#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from _common import repo_root

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_quant.logging_utils import read_json, write_text_file  # noqa: E402


@dataclass(frozen=True)
class RequirementPin:
    line: int
    value: str

def load_dependency_hashes(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = read_json(path, label="Dependency hash manifest")
    raw = payload.get("requirements", {})
    if not isinstance(raw, dict):
        raise ValueError("dependency hash file must contain a 'requirements' object")

    manifest: dict[str, dict[str, list[str]]] = {}
    for requirement_path, entries in raw.items():
        if not isinstance(requirement_path, str) or not isinstance(entries, dict):
            raise ValueError("requirement manifests must map file paths to requirement/hash entries")
        normalized_entries: dict[str, list[str]] = {}
        for requirement, hashes in entries.items():
            if not isinstance(requirement, str) or not isinstance(hashes, list):
                raise ValueError("requirement hash entries must map exact pins to hash lists")
            normalized_hashes: list[str] = []
            for value in hashes:
                if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                    raise ValueError(f"Invalid dependency hash for {requirement!r}: {value!r}")
                normalized_hashes.append(value)
            if not normalized_hashes:
                raise ValueError(f"Requirement {requirement!r} must have at least one sha256 hash")
            normalized_entries[requirement] = normalized_hashes
        manifest[requirement_path] = normalized_entries
    return manifest


def parse_requirement_pins(path: Path) -> list[RequirementPin]:
    pins: list[RequirementPin] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-", "--")):
            raise ValueError(f"{path}:{line_number}: options are not supported in hash-verified requirements")
        if "==" not in line or any(token in line for token in ("[", "]", ";", "@", ">", "<", "~=")):
            raise ValueError(
                f"{path}:{line_number}: only exact 'package==version' pins are supported, got {line!r}"
            )
        name, version = line.split("==", 1)
        if not name or not version or " " in name or " " in version:
            raise ValueError(f"{path}:{line_number}: invalid requirement pin {line!r}")
        pins.append(RequirementPin(line=line_number, value=f"{name}=={version}"))
    return pins


def render_hashed_requirements(
    root: Path,
    *,
    requirement_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[list[str], list[str], Path]:
    requirement_path = requirement_path or (root / "requirements" / "ci.txt")
    manifest_path = manifest_path or (root / "security" / "dependency_hashes.json")

    manifest = load_dependency_hashes(manifest_path)
    relative_requirement = requirement_path.relative_to(root).as_posix()
    entries = manifest.get(relative_requirement)
    if entries is None:
        raise ValueError(f"{manifest_path.relative_to(root)} does not define hashes for {relative_requirement}")

    pins = parse_requirement_pins(requirement_path)
    rendered: list[str] = []
    errors: list[str] = []
    seen = set()
    for pin in pins:
        seen.add(pin.value)
        hashes = entries.get(pin.value)
        if hashes is None:
            errors.append(
                f"{relative_requirement}:{pin.line}: {pin.value} is missing from {manifest_path.relative_to(root)}"
            )
            continue
        rendered.append(f"{pin.value} \\")
        for index, value in enumerate(hashes):
            suffix = " \\" if index + 1 < len(hashes) else ""
            rendered.append(f"    --hash={value}{suffix}")

    for requirement in sorted(entries):
        if requirement not in seen:
            errors.append(
                f"{manifest_path.relative_to(root)} contains an unused entry for {relative_requirement}: {requirement}"
            )

    return rendered, errors, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify exact dependency pins against a separate sha256 manifest and optionally render a pip --require-hashes file."
    )
    parser.add_argument(
        "--requirement",
        default="requirements/ci.txt",
        help="Path to the pinned requirement file, relative to repo root.",
    )
    parser.add_argument(
        "--manifest",
        default="security/dependency_hashes.json",
        help="Path to the dependency hash manifest, relative to repo root.",
    )
    parser.add_argument(
        "--output",
        help="Write the rendered hash-verified requirements file to this path, relative to repo root unless absolute.",
    )
    args = parser.parse_args(argv)

    root = ROOT
    requirement_path = (root / args.requirement).resolve()
    manifest_path = (root / args.manifest).resolve()

    try:
        rendered, errors, resolved_manifest_path = render_hashed_requirements(
            root,
            requirement_path=requirement_path,
            manifest_path=manifest_path,
        )
    except (OSError, ValueError) as exc:
        print(f"verify_hashes.py: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("== Dependency hash verification failed ==", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = root / output_path
        write_text_file(output_path, "\n".join(rendered) + "\n")
        output_label = output_path.relative_to(root) if output_path.is_relative_to(root) else output_path
        print(f"Wrote hash-verified requirements to {output_label}.")

    print(
        f"OK: verify_hashes.py — dependency hashes for {requirement_path.relative_to(root)} match {resolved_manifest_path.relative_to(root)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
