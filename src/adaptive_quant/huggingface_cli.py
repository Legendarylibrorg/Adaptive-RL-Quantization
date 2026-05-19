"""Thin, safe wrapper around the Hugging Face command-line interface.

Supports both the modern ``hf`` (huggingface_hub >= 0.34) and legacy ``huggingface-cli`` binaries.
The wrapper only **builds** and **runs** validated argv lists (no shells, no string interpolation
into a shell). Network access is the user's responsibility — we never auto-resolve ``HF_TOKEN``
and we never log secrets. Output paths are returned as ``Path`` objects when discoverable.

Public surface:

- ``find_huggingface_cli`` — locate either binary on PATH (or via ``HF_CLI`` env override).
- ``HuggingFaceCli`` — small dataclass that captures the resolved binary + its dialect.
- ``build_download_command`` — pure function returning argv for a download invocation.
- ``run_download`` — execute the command with a timeout; returns a structured ``DownloadResult``.
- ``parse_local_path`` — best-effort extract of the saved path from CLI stdout.

The wrapper is import-light and stdlib-only so that it works in the simulator-first CI path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_quant.configuration.validation import path_has_parent_reference

# Conservative regex that accepts well-formed Hugging Face Hub identifiers and filenames.
# Repos may contain a single forward slash (org/name); files allow nested directories with
# letters/digits/dot/dash/underscore/slash. We reject anything else to guarantee that argv
# entries cannot smuggle option flags or shell metacharacters even if accidentally splatted.
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

_HF_CLI_NEW = "hf"
_HF_CLI_LEGACY = "huggingface-cli"

_WIN_CLI_SUFFIXES = frozenset({".exe", ".bat", ".cmd", ".ps1", ".com"})


def _is_plausibly_executable(path: Path) -> bool:
    """Whether ``path`` may be executed as a CLI entrypoint (platform-aware).

    On POSIX, use the execute bit. On Windows, ``os.access(..., X_OK)`` is not meaningful
    for ordinary files, so we accept typical launcher suffixes or a PE ``MZ`` header for
    extensionless binaries.
    """
    if not path.is_file():
        return False
    if os.name == "nt":
        suffix = path.suffix.lower()
        if suffix in _WIN_CLI_SUFFIXES:
            return True
        if suffix == "":
            try:
                with path.open("rb") as handle:
                    return handle.read(2) == b"MZ"
            except OSError:
                return False
        return False
    return os.access(path, os.X_OK)


_LOCAL_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Successfully downloaded\s+(?P<path>\S+)", re.IGNORECASE),
    re.compile(r"Downloaded to\s+(?P<path>\S+)", re.IGNORECASE),
    re.compile(r"^(?P<path>/\S+\.gguf)\s*$", re.IGNORECASE | re.MULTILINE),
)


@dataclass(frozen=True)
class HuggingFaceCli:
    """Resolved Hugging Face CLI binary.

    ``dialect`` is ``"hf"`` for the modern ``hf <command>`` layout and ``"legacy"`` for the
    older ``huggingface-cli <command>`` layout. Callers should prefer ``build_download_command``
    rather than constructing argv directly so the dialect is honored.
    """

    binary: str
    dialect: str

    @property
    def kind(self) -> str:
        return self.dialect


@dataclass
class DownloadResult:
    """Structured result of a download invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    local_path: Path | None = None
    timed_out: bool = False
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def find_huggingface_cli(*, override_env: str = "HF_CLI") -> HuggingFaceCli | None:
    """Locate a Hugging Face CLI binary; honour ``HF_CLI`` env override if pointed at an executable."""
    override = os.environ.get(override_env)
    if override:
        candidate = Path(override).expanduser()
        if ".." in candidate.parts:
            return None
        if candidate.is_file() and _is_plausibly_executable(candidate):
            dialect = _HF_CLI_NEW if candidate.stem.lower() == _HF_CLI_NEW else "legacy"
            return HuggingFaceCli(binary=str(candidate), dialect=dialect)

    new_path = shutil.which(_HF_CLI_NEW)
    if new_path:
        return HuggingFaceCli(binary=new_path, dialect=_HF_CLI_NEW)

    legacy_path = shutil.which(_HF_CLI_LEGACY)
    if legacy_path:
        return HuggingFaceCli(binary=legacy_path, dialect="legacy")

    return None


def require_huggingface_cli(*, override_env: str = "HF_CLI") -> HuggingFaceCli:
    """Resolve the CLI or raise a ``FileNotFoundError`` with installation guidance."""
    cli = find_huggingface_cli(override_env=override_env)
    if cli is None:
        raise FileNotFoundError(
            "Hugging Face CLI not found on PATH. Install via "
            "'pip install -U huggingface_hub' (provides 'hf', or legacy 'huggingface-cli'), "
            f"or set {override_env}=/path/to/hf."
        )
    return cli


def build_download_command(
    cli: HuggingFaceCli,
    *,
    repo_id: str,
    filename: str | None = None,
    revision: str | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    include_patterns: list[str] | None = None,
) -> list[str]:
    """Return a validated argv list for downloading from a Hugging Face repo.

    The function does **not** spawn subprocesses; it only builds a vetted argv. Pass the
    return value to ``run_download`` (or ``subprocess.run`` directly) to execute.
    """
    _validate_repo_id(repo_id)
    if filename is not None:
        _validate_filename(filename)
    if revision is not None:
        _validate_revision(revision)
    if include_patterns:
        for pattern in include_patterns:
            _validate_filename(pattern)

    argv: list[str] = [cli.binary, "download", repo_id]
    if filename is not None:
        argv.append(filename)
    if revision is not None:
        argv.extend(["--revision", revision])
    if local_dir is not None:
        argv.extend(["--local-dir", str(_validate_local_dir(local_dir))])
    if include_patterns:
        for pattern in include_patterns:
            argv.extend(["--include", pattern])
    return argv


def run_download(
    cli: HuggingFaceCli,
    *,
    repo_id: str,
    filename: str | None = None,
    revision: str | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    include_patterns: list[str] | None = None,
    timeout_s: float = 600.0,
    env: dict[str, str] | None = None,
) -> DownloadResult:
    """Build and execute a download invocation, returning a structured result.

    The function never raises on a non-zero exit code; inspect ``DownloadResult.ok``
    instead. Timeouts surface as ``timed_out=True`` rather than an exception so callers
    can decide whether to retry.
    """
    command = build_download_command(
        cli,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
        include_patterns=include_patterns,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return DownloadResult(
            command=command,
            returncode=-1,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return DownloadResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=stdout,
        stderr=stderr,
        local_path=parse_local_path(stdout + "\n" + stderr),
    )


def parse_local_path(text: str) -> Path | None:
    """Best-effort extraction of the saved path from CLI output (stdout + stderr)."""
    if not text:
        return None
    for pattern in _LOCAL_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                resolved = Path(match.group("path")).expanduser()
            except OSError:
                continue
            if ".." in resolved.parts:
                continue
            return resolved
    return None


def _validate_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not _REPO_RE.match(repo_id):
        raise ValueError(
            f"Invalid Hugging Face repo_id {repo_id!r}: expected '<org>/<name>' with "
            "alphanumeric, '.', '_', or '-' only."
        )


def _validate_filename(filename: str) -> None:
    if not isinstance(filename, str) or not _FILENAME_RE.match(filename):
        raise ValueError(
            f"Invalid Hugging Face filename {filename!r}: expected alphanumeric, '.', '_', '-', or '/'."
        )
    if filename.startswith("-"):
        raise ValueError(f"Filename must not start with '-' (would be parsed as a flag): {filename!r}")
    if ".." in Path(filename).parts:
        raise ValueError(f"Filename must not contain '..': {filename!r}")


def _validate_revision(revision: str) -> None:
    if not isinstance(revision, str) or not _REVISION_RE.match(revision):
        raise ValueError(
            f"Invalid revision {revision!r}: expected alphanumeric, '.', '_', '-', or '/'."
        )
    if revision.startswith("-"):
        raise ValueError(f"Revision must not start with '-' (would be parsed as a flag): {revision!r}")
    if ".." in Path(revision).parts:
        raise ValueError(f"Revision must not contain '..': {revision!r}")


def _validate_local_dir(local_dir: str | os.PathLike[str]) -> Path:
    path = Path(local_dir)
    raw = str(path)
    if not raw.strip():
        raise ValueError("local_dir must be non-empty")
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise ValueError("local_dir contains invalid control characters")
    if ".." in path.parts:
        raise ValueError(f"local_dir must not contain '..': {raw!r}")
    if path.name.startswith("-"):
        raise ValueError(f"local_dir final component must not start with '-': {raw!r}")
    return path


__all__ = [
    "DownloadResult",
    "HuggingFaceCli",
    "build_download_command",
    "find_huggingface_cli",
    "parse_local_path",
    "require_huggingface_cli",
    "run_download",
]
