"""Tests for repository root discovery (Rust CLI paths)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from adaptive_quant.repo_paths import default_rust_binary_paths, find_repo_root


class RepoPathsTests(unittest.TestCase):
    def test_find_repo_root_from_cwd(self) -> None:
        repo = find_repo_root(start=Path(__file__).resolve().parent.parent)
        self.assertIsNotNone(repo)
        assert repo is not None
        self.assertTrue((repo / "pyproject.toml").is_file())
        self.assertTrue((repo / "rust" / "Cargo.toml").is_file())

    def test_find_repo_root_honors_env(self) -> None:
        repo = find_repo_root(start=Path(__file__).resolve().parent.parent)
        assert repo is not None
        os.environ["ADAPTIVE_RL_REPO_ROOT"] = str(repo)
        try:
            self.assertEqual(find_repo_root(), repo)
        finally:
            os.environ.pop("ADAPTIVE_RL_REPO_ROOT", None)

    def test_default_rust_binary_paths_under_repo(self) -> None:
        repo = find_repo_root(start=Path(__file__).resolve().parent.parent)
        assert repo is not None
        paths = default_rust_binary_paths(repo)
        self.assertTrue(str(paths[0]).endswith("rust/target/release/adaptive-rl-quant-rust"))


if __name__ == "__main__":
    unittest.main()
