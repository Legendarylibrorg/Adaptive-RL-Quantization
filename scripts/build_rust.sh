#!/usr/bin/env bash
# Build optional Rust CLI accelerators (sim-eval). Python orchestrator unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/rust"
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found; install Rust from https://rustup.rs" >&2
  exit 1
fi
cargo build --release -p adaptive_rl_sim
echo "Built: $ROOT/rust/target/release/adaptive-rl-quant-rust"
