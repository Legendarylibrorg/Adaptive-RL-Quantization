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
BINARY="$ROOT/rust/target/release/adaptive-rl-quant-rust"
if [[ ! -f "$BINARY" ]]; then
  BINARY="$ROOT/rust/target/release/adaptive-rl-quant-rust.exe"
fi
if [[ ! -f "$BINARY" ]]; then
  echo "Build finished but binary missing under rust/target/release/" >&2
  exit 1
fi
echo "Built: $BINARY"
echo "Enable in config: rust_simulator_enabled=true (backend=simulator, moe_enabled=false)"
echo "Or export: ADAPTIVE_RL_RUST_CLI=$BINARY"
