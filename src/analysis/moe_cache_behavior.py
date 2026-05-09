from __future__ import annotations

try:
    from analysis.shim_support import dispatch_named_cli
except ImportError:
    from shim_support import dispatch_named_cli

if __name__ == "__main__":
    dispatch_named_cli("moe_cache_behavior")
