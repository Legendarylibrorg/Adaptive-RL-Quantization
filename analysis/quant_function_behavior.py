from __future__ import annotations

if __package__ is None:
    import shim_support as _shim
else:
    from analysis import shim_support as _shim

if __name__ == "__main__":
    _shim.dispatch_cli(__file__, "quant_function_behavior")
