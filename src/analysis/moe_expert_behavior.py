from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.shim_support import run_shim_main

if __name__ == "__main__":
    run_shim_main(__file__)
