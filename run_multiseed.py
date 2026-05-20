from __future__ import annotations

from _repo_entrypoint import load_main

main = load_main("adaptive_quant.cli.multiseed")

if __name__ == "__main__":
    main()
