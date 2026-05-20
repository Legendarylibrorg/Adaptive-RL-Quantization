from __future__ import annotations

from _repo_entrypoint import load_main

main = load_main("adaptive_quant.cli.pytorch")

if __name__ == "__main__":
    main()
