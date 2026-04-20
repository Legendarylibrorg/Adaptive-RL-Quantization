# Security policy

## Supported versions

Security fixes are applied against the default branch (**`main`**). Releases follow tags when published; until then, run from **`main`** for the latest fixes.

## Reporting a vulnerability

**Do not** open a public GitHub issue for undisclosed security problems. (General conduct concerns belong under [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), not the security reporting channel.)

For the canonical GitHub repository, the expected private reporting channel is **GitHub Security Advisories**. Maintainers should keep repository security reporting enabled before advertising public releases or mirrors.

If you are reporting against a fork or mirror that does not provide private advisories, contact that fork's maintainer through the private channel they publish for that copy. Do not use this repository's public issue tracker for undisclosed vulnerabilities.

Include:

- A short description and impact
- Steps to reproduce (or proof-of-concept), if safe to share
- Affected commit or version range, if known

## Hardening notes

- Treat **downloaded checkpoints** (`.pt`, legacy pickle) as **untrusted** unless you produced them locally.
- **`FrameworkConfig` path fields** (`*_dir`, optional `resume_from_checkpoint`, `llama_cpp_*` paths) reject `..` path components and NUL/newlines to limit traversal surprises when merging untrusted JSON/TOML with `run_name`-based filenames.
- **JSONL analysis** caps per-file size, line count, and **per-line UTF-8 byte length** so a single huge line cannot exhaust memory as easily.
- **`.env`**, keys, and tokens are **gitignored**; use `.env.example` as a template only.
- **Local pre-commit** runs a small **[`scripts/secret_scan.py`](scripts/secret_scan.py)** heuristic scan over tracked text files, and the dedicated **Secret Scan** GitHub Actions workflow runs **`python scripts/secret_scan.py --history`** with full fetch depth to catch high-signal leaks in reachable history too. It catches common mistakes, not every leak; enable platform secret scanning where you host the repo.
- **CI bootstrap dependencies** are pinned in **[`requirements/ci.txt`](requirements/ci.txt)** and checked against **[`security/dependency_hashes.json`](security/dependency_hashes.json)** before install, so GitHub Actions uses **`pip --require-hashes`** for fetched Python packages.
- **Dependabot** monitors the root pip manifests and **`requirements/`** so dependency updates stay reviewable and centralized.

For more detail see the **Security** section in [README.md](README.md).
