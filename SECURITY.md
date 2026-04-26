# Security policy

We take security reports seriously. This document is the **canonical disclosure policy** for this repository: how to report a vulnerability, what to expect from maintainers, what is in scope, and what good-faith research looks like here.

> **TL;DR — Report a vulnerability privately at:** <https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new>

## Supported versions

Security fixes are applied against the default branch (**`main`**). Releases follow tags when published; until then, run from **`main`** for the latest fixes. We do not backport fixes to forks or unreleased experimental branches.

| Version       | Supported security fixes |
|---------------|--------------------------|
| `main` (HEAD) | Yes                      |
| Tagged release `>= latest` | Yes (current minor) |
| Older tags / forks | Best effort only — please retest against `main` first |

## Reporting a vulnerability

**Do not** open a public GitHub issue, pull request, discussion, or social-media post for an undisclosed security problem. (General conduct concerns belong under [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), not the security reporting channel.)

Use **GitHub Security Advisories (private vulnerability reporting)**:

1. Go to <https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new>.
2. Click **Report a vulnerability** and fill out the advisory form.
3. Wait for a maintainer reply on that advisory thread. All triage, fix coordination, and CVE/GHSA assignment happen there.

If you are reporting against a **fork or mirror** that does not provide private advisories, contact that fork's maintainer through the private channel they publish for that copy. Do not use this repository's public issue tracker for undisclosed vulnerabilities.

If GitHub Security Advisories is unavailable to you (e.g., you are operating from a network that blocks GitHub), open an empty placeholder issue titled *"Requesting private security contact"* — **without** vulnerability details — and a maintainer will respond with an alternative channel.

### What to include

A high-signal report makes triage fast:

- A short description and **impact** (what an attacker can do, and against whom).
- **Steps to reproduce** or a minimal proof-of-concept, if safe to share.
- **Affected commit / version range**, if known (e.g. `git log -1 --format=%H`).
- Configuration that triggers the issue (sanitize secrets first).
- Your proposed fix, if you have one.

Machine-readable disclosure metadata is published at [`.well-known/security.txt`](./.well-known/security.txt) (RFC 9116) for scanners and automation.

## Our response

When you submit a private advisory, you can expect:

| Stage              | Target                                                            |
|--------------------|-------------------------------------------------------------------|
| Acknowledgement    | within **5 business days**                                        |
| Initial assessment | within **14 days** of acknowledgement (severity, scope, ownership)|
| Fix or mitigation  | severity-driven; **critical/high** prioritized                    |
| Public disclosure  | coordinated with you once a fix or mitigation is available        |

We will keep you in the loop on the advisory thread, and credit you in the published advisory unless you ask us not to. CVE / GHSA identifiers are requested through the GitHub Security Advisories workflow when warranted.

## Scope

**In scope** (please report):

- Code in this repository, including:
  - the `adaptive_quant` package and `analysis` package,
  - configuration loading (`adaptive_quant/configuration.py`, `adaptive_quant/easy_config.py`),
  - artifact and SVG generation (`adaptive_quant/analysis_utils.py`, `adaptive_quant/logging_utils.py`),
  - subprocess and external-binary integrations (`llama.cpp` calibration, `git` invocations),
  - bootstrap and verification scripts under `scripts/` (e.g., `verify_hashes.py`, `secret_scan.py`, `setup_from_clone.py`, `pre_commit_check.py`),
  - CI and supply-chain configuration in `.github/` (`requirements/ci.txt`, `security/dependency_hashes.json`).
- Documented workflows and CLIs shipped in this repo (`run_research.py`, `run_pytorch.py`, console scripts in `pyproject.toml`).

**Out of scope** (we will not treat as vulnerabilities):

- Issues that require local code-execution rights the user already has on their own machine (e.g., "if I edit `config.py`, the program does what I told it to").
- Behavior of **third-party dependencies** themselves (PyTorch, NumPy, llama.cpp). Report those upstream; if our pinning or hash-verification is the actual bug, that *is* in scope.
- Findings in **untrusted checkpoints / models** loaded against our explicit warnings — the README and SECURITY policy already say to treat downloaded `.pt` / legacy pickle artifacts as untrusted.
- Reports that consist only of automated-scanner output with no demonstrated impact.
- Self-XSS or similar issues that require an attacker to convince a victim to paste arbitrary content into their own configuration / shell.
- Denial-of-service via resource exhaustion when the user explicitly raises caps (e.g., setting `training_episodes=10**9`).
- Missing security-hardening features that are not required by our threat model (we accept feature requests for those, but they are not vulnerabilities).

If you are unsure whether something is in scope, **ask** in the advisory — we would rather see a borderline report than miss a real one.

## Safe harbor

We support good-faith security research. If you make a sincere effort to comply with this policy, we will:

- Treat your research as authorized under our policy and not pursue or support any legal action related to it.
- Work with you to understand and resolve the issue quickly.
- Recognize your contribution publicly if you wish.

In exchange we ask that you:

- **Do not** access, modify, or destroy data that is not yours.
- **Do not** degrade availability for other users (no flooding, brute-force, or DoS).
- **Stop** when you have established proof-of-concept; do not pivot or escalate.
- **Report privately** through the channels above; **do not** publish details before a coordinated fix.

This safe harbor applies only to this repository's code. Third-party services, social-engineering, physical attacks, and law enforcement matters are outside its scope.

## Coordinated disclosure

We follow a **coordinated disclosure** model:

- We aim to release a fix or mitigation **before** any public disclosure.
- We will publish a GitHub Security Advisory at disclosure time with the affected versions, fixed version(s), credits, and a workaround if applicable.
- If a fix is not feasible within **90 days** of acknowledgement, we will discuss timeline with you and may publish guidance even without a code fix. Either party may request an extension; we expect collaborative timelines, not unilateral deadlines.

## Credit

Reporters are credited in the published advisory by the name and link they choose, unless they ask to remain anonymous. Anonymous reports are welcome.

## Hardening notes (defense in depth)

The following are deliberate, repo-level mitigations. They are not a substitute for the disclosure flow above; they exist so that maintainers and users understand the threat model:

- Treat **downloaded checkpoints** (`.pt`, legacy pickle) as **untrusted** unless you produced them locally.
- **`FrameworkConfig` path fields** (`*_dir`, optional `resume_from_checkpoint`, `llama_cpp_*` paths) reject `..` path components and NUL/newlines to limit traversal surprises when merging untrusted JSON/TOML with `run_name`-based filenames. Absolute paths are allowed by design so users can point at their own data; if you load shared/CI-supplied configs, prefer relative paths under the repo and review absolute targets before running.
- **SVG charts** (analysis bar/scatter plots) HTML-escape titles and labels so user-provided strings (e.g. MoE variant names) cannot inject markup if a viewer renders the SVG inline.
- **JSONL analysis** caps per-file size, line count, and **per-line UTF-8 byte length** so a single huge line cannot exhaust memory as easily.
- **`.env`**, keys, and tokens are **gitignored**; use `.env.example` as a template only.
- **CI / local pre-commit** run a small **[`scripts/secret_scan.py`](scripts/secret_scan.py)** heuristic scan over tracked text files. It catches common mistakes, not every leak; enable platform secret scanning where you host the repo.
- **CI bootstrap dependencies** are pinned in **[`requirements/ci.txt`](requirements/ci.txt)** and checked against **[`security/dependency_hashes.json`](security/dependency_hashes.json)** before install, so GitHub Actions uses **`pip --require-hashes`** for fetched Python packages.
- **`scripts/setup_from_clone.py`** does not download `get-pip.py` from the internet by default. Network bootstrap is gated behind the explicit opt-in env var `ADAPTIVE_RL_ALLOW_NETWORK_PIP_BOOTSTRAP=1` so misconfigured environments fail loudly instead of silently fetching code.
- **Dependabot** monitors the root pip manifests and **`requirements/`** so dependency updates stay reviewable and centralized.
- **Dependency review** runs on pull requests through `.github/workflows/dependency-review.yml` for public clones of this repo.

For more detail see the **Security** section in [README.md](README.md).
