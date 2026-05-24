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
  - configuration loading (`src/adaptive_quant/configuration/`, `src/adaptive_quant/easy_config.py`),
  - artifact and SVG generation (`src/adaptive_quant/analysis_utils.py`, `src/adaptive_quant/logging_utils.py`),
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
  - The PyTorch v2 checkpoint loader prefers `torch.load(weights_only=True)` and does **not** fall back to pickle-capable loads. Legacy single-file pickle checkpoints are refused in this runtime; convert them only in a separate trusted environment.
  - The stdlib trainer's JSON checkpoint refuses to load any payload that lacks a serialized `policy_state` (i.e., legacy checkpoints from before this format are rejected by default).
- **`FrameworkConfig` path fields** (`*_dir`, optional `resume_from_checkpoint`, `llama_cpp_*` paths) reject `..` path components and NUL/newlines to limit traversal surprises when merging untrusted JSON/TOML with `run_name`-based filenames. Absolute paths are allowed by design so users can point at their own data; if you load shared/CI-supplied configs, prefer relative paths under the repo and review absolute targets before running.
- **Optional Hugging Face router embeddings** (`router_feature_backend="hf"` in [`adaptive_quant/routing.py`](src/adaptive_quant/routing.py)): `AutoModel.from_pretrained` loads **safetensors** weights only (`use_safetensors=True`) with **`trust_remote_code=False`**. Config load **requires** a non-empty `router_hf_allowed_models` allowlist and a pinned `router_hf_embedding_revision`; model ids must use `org/name` format. Prefer `router_hf_local_files_only=true` after caching a vetted snapshot.
- **Hugging Face Hub downloads** ([`huggingface_cli.py`](src/adaptive_quant/huggingface_cli.py), route catalog): `repo_id`, filenames, and revisions are regex-validated; GGUF routes must use a `.gguf` filename. **Downloads are denied by default** unless you set `ADAPTIVE_RL_HF_ALLOWED_REPOS` and/or `route_hf_allowed_repos`. For trusted local-only workflows, opt out with `ADAPTIVE_RL_HF_ALLOW_UNLISTED=1` (not recommended on shared infrastructure).
- **`LlamaCppBackend` shells out to a user-supplied `llama_cpp_main_path`** (and optional model/calibration files). Treat that binary and **GGUF** (or other on-disk model) paths as part of your trust boundary: do **not** point them at unverified third-party builds when running against shared infrastructure. The default backend is the in-process simulator; switch to `llama_cpp` only when you own the artifact.
- **SVG charts** (analysis bar/scatter plots) HTML-escape titles and labels so user-provided strings (e.g. MoE variant names) cannot inject markup if a viewer renders the SVG inline.
- **JSONL analysis** caps per-file size, line count, and **per-line UTF-8 byte length** so a single huge line cannot exhaust memory as easily.
- **`.env`**, keys, and tokens are **gitignored**; use `.env.example` as a template only.
- **CI / local pre-commit** run a small **[`scripts/secret_scan.py`](scripts/secret_scan.py)** heuristic scan over tracked text files. It catches common mistakes, not every leak; enable platform secret scanning where you host the repo.
- **CI bootstrap dependencies** are pinned in **[`requirements/ci.txt`](requirements/ci.txt)** and checked against **[`security/dependency_hashes.json`](security/dependency_hashes.json)** before install, so GitHub Actions uses **`pip --require-hashes`** for fetched Python packages.
- **CI dev** installs use a hash-pinned lockfile **[`requirements/dev.txt`](requirements/dev.txt)** (includes `pip-audit`); **[`requirements/pytorch-cpu.txt`](requirements/pytorch-cpu.txt)** is hash-pinned for optional Docker `torch` installs. [`scripts/verify_lockfiles.py`](scripts/verify_lockfiles.py) checks inline hashes (regenerate via [`scripts/compile_locked_requirements.py`](scripts/compile_locked_requirements.py)).
- **Weekly pip-audit** on `main` audits bootstrap + dev lockfiles via [`.github/workflows/pip-audit-scheduled.yml`](.github/workflows/pip-audit-scheduled.yml) (PyTorch lockfile is Dependabot-reviewed but not scanned in CI because Linux `torch` pulls platform-specific CUDA packaging that cannot be hash-pinned in a single cross-platform lockfile).
- **Router / online prompts** are length-capped (`MAX_ROUTER_TASK_TEXT_CHARS`) to limit tokenizer RAM exhaustion; analysis CLI paths reject `..` and control characters. User prompt text is **NFKC-normalized** and **zero-width / format characters are stripped** before routing, online learning, or `llama.cpp` invocation.
- **Online learning replay** caps how many replay-buffer entries may originate from the same prompt hash (`online_max_replay_entries_per_prompt_hash`) to limit adversarial prompt-stream poisoning.
- **Checkpoint integrity** — Python JSON checkpoints and PyTorch v2 sidecars include a SHA-256 `integrity_sha256` tag; loads verify the digest unless `ADAPTIVE_RL_SKIP_CHECKPOINT_INTEGRITY=1`. Legacy sidecars without the field still load unless `ADAPTIVE_RL_REQUIRE_CHECKPOINT_INTEGRITY=1`.
- **JSONL integrity chain** — when `ADAPTIVE_RL_JSONL_INTEGRITY_CHAIN=1`, each JSONL record links to the previous line hash (enabled in hardened Docker Compose). `load_jsonl` verifies `_integrity_hash` / `_integrity_prev` when present; set `ADAPTIVE_RL_JSONL_REQUIRE_INTEGRITY_CHAIN=1` to reject legacy lines without tags.
- **Security bypass visibility** — pipeline entrypoints warn on stderr when `ADAPTIVE_RL_HF_ALLOW_UNLISTED=1` or `ADAPTIVE_RL_SKIP_CHECKPOINT_INTEGRITY=1` is set; set `ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS=1` to fail fast in CI/lab images.
- **Security audit records** — pipeline and online summaries include a `security_audit` block (resolved `llama.cpp` paths, HF allowlist state, router config).
- **`scripts/setup_from_clone.py`** does not download `get-pip.py` from the internet by default. Network bootstrap is gated behind the explicit opt-in env var `ADAPTIVE_RL_ALLOW_NETWORK_PIP_BOOTSTRAP=1` **and** an `ADAPTIVE_RL_PIP_BOOTSTRAP_SHA256` digest the script verifies before executing the downloaded script, so a misconfigured environment fails loudly rather than running an unverified bootstrap. The download uses an explicit `ssl.create_default_context()`, a 30 s socket timeout, and a 16 MiB read cap, so a hung or oversized response cannot stall the installer or exhaust memory.
- **Embedded `git` calls** (`adaptive_quant.pipeline.vcs.git_commit_hash`, `scripts/env_report._git`, `scripts/secret_scan.scan_tracked_files`, `scripts/pre_commit_check._has_staged_changes` and the `git diff --check` invocations it shells out) all run with short subprocess timeouts so a hung credential prompt, partitioned network, or unresponsive filesystem (NFS, WSL2 over a Windows mount, etc.) cannot indefinitely block a pipeline, doctor command, or pre-commit hook. `secret_scan.py` also skips files larger than 4 MiB to avoid OOM on accidentally checked-in blobs; binary content is detected by NUL byte and skipped.
- **Stdlib (Python) checkpoint deserialization** (`adaptive_quant.policy._restore_categorical_head`, `_gaussian_head_from_payload`, `_value_head_from_payload`) requires every weight, bias, and `stddev` to be a finite float. JSON-encoded `Infinity` / `NaN` are rejected so an attacker-supplied checkpoint cannot inject poison values that silently corrupt later training updates. `previous_action` is also length-validated and finite-checked on load (`adaptive_quant.base_trainer.coerce_previous_action`) for both the Python and the PyTorch trainers.
- **Online learning** (`adaptive_quant.online_learning.OnlineLearningLoop`) records candidate policy actions, applies a guardrail/canary check, and rolls back to the best recent snapshot when the rolling served-reward mean falls more than `online_drift_reward_delta` below the best window seen so far. The replay/telemetry JSONL files contain prompts, hardware labels, and reward signals — treat them as **operational telemetry**, not user data, and scrub before publishing if you ran the loop against private prompts.
- **Dependabot** monitors the root pip manifests and **`requirements/`** so dependency updates stay reviewable and centralized.
- **Dependency review** runs on pull requests through `.github/workflows/dependency-review.yml` for public clones of this repo (`fail-on-severity: high`).
- **Runtime path validation** rejects `..` and control characters on llama.cpp binaries/models, route-catalog `local_path`, and `llama_cpp:` router routes (config load and subprocess invocation).
- **Config episode caps** — `training_episodes`, `evaluation_episodes`, `max_training_episodes`, and related counters loaded from JSON/TOML are bounded by `MAX_EPISODE_COUNT` (1,000,000) to limit accidental or hostile DoS via huge integers.
- **Structural config caps** — architecture and workload integers (`num_layers`, `torch_hidden_dim`, `llama_cpp_context`, MoE expert counts, etc.) are bounded in `FrameworkConfig` validation so a single hostile config cannot allocate multi-gigabyte policy tensors.
- **Recommendation / llama.cpp caps** — `recommendation_eval_episodes`, `recommendation_candidate_limit`, `llama_cpp_generate_tokens`, `jsonl_flush_every`, and `llama_cpp_cache_max_entries` are bounded at config load (see `MAX_*` in `configuration/validation.py`).
- **Optional llama.cpp binary allowlist** — when `ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES` is set (``os.pathsep``-separated roots), `require_llama_cpp_paths` refuses binaries that resolve outside those directories.
- **Bootstrap `pip-audit`** — CI audits hash-pinned packages in `requirements/ci.txt` and `requirements/dev.txt` in addition to PR dependency review.

For untrusted artifacts, prefer **Tier 1** in [docs/SECURE_RUN.md](docs/SECURE_RUN.md): disposable Linux VM → hardened Docker Compose (`make docker-preflight` / `make docker-gpu-preflight` inside the VM). Real CUDA training belongs in a CUDA-matched venv in that same VM, not the CPU `torch` wheel baked into the optional GPU image smoke path.
