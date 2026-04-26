# Getting help

Thanks for using **Adaptive-RL-Quantization**. This document tells you where each kind of question goes so we can route it correctly and keep the security channel quiet for actual vulnerabilities.

## Choose the right channel

| You want to…                                                         | Use this channel                                                                 |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Report a security vulnerability** (private)                        | [GitHub Security Advisories](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/security/advisories/new) — see [SECURITY.md](SECURITY.md). **Do not** open a public issue. |
| **Report a bug** in shipped code, tests, or documented workflows     | [Open a Bug report issue](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/issues/new?template=bug_report.yml) |
| **Propose a feature or research idea**                               | [Open a Feature / research issue](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/issues/new?template=feature_request.yml) |
| **Ask "how do I…?" / general usage question**                        | Browse existing issues and the [README](README.md) / [`docs/`](docs/) first; if your question is still unanswered, open a documentation-flavored issue describing what you tried. |
| **Submit a fix or improvement**                                      | Open a pull request — see [CONTRIBUTING.md](CONTRIBUTING.md) and [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) |
| **Conduct concerns** (harassment, abusive comments)                  | See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Conduct is **not** a security channel. |

## Before you open an issue

A short checklist that keeps issues actionable:

- **Search first.** Open and closed [issues](https://github.com/Legendarylibrorg/Adaptive-RL-Quantization/issues?q=is%3Aissue) often answer common questions, especially around install and CUDA setup.
- **Read the relevant docs.** [README.md](README.md) covers install, CLIs, and outputs; [`docs/INSTALL.md`](docs/INSTALL.md) covers GPU/PyTorch specifics.
- **Run the local quality gate** (`python3 scripts/pre_commit_check.py`) if your bug looks like a regression — it runs the same secret scan, hash verification, syntax check, and unit tests CI runs.
- **Provide reproduction details.** OS, Python version, the command you ran, the config / preset, and the full error or unexpected output. Sanitize secrets before pasting logs.
- **Mention if PyTorch / CUDA is involved.** Many simulator-path bugs have nothing to do with GPU and vice versa; calling that out up front saves a round trip.

## Response expectations

This is a maintained open-source project, not a paid support product. Maintainers respond on a best-effort basis. Security reports follow the response targets in [SECURITY.md](SECURITY.md); other channels do not have an SLA.

## Commercial / private support

There is currently no commercial support tier. If you need bespoke help (e.g., integrating the policy into a private inference stack), open a feature issue describing your use case and constraints — community contributions and discussion happen there in public.
