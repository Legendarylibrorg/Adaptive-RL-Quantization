# Security policy

## Supported versions

Security fixes are applied against the default branch (**`main`**). Releases follow tags when published; until then, run from **`main`** for the latest fixes.

## Reporting a vulnerability

**Do not** open a public GitHub issue for undisclosed security problems.

Please report sensitive issues privately to the repository maintainers (use GitHub **Security Advisories** for this repo if enabled, or contact the org owner listed on the upstream GitHub project). Include:

- A short description and impact
- Steps to reproduce (or proof-of-concept), if safe to share
- Affected commit or version range, if known

## Hardening notes

- Treat **downloaded checkpoints** (`.pt`, legacy pickle) as **untrusted** unless you produced them locally.
- **`.env`**, keys, and tokens are **gitignored**; use `.env.example` as a template only.

For more detail see the **Security** section in [README.md](README.md).
