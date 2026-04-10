#!/usr/bin/env bash
# Lightweight secret grep on tracked files (stdlib git only — no Gitleaks).
# Heuristic; tune patterns in-repo if you hit false positives.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# -I: skip binary-looking blobs; -n: line numbers; -E: extended regexp
PATTERNS=(
  # PEM / PKCS blocks
  'BEGIN[[:space:]]+(RSA|OPENSSH|EC|DSA|PGP)[[:space:]]+PRIVATE[[:space:]]+KEY'
  'BEGIN[[:space:]]+PRIVATE[[:space:]]+KEY'
  # Common cloud / VCS tokens (high signal)
  'ghp_[0-9a-zA-Z]{36}'
  'github_pat_[0-9a-zA-Z_]{20,}'
  'AKIA[0-9A-Z]{16}'
)

hits=0
for pat in "${PATTERNS[@]}"; do
  set +e
  out="$(git grep -nEI "${pat}" -- . 2>/dev/null)"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "== Possible secret matched pattern (redact before sharing logs): ==" >&2
    echo "${out}" >&2
    hits=1
  fi
done

if [[ "${hits}" -ne 0 ]]; then
  echo >&2
  echo "secret_scan.sh: failing — remove or rotate leaked material, then amend the commit." >&2
  exit 1
fi

echo "OK: secret_scan.sh — no high-signal patterns in tracked files."
