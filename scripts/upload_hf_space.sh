#!/usr/bin/env bash
# Upload only tracked source to a Hugging Face Space. Do not run
#   hf upload $REPO . .
# from the repo root: `hf` may still traverse .git and .venv (tens of GB) even
# with --exclude, which matches a failed 57k-file / ~3.3GB upload in CI.

set -euo pipefail

REPO_ID="${1:-aryannzzz/crisisops}"
STAGE="$(mktemp -d -t crisisops-hf-XXXXXX)"
REF="${2:-HEAD}"

trap 'rm -rf "$STAGE"' EXIT

git -C "$(dirname "$0")/.." archive --format=tar "$REF" | (cd "$STAGE" && tar xf -)

echo "Staged $(find "$STAGE" -type f | wc -l) files; uploading to $REPO_ID ..."

hf upload "$REPO_ID" "$STAGE" . --repo-type space \
  --commit-message "${3:-chore: upload from git archive}"

echo "Done: https://huggingface.co/spaces/${REPO_ID}"
