#!/usr/bin/env bash
set -euo pipefail
echo "== status =="
git status -sb || true
echo
echo "== last commit =="
git log -1 --oneline || true
echo
echo "== files (max depth 4) =="
find . -maxdepth 4 -type f \
  ! -path "./.git/*" \
  ! -path "./.venv/*" \
  ! -path "./data/raw/*" \
  ! -path "./data/processed/*" \
  ! -path "./artifacts/*" \
  ! -path "./outputs/*" \
  ! -path "./reports/*" \
  | sort