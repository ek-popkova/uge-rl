#!/usr/bin/env bash
# Build the report -> build/main.pdf
# Usage: ./build.sh [clean]
# Uses a local latexmk if available, otherwise the texlive/texlive Docker image.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
  rm -rf build
  echo "Removed build/"
  exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk main.tex
elif command -v docker >/dev/null 2>&1; then
  # MSYS_NO_PATHCONV: stop Git Bash on Windows from rewriting "/doc" into a Windows path
  MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/doc" -w /doc texlive/texlive:latest latexmk main.tex
else
  echo "Neither latexmk nor docker found. Install TeX Live or Docker." >&2
  exit 1
fi
echo "Done: build/main.pdf"
