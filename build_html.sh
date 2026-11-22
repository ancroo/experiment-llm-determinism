#!/usr/bin/env bash
# Regenerate publications/*.html from publications/*.md (pandoc, standalone).
# Run before committing a changed text, so the HTML never lags behind.
set -euo pipefail
cd "$(dirname "$0")"
for md in publications/*.md; do
  pandoc -s --metadata pagetitle="$(basename "$md" .md)" "$md" -o "${md%.md}.html"
  echo "built ${md%.md}.html"
done
