#!/usr/bin/env bash
# Run full pipeline on all PDFs in data/pdfs/
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-openai/gpt-4o}"
API_BASE="${API_BASE:-}"

for pdf in data/pdfs/*.pdf data/pdfs/*.md; do
  [ -e "$pdf" ] || continue
  echo "=== Processing $pdf ==="
  uv run drbrain ingest "$pdf" --model "$MODEL" ${API_BASE:+--api-base "$API_BASE"}
done

echo "=== Done ==="
uv run drbrain stats
