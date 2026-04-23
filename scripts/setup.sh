#!/usr/bin/env bash
# Setup DrBrain: install dependencies and mineru-open-api CLI
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Installing Python dependencies ==="
uv sync

echo "=== Installing mineru-open-api CLI ==="
if command -v mineru-open-api &>/dev/null; then
  echo "mineru-open-api already installed: $(mineru-open-api version 2>&1 | head -1)"
else
  echo "Downloading mineru-open-api CLI..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -fsSL https://cdn-mineru.openxlab.org.cn/open-api-cli/install.sh | sh
  else
    curl -fsSL https://cdn-mineru.openxlab.org.cn/open-api-cli/install.sh | sh
  fi
fi

echo "=== Setup complete ==="
echo "Next: cp config.local.yaml.example config.local.yaml  # edit with your API keys"
echo "      drbrain ingest data/pdfs/your-paper.pdf"
