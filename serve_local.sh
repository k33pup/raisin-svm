#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
server="${LLAMA_SERVER:-$root/.runtime/build/bin/llama-server}"
weights="${LLAMA_MODEL_PATH:-$root/.runtime/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
exec "$server" -m "$weights" --alias qwen3-4b --host 127.0.0.1 --port 8094 \
  --jinja -c 12288 -np 1 -ngl 99 -fa on
