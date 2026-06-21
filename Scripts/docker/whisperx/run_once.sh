#!/usr/bin/env bash
set -euo pipefail

AUDIO="${1:?Usage: run_once.sh /path/to/audio [model]}"
MODEL="${2:-small}"

# Prefer env var so you don't hardcode tokens into scripts/logs
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
  echo "[run_once] ERROR: HF_TOKEN env var not set"
  echo "          Example: -e HF_TOKEN=hf_xxx"
  exit 1
fi

echo "[run_once] audio=$AUDIO"
echo "[run_once] model=$MODEL"

# Persist caches/models on the NAS (so you don't redownload every run)
export XDG_CACHE_HOME="/pipeline/cache"
mkdir -p /pipeline/cache /pipeline/diarisation /pipeline/transcripts

python -m whisperx \
  "$AUDIO" \
  --model "$MODEL" \
  --language en \
  --compute_type int8 \
  --output_dir /pipeline/diarisation \
  --diarize \
  --hf_token "$HF_TOKEN"
