#!/usr/bin/env bash
set -euo pipefail

IN="/pipeline/incoming"
PROC="/pipeline/processing"
DONE="/pipeline/transcripts"
FAIL="/pipeline/failed"
LOGDIR="/pipeline/logs"

MODEL="${MODEL:-medium}"

mkdir -p "$PROC" "$DONE" "$FAIL" "$LOGDIR"

echo "$(date '+%F %T') [watcher] started"

while true; do
  shopt -s nullglob
  FILES=("$IN"/*.mp3 "$IN"/*.m4a "$IN"/*.wav)

  if [ ${#FILES[@]} -eq 0 ]; then
    sleep 20
    continue
  fi

  for f in "${FILES[@]}"; do
    base=$(basename "$f")
    logfile="$LOGDIR/${base}.log"

    echo "$(date '+%F %T') [watcher] claiming $base"
    mv "$f" "$PROC/$base"

    start=$(date +%s)

    {
      echo "===================================="
      echo "File: $base"
      echo "Model: $MODEL"
      echo "Started: $(date '+%F %T')"
      echo "===================================="
    } > "$logfile"

    set +e
    /pipeline/scripts/docker/whisperx/run_once.sh "$PROC/$base" "$MODEL" >> "$logfile" 2>&1
    status=$?
    set -e

    end=$(date +%s)
    duration=$((end-start))

    echo "Finished: $(date '+%F %T')" >> "$logfile"
    echo "Duration: ${duration}s" >> "$logfile"
    echo "Exit status: $status" >> "$logfile"

    if [ $status -eq 0 ]; then
      echo "$(date '+%F %T') [watcher] success $base (${duration}s)"
      mv "$PROC/$base" "$DONE/$base"
    else
      echo "$(date '+%F %T') [watcher] FAILED $base (${duration}s) — see $logfile"
      mv "$PROC/$base" "$FAIL/$base"
    fi
  done
done
