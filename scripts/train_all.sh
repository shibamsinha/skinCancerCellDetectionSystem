#!/usr/bin/env bash
# Train and evaluate every model sequentially (primary model first).
# Re-running with --resume continues any interrupted run and skips finished ones.
#
#   bash scripts/train_all.sh            # fresh runs
#   bash scripts/train_all.sh --resume   # continue after an interruption
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
MODELS="${MODELS:-modified_densenet201 mobilenet_v2 resnet50 xception googlenet densenet201}"
mkdir -p results
for m in $MODELS; do
  echo "=== $(date '+%F %T') training $m ==="
  if ! "$PY" scripts/train.py --model "$m" "$@"; then
    echo "!!! $m failed (see results/$m/train.log); continuing with the next model"
  fi
done
"$PY" scripts/evaluate.py --all >/dev/null 2>&1 || true
echo "=== $(date '+%F %T') all done ==="
