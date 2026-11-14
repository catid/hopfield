#!/bin/bash
set -euo pipefail
TASK_ROOT=/home/catid/hopfield
PY=$TASK_ROOT/upstream/.venv/bin/python
export CRAM_DATA=$TASK_ROOT/upstream/data
export CUDA_VISIBLE_DEVICES=
for seed in 0 1 2; do
  out=$TASK_ROOT/artifacts/fresh_maze/maze_unique/seed$seed
  while ! test -f "$out/summary.json"; do sleep 5; done
  "$PY" "$TASK_ROOT/scripts/run_maze_reproduction.py" unique-time --seed "$seed" --out "$out" --device cpu
done
