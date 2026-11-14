#!/bin/bash
set -euo pipefail
TASK_ROOT=/home/catid/hopfield
PY=$TASK_ROOT/upstream/.venv/bin/python
export CRAM_DATA=$TASK_ROOT/upstream/data
for seed in 0 1 2; do
  out=$TASK_ROOT/artifacts/fresh_maze/maze_unique/seed$seed
  "$PY" "$TASK_ROOT/scripts/run_maze_reproduction.py" unique --seed "$seed" --out "$out"
  "$PY" "$TASK_ROOT/scripts/run_maze_reproduction.py" unique-time --seed "$seed" --out "$out" --device cuda
  CUDA_VISIBLE_DEVICES= "$PY" "$TASK_ROOT/scripts/run_maze_reproduction.py" unique-time --seed "$seed" --out "$out" --device cpu
 done
