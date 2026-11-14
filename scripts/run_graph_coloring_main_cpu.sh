#!/usr/bin/env bash
set -euo pipefail
size="$1"
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$base/upstream"
logs="$base/artifacts/multi_solution"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$repo"
"$repo/.venv/bin/python" -u checkpoints/evaluate.py --tasks "graph-coloring-$size" --seeds 0 1 2 > "$logs/released-gc${size}.log" 2>&1
cd experiments
"$repo/.venv/bin/python" -u graph_coloring.py --mode fit --n "$size" --arm full --device cpu > "$logs/fresh-gc${size}-cpu-fit.log" 2>&1
if [ "$size" = 8 ]; then temperature=2; else temperature=3; fi
"$repo/.venv/bin/python" -u graph_coloring.py --mode test --n "$size" --arm full --stop terms --temperature "$temperature" --seeds 0 1 2 > "$logs/fresh-gc${size}-test.log" 2>&1
"$repo/.venv/bin/python" -u graph_coloring_hand_set.py --mode dev --n "$size" > "$logs/hand-gc${size}-dev.log" 2>&1
for seed in 0 1 2; do
  "$repo/.venv/bin/python" -u graph_coloring_hand_set.py --mode test --n "$size" --seed "$seed" > "$logs/hand-gc${size}-s${seed}.log" 2>&1
done
