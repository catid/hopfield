#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$base/upstream"
logs="$base/artifacts/multi_solution"
export CUDA_VISIBLE_DEVICES="${1:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$repo/experiments"
mkdir -p runs/CRAM-GC
cp --update=none recipes/graph-coloring.json runs/CRAM-GC/selected.json
for size in 8 10; do
  "$repo/.venv/bin/python" -u graph_coloring.py --mode fit --n "$size" --arm full --device cuda > "$logs/fresh-gc${size}-gpu-fit.log" 2>&1
done
