#!/usr/bin/env bash
set -euo pipefail
size="$1"
gpu="$2"
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$base/upstream"
logs="$base/artifacts/multi_solution"
mkdir -p "$logs"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$repo"
cd experiments
if [ "$size" = 8 ]; then stage=shared; else stage=final; fi
for seed in 0 1 2; do
  "$repo/.venv/bin/python" -u nqueens_offset.py --mode "$stage" --n "$size" --method pl --arm ising --seed "$seed" > "$logs/fresh-nq${size}-s${seed}.log" 2>&1
  "$repo/.venv/bin/python" -u nqueens_offset.py --mode retime --n "$size" --stage "$stage" --method pl --arm ising --seed "$seed" > "$logs/retime-nq${size}-s${seed}.log" 2>&1
done
"$repo/.venv/bin/python" -u nqueens_free.py --mode hand --n "$size" --split dev > "$logs/hand-nq${size}-dev.log" 2>&1
for seed in 0 1 2; do
  "$repo/.venv/bin/python" -u nqueens_free.py --mode hand --n "$size" --split test --seed "$seed" > "$logs/hand-nq${size}-s${seed}.log" 2>&1
done
