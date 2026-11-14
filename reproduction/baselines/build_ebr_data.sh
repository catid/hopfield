#!/bin/bash
set -euo pipefail
source /home/catid/hopfield/reproduction/baselines/env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
cd "$EBR_SRC"
"$EBR_PY" -m dataset.build_sudoku_dataset --source-dir "$CRAM_DATA/sudoku-extreme" --output-dir "$EBR_DATA/sudoku-extreme-1k-aug-1000" --subsample-size 1000 --num-aug 1000
"$EBR_PY" -m dataset.build_nqueens_dataset --n 8 --output-dir "$EBR_DATA/nqueens-8"
"$EBR_PY" -m dataset.build_nqueens_dataset --n 10 --output-dir "$EBR_DATA/nqueens-10"
"$EBR_PY" -m dataset.build_graph_coloring_dataset --nodes 8 --output-dir "$EBR_DATA/graph-coloring-8"
"$EBR_PY" -m dataset.build_graph_coloring_dataset --nodes 10 --output-dir "$EBR_DATA/graph-coloring-10"
PYTHONPATH="$EBR_SRC" "$EBR_PY" "$BASELINE_CODE/ebr/maze_unique.py" "$EBR_DATA/maze-unique"
