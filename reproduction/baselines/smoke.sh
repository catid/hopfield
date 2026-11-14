#!/bin/bash
# Consequential GPU checks; writes only to the dedicated smoke folder.
set -euo pipefail
source /home/catid/hopfield/reproduction/baselines/env.sh
: "${CUDA_VISIBLE_DEVICES:?set an allocated GPU explicitly}"
case "${1:?lattice or act}" in
lattice)
  export TORCHINDUCTOR_CACHE_DIR=$BASELINE_ROOT/smoke/lattice/.inductor
  exec "$TRMP_PY" -u "$BASELINE_CODE/trmp/lattice_fit.py" --seed 0 --steps 32 --schedule-steps 20000 --no-validate --log-every 8 --out "$BASELINE_ROOT/smoke/lattice"
  ;;
act)
  mkdir -p "$BASELINE_ROOT/smoke/act"
  export TORCHINDUCTOR_CACHE_DIR=$BASELINE_ROOT/smoke/act/.inductor
  exec "$TRMP_PY" -u "$BASELINE_CODE/trmp/check.py" --task sudoku-extreme --slots 8 --k 3 --out "$BASELINE_ROOT/smoke/act/check.json"
  ;;
*) exit 2;;
esac
