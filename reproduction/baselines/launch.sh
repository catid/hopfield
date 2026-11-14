#!/bin/bash
# Run only after a GPU has been allocated by the experiment coordinator.
# Author shell scripts run directly; SBATCH headers are comments outside Slurm.
set -euo pipefail
source /home/catid/hopfield/reproduction/baselines/env.sh
kind=${1:?method}; task=${2:?task}; seed=${3:?seed}
[[ $seed =~ ^[012]$ ]] || { echo 'main-paper seeds are 0, 1, 2'; exit 2; }
: "${CUDA_VISIBLE_DEVICES:?set an allocated GPU explicitly}"
export CODE=$BASELINE_CODE SLURM_SUBMIT_DIR=$BASELINE_CODE SLURM_JOB_ID="local-$kind-$task-s$seed"
export SLURM_GPUS_ON_NODE=1 SLURM_RESTART_COUNT=0
cd "$BASELINE_CODE"
case "$kind" in
trmp-lattice) [[ $task == sudoku ]] || exit 2; exec bash trmp/lattice_fit.sbatch "$seed" ;;
trmp-act) [[ $task == sudoku-extreme || $task == maze-unique ]] || exit 2
  if [[ $task == maze-unique && ! -f ${TRMP_MAZE_VALIDATION:-$BASELINE_ROOT/experiments/runs/TRMP-audit/generated-validation}/final.json ]]; then
    echo 'blocked: original TRMP generated Maze validation set is absent'; exit 3
  fi
  exec bash trmp/fit.sbatch "$task" "$seed" ;;
trm) [[ $task == sudoku || $task == mazeunique ]] || exit 2; exec bash trm/fit.sbatch "$task" "$seed" ;;
hrm) [[ $task == sudoku || $task == mazeunique ]] || exit 2
  "$HRM_PY" -c 'import adam_atan2_backend; import flash_attn' || { echo 'blocked: native HRM dependencies not validated'; exit 3; }
  if [[ $task == sudoku ]]; then export HRM_EPOCHS=40000; fi
  exec bash hrm/fit.sbatch "$task" "$seed" ;;
gram) exec bash gram/fit.sbatch "$task" "$seed" ;;
ebr) exec bash ebr/fit.sbatch "$task" "$seed" ;;
eqr) [[ $task == sudoku || $task == maze-repo || $task == maze-paper ]] || exit 2; exec bash eqr/fit.sbatch "$task" "$seed" ;;
frm) echo 'blocked: released source revision and original campaign/data plan unavailable'; exit 3 ;;
*) echo "unknown method $kind"; exit 2 ;;
esac
