#!/bin/bash
# Laminate (linear elasticity) hierarchy sweep: 13 configurations x 3 repetitions, through scripts/vr_run.py.
#
#   hierarchies 0,1,2 and 0,2  x  J = 2, 5, 10  x  randomised / fixed promotion, all with AEM   (12)
#   + 0,1,2, J = 10, randomised, without AEM                                                    ( 1)
#
# 20000 fine iterations, of which 4000 burn-in (vr_run.py uses iterations // 5). Task t runs config t // 3,
# repetition t % 3; `bash slurm/el_sweep.sbatch --list` prints the table without submitting anything.
#
# Submit from the lin_el root; the log directory must exist first:
#
#   mkdir -p results/el_sweep
#   # smoke test: the most expensive config of each hierarchy, short, to measure the cost per iteration
#   sbatch --array=12,30 --time=01:00:00 --export=ALL,ITERATIONS=200 slurm/el_sweep.sbatch
#   # full sweep (adjust --time from the smoke test, see below)
#   sbatch slurm/el_sweep.sbatch
#
#SBATCH --job-name=el_sweep
#SBATCH --partition=cpu_il
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-38
#SBATCH --output=results/el_sweep/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

set -euo pipefail

# name      levels  J   randomise  aem
CONFIGS=(
  "l012_J2_rand    0,1,2  2   1  1"
  "l012_J2_fixed   0,1,2  2   0  1"
  "l012_J5_rand    0,1,2  5   1  1"
  "l012_J5_fixed   0,1,2  5   0  1"
  "l012_J10_rand   0,1,2  10  1  1"
  "l012_J10_fixed  0,1,2  10  0  1"
  "l02_J2_rand     0,2    2   1  1"
  "l02_J2_fixed    0,2    2   0  1"
  "l02_J5_rand     0,2    5   1  1"
  "l02_J5_fixed    0,2    5   0  1"
  "l02_J10_rand    0,2    10  1  1"
  "l02_J10_fixed   0,2    10  0  1"
  "l012_J10_rand_noaem  0,1,2  10  1  0"
)
N_REPS=3

if [[ "${1:-}" == "--list" ]]; then
  for i in "${!CONFIGS[@]}"; do
    for r in $(seq 0 $((N_REPS - 1))); do echo "task $((i * N_REPS + r)): ${CONFIGS[$i]%% *}  rep ${r}"; done
  done
  exit 0
fi

cd "${SLURM_SUBMIT_DIR}"

# pixi is installed per user; batch jobs don't read ~/.bashrc, so put it on PATH here.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "pixi not found on PATH=$PATH"; exit 1; }
export PIXI_FROZEN=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Only if MPI_Finalize complains about the network provider: every model uses MPI.COMM_SELF.
# export FI_PROVIDER=shm

CFG=$(( SLURM_ARRAY_TASK_ID / N_REPS ))
REP=$(( SLURM_ARRAY_TASK_ID % N_REPS ))
read -r NAME LEVELS J RAND AEM <<< "${CONFIGS[$CFG]}"
ITERATIONS=${ITERATIONS:-20000}

ARGS=(--rep "${REP}" --levels "${LEVELS}" --subchain-length "${J}" --iterations "${ITERATIONS}"
      --tag "${NAME}" --outdir results/el_sweep)
[[ "${RAND}" == 0 ]] && ARGS+=(--no-randomize)
[[ "${AEM}" == 0 ]] && ARGS+=(--no-aem)

echo "task=${SLURM_ARRAY_TASK_ID} config=${NAME} rep=${REP} levels=${LEVELS} J=${J} rand=${RAND} aem=${AEM}" \
     "iterations=${ITERATIONS} host=$(hostname) started=$(date -Is)"
pixi run python scripts/vr_run.py "${ARGS[@]}"
echo "finished=$(date -Is)"

# Output: results/el_sweep/vr_<config>_rep<NN>.npz      summary (aligned QoI per level, acceptance, pairing, ESS)
#         results/el_sweep/chains_<config>_rep<NN>.npz  full chains, written ten times during the run
#
# Cost. One fine iteration needs J^2 + J + 1 model solves on three levels and J + 1 on two:
#   0,1,2:  J=2 -> 7,  J=5 -> 31,  J=10 -> 111 solves      0,2:  J=2 -> 3,  J=5 -> 6,  J=10 -> 11
# The smoke test prints minutes per 200 iterations; multiply by 100 for the full run. The three-level J=10
# configurations are by far the most expensive: if they need more than 24 h, raise --time for them only, e.g.
#   sbatch --array=12-17,36-38 --time=48:00:00 slurm/el_sweep.sbatch
# and submit the rest with --array=0-11,18-35.
#
# Afterwards: pixi run python scripts/analyse_el_sweep.py
