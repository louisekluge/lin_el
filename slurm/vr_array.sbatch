#!/bin/bash
#SBATCH --job-name=lin_el_vr
#SBATCH --partition=cpu_il
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --output=results/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

# Submit from the repo root: Slurm resolves --output against the submit
# directory and opens it before this script runs, so results/ must exist.
#
#   mkdir -p results
#   # test: one repetition of each hierarchy, short, to measure the cost
#   sbatch --array=0,6 --time=01:00:00 --export=ALL,ITERATIONS=500 slurm/vr_array.sbatch
#   # then the full sweep
#   sbatch slurm/vr_array.sbatch
cd "${SLURM_SUBMIT_DIR}" || exit 1

# dolfinx comes from the pixi environment (`pixi install` once on the login
# node). OMP_NUM_THREADS=1 keeps BLAS from grabbing every core on the node.
export PIXI_FROZEN=1
export OMP_NUM_THREADS=1
# Only if MPI_Finalize complains about the network provider, as it did on the
# laptop: every model uses MPI.COMM_SELF, so shared memory is all it needs.
# export FI_PROVIDER=shm

ITERATIONS=${ITERATIONS:-5000}

# Six repetitions each of the 2-level and 3-level hierarchies. Two levels run
# on DAChain, three on MLDAChain, both through lin_el/mlda_sampling.py, so
# both export the same structure and exact acceptance.
N_REPS=6
CFG=$(( SLURM_ARRAY_TASK_ID / N_REPS ))
REP=$(( SLURM_ARRAY_TASK_ID % N_REPS ))

case ${CFG} in
  0) LEVELS="0,2"   ;;
  1) LEVELS="0,1,2" ;;
  *) echo "unknown config ${CFG}"; exit 1 ;;
esac

echo "task=${SLURM_ARRAY_TASK_ID} levels=${LEVELS} rep=${REP} iterations=${ITERATIONS} host=$(hostname) started=$(date -Is)"
pixi run python scripts/vr_run.py \
    --rep "${REP}" \
    --levels "${LEVELS}" \
    --iterations "${ITERATIONS}" \
    --subchain-length 10
echo "finished=$(date -Is)"
