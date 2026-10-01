#!/bin/bash
#SBATCH --job-name=lin_el_vr
#SBATCH --partition=cpu_il
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
#SBATCH --output=%x_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

# Submit from the repo root so ${SLURM_SUBMIT_DIR} resolves there; the runner
# writes into <repo>/results/ regardless of the working directory.
cd "${SLURM_SUBMIT_DIR}" || exit 1

# dolfinx comes from the pixi environment, not a module + venv: FEniCSx is not
# pip-installable, so the predator-prey recipe (module load python; venv) does
# not transfer. `pixi install` once on the login node, then:
export PIXI_FROZEN=1
export OMP_NUM_THREADS=1

# Six repetitions each of the 2-level and 3-level hierarchies. The two use
# different tinyDA code paths (DA vs MLDA) and -- see README -- different
# sample pairings, so they are separate experiments, not one sweep.
N_REPS=6
CFG=$(( SLURM_ARRAY_TASK_ID / N_REPS ))
REP=$(( SLURM_ARRAY_TASK_ID % N_REPS ))

case ${CFG} in
  0) LEVELS="0,2"   ;;
  1) LEVELS="0,1,2" ;;
  *) echo "unknown config ${CFG}"; exit 1 ;;
esac

pixi run python scripts/vr_run.py \
    --rep "${REP}" \
    --levels "${LEVELS}" \
    --iterations 5000 \
    --subchain-length 10
