#!/bin/bash
# Fig. 9 repetitions: one chain of the notebook's MLDA run per array task.
# Uses the lin_el pixi environment (tinyDA v0.3-variance-reduction); the scripts do not import lin_el or
# dolfinx, only tinyDA, numpy, scipy, arviz and matplotlib.
#
# Submit from the repository root (the lin_el checkout). Slurm opens the log file before
# the script runs, so the results directory must exist first:
#
#   mkdir -p results/fig9
#   sbatch --array=0 --time=00:20:00 --export=ALL,ITERATIONS=150,BURNIN=50 slurm/fig9_array.sbatch   # smoke test
#   sbatch slurm/fig9_array.sbatch                                                                    # 20 repetitions
#
# then, once they are done (a few minutes, mostly the reference):
#
#   sbatch --wrap "pixi run python scripts/fig9_analyse.py" --time=01:00:00 --mem=4G \
#          --partition=cpu_il --output=results/fig9/analyse_%j.out
#
#SBATCH --job-name=fig9
#SBATCH --partition=cpu_il
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-19
#SBATCH --output=results/fig9/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# One thread per task: numpy would otherwise grab every core on the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIXI_FROZEN=1

ITERATIONS=${ITERATIONS:-25000}
BURNIN=${BURNIN:-5000}

echo "task=${SLURM_ARRAY_TASK_ID} iterations=${ITERATIONS} burnin=${BURNIN} host=$(hostname) started=$(date -Is)"
pixi run python scripts/fig9_sample.py --rep "${SLURM_ARRAY_TASK_ID}" \
    --iterations "${ITERATIONS}" --burnin "${BURNIN}"
echo "finished=$(date -Is)"
