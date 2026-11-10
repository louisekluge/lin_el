#!/bin/bash
# Predator-prey sampler sweep: 9 hierarchy configurations x 20 repetitions (scripts/mlda_diagnostics.py).
# Every task keeps its full chain; total output ~8 GB in results/diagnostics/.
#
# Submit from the lin_el root. Slurm opens the log file before the script runs, so the directory must
# exist first:
#
#   mkdir -p results/diagnostics
#   sbatch --array=0,120,160 --time=00:30:00 --export=ALL,ITERATIONS=200 slurm/diag_array.sbatch   # smoke test
#   sbatch slurm/diag_array.sbatch                                                                   # full sweep
#
# Task t runs config t // 20, repetition t % 20 (configs in the order of CONFIGS in the script:
# 0 base, 1 nsub1, 2 nsub2, 3 nsub5, 4 nsub20, 5 fixed_nsub, 6 two_level_hi, 7 two_level_lo, 8 single).
#
#SBATCH --job-name=mlda_diag
#SBATCH --partition=cpu_il
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-179%20
#SBATCH --output=results/diagnostics/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=louise.kluge@posteo.net

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# pixi is installed per user; batch jobs don't read ~/.bashrc, so put it on PATH here.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "pixi not found on PATH=$PATH"; exit 1; }
export PIXI_FROZEN=true

# One thread per task. numpy and scipy each default to grabbing every core on the node; with 20
# concurrent array tasks that oversubscribes it and every task slows down together.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXTRA=()
if [[ -n "${ITERATIONS:-}" ]]; then EXTRA+=(--iterations "${ITERATIONS}"); fi

echo "task=${SLURM_ARRAY_TASK_ID} host=$(hostname) started=$(date -Is)"
pixi run python scripts/mlda_diagnostics.py --task-id "${SLURM_ARRAY_TASK_ID}" "${EXTRA[@]}"
echo "finished=$(date -Is)"

# Output: results/diagnostics/diag_<config>_rep<NN>.npz    summaries, a few kB
#         results/diagnostics/chains_<config>_rep<NN>.npz  full chains
#         results/diagnostics/mlda_diag_<jobid>_<task>.out this task's log
#
# Sizing, measured: 30-40 compressed bytes per link; a three-level run stores N(2 + 2J + J^2) links,
# two-level N(2 + J), single level N.
#
#   base/fixed_nsub  J=10  20000   98 MB each    21 min
#   nsub20           J=20   8000  142 MB         30 min
#   nsub5            J= 5  20000   30 MB          7 min
#   two_level_*      J=10  20000   10 MB each   2-4 min
#   nsub1/nsub2/single          4-8 MB each      1-2 min
#
# ~400 MB and ~89 min per repetition of the whole sweep, so ~8 GB and ~30 CPU-hours at 20 repetitions,
# roughly doubling on a cluster core.
#
# Afterwards:
#   pixi run python scripts/analyse_diagnostics.py               # efficiency, pairing, VR, bias
#   pixi run python scripts/plot_diagnostics.py --config base    # one configuration in detail
