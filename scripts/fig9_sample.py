"""One repetition of the notebook's MLDA run, longer, with its correction terms saved.

Cells 14-16 and 19-20 of the notebook, for one chain. Each Slurm array task runs one repetition; the
repetitions differ only in the sampler's random seed (the data are the notebook's, seed 987).

    python scripts/fig9_sample.py --rep 0                      # paper length: 25000 iterations, burn-in 5000
    python scripts/fig9_sample.py --rep 0 --iterations 150 --burnin 50    # smoke test, about a minute

Writes results/fig9/fig9_rep<NN>.npz with the time-ordered, burned-in arrays the analysis needs:
    Q_top              Q_2 on the fine chain
    Q_0                Q_0 on the coarsest chain
    Y_proposal_1, _2   proposal-paired corrections (Eq. 21)
    Y_state_1, _2      state-paired corrections
plus the pairing fractions and the run's settings. The full tinyDA chain is not stored: at the paper's
length it holds 2.5 million coarse links.
"""

import argparse
import json
import os
import time

import numpy as np

from fig9_setup import *  # noqa: F401,F403  (model, data, posteriors, MAP: notebook cells 1-12)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "results", "fig9")

p = argparse.ArgumentParser()
p.add_argument("--rep", type=int, required=True, help="repetition index; seeds the sampler")
p.add_argument("--iterations", type=int, default=25000, help="fine-level iterations, including burn-in")
p.add_argument("--burnin", type=int, default=5000)
p.add_argument("--subchain-length", type=int, default=10)
p.add_argument("--outdir", default=OUTDIR)
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
outfile = os.path.join(args.outdir, f"fig9_rep{args.rep:02d}.npz")

# Each repetition gets its own sampler stream. The data were drawn above under seed 987 and are the
# same in every repetition.
np.random.seed(10_000 + args.rep)


# --- notebook cell 14: proposal --------------------------------------------------------------

# random walk Metropolis
rwmh_cov = np.eye(2)
rmwh_scaling = 0.1
rwmh_adaptive = True
my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)


# --- notebook cells 15-16: sample ------------------------------------------------------------

iterations = args.iterations
burnin = args.burnin
n_chains = 1
subchain_length = args.subchain_length

print(f'rep {args.rep}: iterations={iterations} burnin={burnin} J={subchain_length}', flush=True)
t0 = time.time()
my_chains = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=n_chains,
                       initial_parameters=MAP[2], subchain_length=subchain_length,
                       randomize_subchain_length=True)
runtime = time.time() - t0


# --- notebook cell 19: extract ---------------------------------------------------------------

mlqoi = tda.get_multilevel_inference_data(my_chains, attribute='qoi', burnin=burnin)
mlpar = tda.get_multilevel_inference_data(my_chains, attribute='parameters', burnin=burnin)


# --- notebook cell 20: correction terms ------------------------------------------------------

def state_paired_coarse_qoi(fine_parameters, promoted_parameters, promoted_qoi, coarse_posterior):
    # Q_{l-1} at the fine states, looked up among the promoted samples
    lookup = {p.tobytes(): q for p, q in zip(promoted_parameters, promoted_qoi)}
    out = np.empty(len(fine_parameters))
    for n, p in enumerate(fine_parameters):
        key = p.tobytes()
        if key not in lookup:
            lookup[key] = coarse_posterior.create_link(p).qoi
        out[n] = lookup[key]
    return out

def terms(i):
    # the coarsest-level QoI and the two kinds of correction for chain i
    out = {'Q_top': mlqoi['chains'][f'level2_chain{i}'][:, 0],
           'Q_0': mlqoi['chains'][f'level0_chain{i}'][:, 0],
           'proposal': [], 'state': []}
    for l in (1, 2):
        fine_q = mlqoi['chains'][f'level{l}_chain{i}'][:, 0]
        prom_q = mlqoi['promoted'][f'level{l-1}_chain{i}'][:, 0]
        fine_p = mlpar['chains'][f'level{l}_chain{i}']
        prom_p = mlpar['promoted'][f'level{l-1}_chain{i}']

        out['proposal'].append(fine_q - prom_q)

        coarse_q = state_paired_coarse_qoi(fine_p, prom_p, prom_q, my_posteriors[l-1])
        out['state'].append(fine_q - coarse_q)
    return out

t = terms(0)

pairs = {}
for l in (1, 2):
    pairs[f'Y_{l}{l-1}'] = float(np.mean(np.all(mlpar['chains'][f'level{l}_chain0'] == mlpar['promoted'][f'level{l-1}_chain0'], axis=1)))
    print(f'Y_{l}{l-1}: {pairs[f"Y_{l}{l-1}"]:.1%} of proposal-paired pairs sit at a common parameter value')

# Acceptance on the fine level, from state changes. tinyDA does not export the acceptance flags; on the
# top level this undercounts only the rare accepted proposal equal to the current state.
fine = mlpar['chains']['level2_chain0']
acceptance_fine = float(np.mean(np.any(np.diff(fine, axis=0) != 0, axis=1)))
print(f'fine-level acceptance (state changes, after burn-in): {acceptance_fine:.3f}')


# --- save ------------------------------------------------------------------------------------

meta = {'rep': args.rep, 'iterations': iterations, 'burnin': burnin, 'subchain_length': subchain_length,
        'runtime_seconds': runtime, 'pairing': pairs, 'acceptance_fine': acceptance_fine,
        'MAP': [m.tolist() for m in MAP], 'free': list(free)}
np.savez_compressed(outfile, Q_top=t['Q_top'], Q_0=t['Q_0'],
                    Y_proposal_1=t['proposal'][0], Y_proposal_2=t['proposal'][1],
                    Y_state_1=t['state'][0], Y_state_2=t['state'][1],
                    meta=np.array(json.dumps(meta)))
print(f'\nrep {args.rep}: {runtime / 60:.1f} min sampling, saved -> {outfile}')
