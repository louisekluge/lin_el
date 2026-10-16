"""Figure 9 from all repetitions: cells 21-23 of the notebook, with one chain per repetition.

    python scripts/fig9_analyse.py      # -> results/fig9/fig9.png, results/fig9/fig9_summary.txt

Needs results/fig9/reference.npz; runs fig9_reference.py first if it is missing.
"""

import glob
import json
import os
import runpy
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "results", "fig9")
outdir = sys.argv[1] if len(sys.argv) > 1 else OUTDIR

ref_file = os.path.join(outdir, "reference.npz")
if not os.path.exists(ref_file):
    print('no reference yet, computing it (a few minutes)')
    runpy.run_path(os.path.join(HERE, "fig9_reference.py"), run_name="__main__")
ref = np.load(ref_file)
Q_ref, Q_state_limit = float(ref['Q_ref']), float(ref['Q_state_limit'])

files = sorted(glob.glob(os.path.join(outdir, "fig9_rep*.npz")))
if not files:
    raise SystemExit(f'no fig9_rep*.npz in {outdir}')
runs = [np.load(f) for f in files]
metas = [json.loads(str(r['meta'])) for r in runs]
lengths = {len(r['Q_top']) for r in runs}
if len(lengths) > 1:
    print(f'warning: repetitions have different lengths {sorted(lengths)}; truncating to the shortest')
n_min = min(lengths)

# the same structure the notebook builds in cell 20, one entry per repetition ("chain")
my_terms = [{'Q_top': r['Q_top'], 'Q_0': r['Q_0'],
             'proposal': [r['Y_proposal_1'], r['Y_proposal_2']],
             'state': [r['Y_state_1'], r['Y_state_2']]} for r in runs]
n_chains = len(my_terms)
subchain_length = metas[0]['subchain_length']

print(f'{n_chains} repetitions, {n_min} fine samples each after burn-in, J = {subchain_length}')
print('fine-level acceptance: ' + '  '.join(f'{m["acceptance_fine"]:.2f}' for m in metas))
for key in ('Y_10', 'Y_21'):
    print(f'pairing {key}: mean {np.mean([m["pairing"][key] for m in metas]):.1%}')


# --- notebook cell 21: estimates and sampling error along the run ----------------------------

def ess(x):
    return float(az.ess(np.asarray(x)[None, :]))

def head(x, fraction):
    return x[:max(8, int(round(fraction * len(x))))]

def along_run(t, n_samples):
    n_top = len(t['Q_top'])
    rows = {k: [] for k in ('standard', 'proposal', 'state', 'se_standard', 'se_proposal', 'se_state')}
    for n in n_samples:
        f = n / n_top
        top = head(t['Q_top'], f)
        rows['standard'].append(top.mean())
        rows['se_standard'].append(np.sqrt(top.var() / ess(top)))
        q0 = head(t['Q_0'], f)
        for kind in ('proposal', 'state'):
            ys = [head(y, f) for y in t[kind]]
            rows[kind].append(q0.mean() + sum(y.mean() for y in ys))
            rows[f'se_{kind}'].append(np.sqrt(q0.var() / ess(q0) + sum(y.var() / ess(y) for y in ys)))
    return {k: np.array(v) for k, v in rows.items()}

n_top = n_min
n_samples = np.unique(np.geomspace(min(200, n_top // 2), n_top, 15).astype(int))
curves = [along_run(t, n_samples) for t in my_terms]

mean_se = {k: np.mean([c[f'se_{k}'] for c in curves], axis=0) for k in ('standard', 'proposal', 'state')}
estimates = {k: np.array([c[k] for c in curves]) for k in ('standard', 'proposal', 'state')}
rmse = {k: np.sqrt(np.mean((v - Q_ref)**2, axis=0)) for k, v in estimates.items()}
bias = {k: np.mean(v - Q_ref, axis=0) for k, v in estimates.items()}

lines = [f'{"N":>6}' + ''.join(f'{k + " SE":>15}' for k in mean_se) + ''.join(f'{k + " RMSE":>15}' for k in rmse)]
for j, n in enumerate(n_samples):
    lines.append(f'{n:6d}' + ''.join(f'{mean_se[k][j]:15.5f}' for k in mean_se) + ''.join(f'{rmse[k][j]:15.5f}' for k in rmse))


# --- notebook cell 22: figure ----------------------------------------------------------------

style = {'standard': ('k', 'Standard (finest level)'),
         'proposal': ('tab:blue', 'Variance reduction, proposal-paired (Eq. 21)'),
         'state': ('tab:red', 'Variance reduction, state-paired')}

fig, ax = plt.subplots(figsize=(8, 6))
for k, (color, label) in style.items():
    ax.loglog(n_samples, mean_se[k], '-', color=color, label=f'{label}: sampling error')
    ax.loglog(n_samples, rmse[k], 'o', color=color, mfc='none', label=f'{label}: RMSE')
ax.axhline(abs(Q_state_limit - Q_ref), color='tab:red', ls=':', lw=1,
           label='state-paired |bias| for $J \\to \\infty$')
ax.set_xlabel('Fine samples', fontsize=12)
ax.set_ylabel('Error in $\\mathbb{E}[Q]$', fontsize=12)
ax.set_title(f'{n_chains} chains, $J = {subchain_length}$', fontsize=12)
ax.legend(fontsize=9)
fig.savefig(os.path.join(outdir, 'fig9.png'), dpi=150, bbox_inches='tight')


# --- notebook cell 23: final estimates -------------------------------------------------------

lines.append('')
lines.append(f'reference                     {Q_ref:.5f}')
for k in ('standard', 'proposal', 'state'):
    final = estimates[k][:, -1]
    lines.append(f'{k:9s} final estimate  {final.mean():.5f} +- {final.std(ddof=1) / np.sqrt(n_chains):.5f}'
                 f'   bias {bias[k][-1]:+.5f}')

text = '\n'.join(lines)
print('\n' + text)
with open(os.path.join(outdir, 'fig9_summary.txt'), 'w') as fh:
    fh.write(text + '\n')
print(f'\nwrote {os.path.join(outdir, "fig9.png")} and fig9_summary.txt')
