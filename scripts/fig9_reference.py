"""Reference solution by quadrature: cell 13 of the notebook, saved to a file.

    python scripts/fig9_reference.py            # -> results/fig9/reference.npz

Takes a few minutes. fig9_analyse.py runs it itself if the file is missing.
"""

import os

import numpy as np

from fig9_setup import *  # noqa: F401,F403  (model, data, posteriors, MAP: notebook cells 1-12)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "results", "fig9")


# --- notebook cell 13: reference solution ----------------------------------------------------

if "CI" in os.environ:
    n_coarse, n_fine = 21, 31
else:
    n_coarse, n_fine = 41, 91

def evaluate(thetas, posteriors):
    # log-posterior and QoI of every posterior at every point
    links = [[p.create_link(t) for p in posteriors] for t in thetas]
    logpost = np.array([[link.posterior for link in row] for row in links])
    qoi = np.array([[link.qoi for link in row] for row in links])
    return np.where(np.isfinite(logpost), logpost, -np.inf), qoi

def grid(lo, hi, n):
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
    return np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(lo), -1).T

R = np.zeros((3, 3))         # R[l, k] = E_pi_l[Q_k]
theta_mean = np.zeros((3, len(free)))
theta_cov = np.zeros((3, len(free), len(free)))
edge_drops = np.zeros(3)

for l in range(3):
    # coarse passes: start from a wide box around the MAP, then shrink it to the bounding box of
    # everything within 30 log-units of the maximum (about 7.7 standard deviations for a Gaussian)
    lo, hi = MAP[l] - 1.0, MAP[l] + 1.0
    for _ in range(3):
        thetas = grid(lo, hi, n_coarse)
        lp = np.array([my_posteriors[l].create_link(t).posterior for t in thetas])
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        keep = thetas[lp > lp.max() - 30]
        step = (hi - lo) / (n_coarse - 1)
        lo, hi = keep.min(axis=0) - step, keep.max(axis=0) + step

    # fine pass on the final box
    thetas = grid(lo, hi, n_fine)
    logpost, qoi = evaluate(thetas, my_posteriors)

    # self-normalised weights for level l
    w = np.exp(logpost[:, l] - logpost[:, l].max())
    w /= w.sum()
    R[l] = w @ np.nan_to_num(qoi)
    theta_mean[l] = w @ thetas
    dev = thetas - theta_mean[l]
    theta_cov[l] = (w[:, None] * dev).T @ dev

    # how far the posterior has fallen at the edge of the box
    on_edge = np.any((thetas == lo) | (thetas == hi), axis=1)
    edge_drop = logpost[:, l].max() - logpost[on_edge, l].max()
    edge_drops[l] = edge_drop
    print(f'level {l}: mean ' + '  '.join(f'log {names[i]}={v:.4f}' for i, v in zip(free, theta_mean[l]))
          + f'   sd ' + '  '.join(f'{v:.4f}' for v in np.sqrt(np.diag(theta_cov[l])))
          + f'   edge drop {edge_drop:.0f}')

Q_ref = R[2, 2]
Q_state_limit = R[2, 2] + (R[0, 0] - R[1, 0]) + (R[1, 1] - R[2, 1])
print(f'\nR[l, k] = E_pi_l[Q_k]:\n{R}')
print(f'\nreference E_pi_2[Q_2]                    = {Q_ref:.6f}')
print(f'state-paired limit for J -> infinity       = {Q_state_limit:.6f}   (bias {Q_state_limit - Q_ref:+.4f})')


# --- save ------------------------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)
outfile = os.path.join(OUTDIR, "reference.npz")
np.savez(outfile, R=R, Q_ref=Q_ref, Q_state_limit=Q_state_limit, theta_mean=theta_mean,
         theta_cov=theta_cov, edge_drops=edge_drops, n_coarse=n_coarse, n_fine=n_fine)
print(f'\nsaved -> {outfile}')
