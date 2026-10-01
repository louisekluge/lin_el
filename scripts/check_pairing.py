"""Are the MLDA correction terms paired at a common parameter value?

Eq. 21's correction Y_l = Q_l - Q_{l-1} is only a *difference* if both terms
are evaluated at the same theta. tinyDA's MLDA path exports proposal-paired
promoted chains, so some fraction of the exported pairs sit at different
theta, and each such pair contributes O(var Q) to the correction's variance
instead of O(var dQ) -- the difference between variance reduction and
variance inflation.

Measures that fraction directly, then recomputes the correction variances
using only the correctly paired samples and compares against the two-level
run, where the pairing is exact.

    python scripts/check_pairing.py
"""

from __future__ import annotations

import os
import sys
import warnings
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tinyDA as tda  # noqa: E402

from lin_el.models import QOI_NAMES  # noqa: E402
from lin_el.problem import build_problem  # noqa: E402
from lin_el.sampling import make_adaptive_metropolis  # noqa: E402

warnings.filterwarnings("ignore", message=".*qoi group is not defined.*")

p = argparse.ArgumentParser()
p.add_argument("--iterations", type=int, default=800)
p.add_argument("--subchain-length", type=int, default=5)
p.add_argument("--no-randomize", action="store_true")
p.add_argument("--no-aem", action="store_true")
p.add_argument("--seed", type=int, default=4242)
p.add_argument("--levels", default="0,1,2")
args = p.parse_args()

levels = [int(s) for s in args.levels.split(",")]
burnin = args.iterations // 5
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_NPZ = os.path.join(_ROOT, "results", "vr_l02_J5_rep00.npz")

print(f"levels={levels} J={args.subchain_length} "
      f"randomize={not args.no_randomize} aem={not args.no_aem} "
      f"iters={args.iterations} seed={args.seed}")

problem = build_problem(n_levels=max(levels) + 1)
posteriors = problem.posteriors(levels, adaptive_coarse=not args.no_aem)

np.random.seed(args.seed)
proposal = make_adaptive_metropolis(problem, laplace_level=None, verbose=False)

kwargs = dict(
    iterations=args.iterations,
    n_chains=1,
    initial_parameters=problem.map_estimate,
    subchain_length=args.subchain_length,
    randomize_subchain_length=not args.no_randomize,
)
if not args.no_aem:
    kwargs["adaptive_error_model"] = "state-independent"

chain = tda.sample(posteriors, proposal, **kwargs)

theta = tda.get_multilevel_inference_data(chain, attribute="parameters", burnin=burnin)
qoi = tda.get_multilevel_inference_data(chain, attribute="qoi", burnin=burnin)


def at(data, kind, level):
    return np.asarray(data[kind][f"level{level}_chain0"], dtype=float)


def match_fraction(a, b, shift=0):
    """Fraction of rows equal, with b shifted relative to a."""
    if shift > 0:
        a, b = a[shift:], b[:-shift]
    elif shift < 0:
        a, b = a[:shift], b[-shift:]
    return float(np.all(a == b, axis=1).mean())


print()
for level in range(len(levels) - 1):
    t_fine, t_coarse = at(theta, "chains", level + 1), at(theta, "promoted", level)
    q_fine, q_coarse = at(qoi, "chains", level + 1), at(qoi, "promoted", level)
    n = min(len(t_fine), len(t_coarse))
    if len(t_fine) != len(t_coarse):
        print(f"  note: Y_{level+1}{level} lengths differ "
              f"({len(t_fine)} vs {len(t_coarse)}), truncating to {n}")

    t_fine, t_coarse = t_fine[:n], t_coarse[:n]
    matched = np.all(t_fine == t_coarse, axis=1)
    y = q_fine[:n] - q_coarse[:n]

    print(f"Y_{level+1}{level}: {n} pairs, {matched.mean():.1%} at a common theta")
    # An off-by-one in the export would look like mispairing but is a
    # different bug with a different fix, so rule it out explicitly.
    offsets = {s: match_fraction(t_fine, t_coarse, s) for s in (-2, -1, 1, 2)}
    print("    match at offsets: "
          + "  ".join(f"{s:+d}={v:.1%}" for s, v in offsets.items()))
    for k, name in enumerate(QOI_NAMES):
        v_all = y[:, k].var(ddof=1)
        if matched.sum() > 1:
            v_ok = y[matched, k].var(ddof=1)
            print(f"    {name:16s} var(all)={v_all:.3e}  "
                  f"var(paired)={v_ok:.3e}  inflation={v_all / v_ok:6.2f}x")
        else:
            print(f"    {name:16s} var(all)={v_all:.3e}  "
                  "var(paired)=n/a (too few paired samples)")
    print()

if os.path.exists(REFERENCE_NPZ):
    ref = np.load(REFERENCE_NPZ)
    ref_y = ref["qoi__chain_l1"] - ref["qoi__promoted_l0"]
    print(f"two-level reference, mesh 0 -> mesh 2, state-paired "
          f"({os.path.basename(REFERENCE_NPZ)}):")
    for k, name in enumerate(QOI_NAMES):
        print(f"    {name:16s} var={ref_y[:, k].var(ddof=1):.3e}")
else:
    print(f"no two-level reference at {REFERENCE_NPZ}")