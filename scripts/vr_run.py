"""One multilevel variance-reduction repetition on the laminate problem.

A single invocation = one independent repetition. The inverse problem is built
under a fixed seed so every repetition targets the same posterior; only the
sampler's randomness varies with --rep.

Samples through lin_el/mlda_sampling.py rather than tda.sample, which gives

  * exact acceptance per level, from the sampler's own flags, instead of
    inferring it from state changes (that undercounts on fine levels and
    overcounts on coarse ones)
  * two levels run on DAChain, as with tda.sample (--two-level-chain mlda
    switches to MLDAChain, whose two-level setup crashes with the adaptive
    error model in tinyDA v0.3); three levels run on MLDAChain. Both export
    the same keys, so the extraction below handles them alike
  * a progress line every 1% and periodic checkpoints of the full chains, so
    a job killed at the walltime leaves its chains behind

Writes, under results/:
  vr_<tag>_rep<NN>.npz         QoI per level, aligned and burned in, as
                               plot_vr.py reads it; acceptance, pairing, ESS
  chains_<tag>_rep<NN>.npz     full chains (parameters, QoI, stats) for every
                               level's chain and promoted list, no burn-in,
                               for comparing estimator formulations later

    python scripts/vr_run.py --rep 0 --levels 0,2 --subchain-length 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arviz as az  # noqa: E402
import tinyDA as tda  # noqa: E402

from lin_el.models import QOI_NAMES  # noqa: E402
from lin_el.problem import build_problem  # noqa: E402
from lin_el.sampling import make_adaptive_metropolis  # noqa: E402
from lin_el.mlda_sampling import (acceptance_rates, make_checkpointer,  # noqa: E402
                                  pairing_fractions, sample_multilevel)

warnings.filterwarnings("ignore", message=".*qoi group is not defined.*")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--rep", type=int, required=True, help="repetition index; seeds the sampler only")
p.add_argument("--levels", default="0,1,2", help="comma-separated level indices, coarse -> fine")
p.add_argument("--iterations", type=int, default=5000, help="fine-level iterations")
p.add_argument("--subchain-length", type=int, default=10)
p.add_argument("--no-randomize", action="store_true",
               help="fixed subchain length (randomised is what Eq. 21 assumes)")
p.add_argument("--no-aem", action="store_true", help="disable the adaptive error model")
p.add_argument("--c0", default=None,
               help="comma-separated per-parameter proposal sd; overrides the "
                    "Laplace estimate")
p.add_argument("--no-laplace", action="store_true",
               help="start adaptive Metropolis from the identity instead")
p.add_argument("--two-level-chain", default="da", choices=("da", "mlda"),
               help="tinyDA chain for a two-level run (ignored for 3+ levels)")
p.add_argument("--no-chains", action="store_true", help="skip the full-chain file")
p.add_argument("--tag", default="", help="label for the output filename")
p.add_argument("--outdir", default=_DEFAULT_OUTDIR)
args = p.parse_args()

levels = [int(s) for s in args.levels.split(",")]
if len(levels) < 2:
    raise SystemExit("variance reduction needs at least two levels")
burnin = args.iterations // 5
tag = args.tag or "l" + "".join(str(i) for i in levels) + f"_J{args.subchain_length}"

# Fail fast: prove the output path works before spending an hour sampling.
outdir = os.path.expanduser(args.outdir)
os.makedirs(outdir, exist_ok=True)
probe = os.path.join(outdir, f".probe_{tag}_{args.rep}")
with open(probe, "w") as fh:
    fh.write("ok")
os.remove(probe)

stem = os.path.join(outdir, f"vr_{tag}_rep{args.rep:02d}")
# not vr_*: plot_vr.py globs vr_*.npz and expects every match to be a summary
chainfile = os.path.join(outdir, f"chains_{tag}_rep{args.rep:02d}.npz")
print(f"rep={args.rep} levels={levels} J={args.subchain_length} "
      f"iters={args.iterations} burnin={burnin}", flush=True)


# ---------------------------------------------------------------------------
# Problem (fixed seed) and sampler (per-repetition seed)
# ---------------------------------------------------------------------------

problem = build_problem(n_levels=max(levels) + 1)
posteriors = problem.posteriors(levels, adaptive_coarse=not args.no_aem)

np.random.seed(4242 + args.rep)
c0 = (np.diag([float(s) ** 2 for s in args.c0.split(",")])
      if args.c0 else None)
proposal = make_adaptive_metropolis(
    problem, c0=c0, laplace_level=None if args.no_laplace else 0
)

callback = None
if not args.no_chains:
    # ten checkpoints over the run; each write is O(chain length)
    callback = make_checkpointer(chainfile,
                                 every=max(1, args.iterations // 10))

t0 = time.time()
chain = sample_multilevel(
    posteriors, proposal,
    iterations=args.iterations,
    subchain_lengths=args.subchain_length,
    randomize_subchain_length=not args.no_randomize,
    initial_parameters=problem.map_estimate,
    adaptive_error_model=None if args.no_aem else "state-independent",
    n_chains=1,
    callback=callback,
    two_level_chain=args.two_level_chain,
)
runtime = time.time() - t0
if callback is not None:
    callback(chain, args.iterations, args.iterations)     # final state


# ---------------------------------------------------------------------------
# Extraction: aligned, burned-in QoI per level
#
# plot_vr.py pairs qoi__chain_l{k} row by row with qoi__promoted_l{k-1}, so
# the alignment is done here, once. The top-level chain carries its initial
# state, which no proposal produced, so under MLDAChain it is one longer than
# its promoted partner and loses its first entry. DAChain seeds its promoted
# list with that state too, so there the lengths already agree. Aligning by
# length covers both; lower levels pair directly. Burn-in is
# scaled to each level by chain length, so every term is cut at the same
# point in the run.
# ---------------------------------------------------------------------------

n_levels = len(levels)
top = n_levels - 1


def as_array(links, attribute):
    return np.array([np.atleast_1d(getattr(l, attribute)) for l in links],
                    dtype=np.float64)


qoi_chain = {l: as_array(chain[f"chain_l{l}_0"], "qoi") for l in range(n_levels)}
qoi_prom = {l: as_array(chain[f"promoted_l{l}_0"], "qoi") for l in range(top)}
qoi_chain[top] = qoi_chain[top][len(qoi_chain[top]) - len(qoi_prom[top - 1]):]
n_top = len(qoi_chain[top])

qoi = {}
for l in range(n_levels):
    arr = qoi_chain[l]
    qoi[f"chain_l{l}"] = arr[int(round(burnin * len(arr) / n_top)):]
for l in range(top):
    arr = qoi_prom[l]
    qoi[f"promoted_l{l}"] = arr[int(round(burnin * len(arr) / n_top)):]

rates = acceptance_rates(chain, burnin)
pairing = pairing_fractions(chain, burnin)

theta_top = as_array(chain[f"chain_l{top}_0"], "parameters")[1 + burnin:]
ess = {name: float(az.ess(az.convert_to_dataset(theta_top[None, :, j]))["x"].item())
       for j, name in enumerate(("E1", "E2"))}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

payload = {
    "runtime_seconds": np.array(runtime),
    "map_estimate": problem.map_estimate,
    "qoi_true": problem.qoi_true,
    "data": problem.data,
    "meta": np.array(json.dumps({
        "rep": args.rep,
        "levels": levels,
        "subchain_length": args.subchain_length,
        "randomize": not args.no_randomize,
        "aem": not args.no_aem,
        "iterations": args.iterations,
        "burnin": burnin,
        "sampler": chain["sampler"],
        "two_level_chain": args.two_level_chain if n_levels == 2 else None,
        "qoi_names": list(QOI_NAMES),
        "acceptance": rates,
        "acceptance_method": "exact",
        "pairing": pairing,
        "ess_fine": ess,
        "chain_file": None if args.no_chains else os.path.basename(chainfile),
    })),
}
payload.update({f"qoi__{k}": v for k, v in qoi.items()})
payload.update({f"acc__{k}": np.array(v) for k, v in rates.items()})
payload.update({f"pair__{k}": np.array(v) for k, v in pairing.items()})
payload.update({f"ess__{k}": np.array(v) for k, v in ess.items()})

np.savez(stem + ".npz", **payload)

print(f"\n{tag} rep {args.rep}: {runtime/60:.1f} min, sampler={chain['sampler']}")
for k, v in sorted(rates.items()):
    print(f"  acceptance  {k:12s} {v:.3f}")
for k, v in sorted(pairing.items()):
    print(f"  pairing     {k:12s} {v:.3f}")
for k, v in sorted(ess.items()):
    print(f"  ESS (fine)  {k:12s} {v:8.0f}")
for k, v in sorted(qoi.items()):
    print(f"  {k:14s} {v.shape}")
print(f"saved -> {stem}.npz")
if not args.no_chains:
    print(f"         {chainfile}")
