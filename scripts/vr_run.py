"""One multilevel variance-reduction repetition on the laminate problem.

A single invocation = one independent repetition. The inverse problem is built
under a fixed seed so every repetition targets the same posterior; only the
sampler's randomness varies with --rep.

Saves a small .npz holding, per level, the QoI chain and the promoted QoI
chain, plus acceptance rates, ESS and timing. The heavy chain object is never
pickled.

    python scripts/vr_run.py --rep 0 --levels 0,2 --subchain-length 10

Requires the louisekluge/tinyDA fork: the VR extractors
(get_twolevel_inference_data / get_multilevel_inference_data) do not exist in
the released tinyda 0.9.21 on PyPI.
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

for name in ("get_twolevel_inference_data", "get_multilevel_inference_data"):
    if not hasattr(tda, name):
        raise SystemExit(
            f"tinyDA has no {name}: this needs the louisekluge/tinyDA fork, "
            "not the released tinyda on PyPI"
        )

print(f"rep={args.rep} levels={levels} J={args.subchain_length} "
      f"iters={args.iterations} burnin={burnin}")


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

kwargs = dict(
    iterations=args.iterations,
    n_chains=1,
    initial_parameters=problem.map_estimate,
    subchain_length=args.subchain_length,
    randomize_subchain_length=not args.no_randomize,
)
if not args.no_aem:
    kwargs["adaptive_error_model"] = "state-independent"

t0 = time.time()
chain = tda.sample(posteriors, proposal, **kwargs)
runtime = time.time() - t0


# ---------------------------------------------------------------------------
# Extraction
#
# tinyDA takes two different code paths and the VR extractors differ with them:
#   2 levels  -> sampler "DA",   get_twolevel_inference_data  (xarray Datasets)
#   3+ levels -> sampler "MLDA", get_multilevel_inference_data (plain arrays)
# Normalise both to {name: (n_draws, n_qoi) array}.
# ---------------------------------------------------------------------------

def _ds_to_array(dataset, chain_idx=0):
    """xarray Dataset with vars qoi_0..qoi_k -> (n_draws, n_qoi) array."""
    keys = sorted(dataset.data_vars, key=lambda s: int(s.rsplit("_", 1)[-1]))
    return np.column_stack([np.asarray(dataset[k].values)[chain_idx] for k in keys])


def extract_qoi(chain, burnin):
    sampler = chain["sampler"]
    if sampler == "DA":
        d = tda.get_twolevel_inference_data(chain, attribute="qoi", burnin=burnin)
        return {
            "chain_l0": _ds_to_array(d["chain_coarse"]),
            "chain_l1": _ds_to_array(d["chain_fine"]),
            "promoted_l0": _ds_to_array(d["promoted_coarse"]),
        }
    if sampler == "MLDA":
        d = tda.get_multilevel_inference_data(chain, attribute="qoi", burnin=burnin)
        out = {}
        for key, value in d["chains"].items():
            lvl = key.split("_")[0].replace("level", "")
            out[f"chain_l{lvl}"] = np.atleast_2d(np.asarray(value, dtype=float))
        for key, value in d["promoted"].items():
            lvl = key.split("_")[0].replace("level", "")
            out[f"promoted_l{lvl}"] = np.atleast_2d(np.asarray(value, dtype=float))
        return out
    raise RuntimeError(f"unexpected sampler {sampler!r}")


def chain_key(chain, level, n_levels):
    """Parameter-chain key: tinyDA names these three different ways."""
    if chain["sampler"] == "MH":
        return "chain_0"
    if chain["sampler"] == "DA":
        return "chain_fine_0" if level == n_levels - 1 else "chain_coarse_0"
    return f"chain_l{level}_0"


def acceptance_rates(chain, n_levels, burnin):
    """Post-burn-in acceptance per level, from state changes.

    sample() returns Link objects rather than acceptance flags, so a step
    counts as accepted iff the parameter vector moved. burnin is in fine-level
    iterations and is scaled to each level by the ratio of chain lengths.
    """
    as_array = lambda links: np.array([l.parameters for l in links], dtype=float)
    n_fine = len(chain[chain_key(chain, n_levels - 1, n_levels)])
    rates = {}
    for level in range(n_levels):
        samples = as_array(chain[chain_key(chain, level, n_levels)])
        cut = int(round(burnin * samples.shape[0] / n_fine))
        post = samples[cut:]
        if post.shape[0] < 2:
            continue
        moved = np.any(np.diff(post, axis=0) != 0, axis=1)
        rates[f"level{level}"] = float(moved.mean())
    return rates


def ess_fine(chain, n_levels, burnin):
    level = "fine" if chain["sampler"] == "DA" else n_levels - 1
    idata = tda.to_inference_data(chain, level=level, burnin=burnin,
                                  parameter_names=["E1", "E2"])
    ess = az.ess(idata)
    return {str(v): float(np.ravel(ess[v].values)[0]) for v in ess.data_vars}


n_levels = len(levels)
qoi = extract_qoi(chain, burnin)
rates = acceptance_rates(chain, n_levels, burnin)
ess = ess_fine(chain, n_levels, burnin)


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
        "qoi_names": list(QOI_NAMES),
        "acceptance": rates,
        "ess_fine": ess,
    })),
}
payload.update({f"qoi__{k}": v for k, v in qoi.items()})
payload.update({f"acc__{k}": np.array(v) for k, v in rates.items()})
payload.update({f"ess__{k}": np.array(v) for k, v in ess.items()})

outfile = os.path.join(outdir, f"vr_{tag}_rep{args.rep:02d}.npz")
np.savez(outfile, **payload)

print(f"\n{tag} rep {args.rep}: {runtime/60:.1f} min, sampler={chain['sampler']}")
for k, v in sorted(rates.items()):
    print(f"  acceptance  {k:12s} {v:.3f}")
for k, v in sorted(ess.items()):
    print(f"  ESS (fine)  {k:12s} {v:8.0f}")
for k, v in sorted(qoi.items()):
    print(f"  {k:14s} {v.shape}")
print(f"saved -> {outfile}")
