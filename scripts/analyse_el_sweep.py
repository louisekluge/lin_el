"""Compare the configurations of the laminate hierarchy sweep (slurm/el_sweep.sbatch).

Reads the vr_<config>_rep<NN>.npz summaries that scripts/vr_run.py writes, not the chain files, and prints
four tables in the spirit of analyse_diagnostics.py:

1. Efficiency: fine-level ESS per hour, using the worse of E1 and E2 (a chain is only as converged as its
   worst-mixing direction), relative to the configuration given by --baseline.
2. Acceptance per level (exact, from the sampler's flags) and pairing: the fraction of proposal-paired
   corrections whose two terms sit at the same parameter value.
3. Variance reduction per QoI: the standard error of the plain fine-level mean divided by that of the
   proposal-paired multilevel estimator, both from within-run var/ESS ("predicted", as the lines of Fig. 9).
   With 3 repetitions the across-repetition spread is too noisy to stand in for the error; it is printed
   alongside only as a sanity check.
4. Consistency: each configuration's multilevel and fine-level estimates against the consensus of the
   fine-level estimates of all configurations, in units of their combined standard error. All
   configurations target the same posterior. Fixed (non-randomised) promotion is expected to bias the
   multilevel estimator; randomised promotion should not.

    python scripts/analyse_el_sweep.py
    python scripts/analyse_el_sweep.py --baseline l02_J10_rand
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import arviz as az  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(_REPO_ROOT, "results", "el_sweep"))
p.add_argument("--baseline", default="l012_J10_rand", help="configuration the efficiency is relative to")
p.add_argument("--out", default=None, help="default: <indir>/el_sweep.png")
p.add_argument("--csv", default=None, help="default: <indir>/el_sweep.csv")
args = p.parse_args()

indir = os.path.expanduser(args.indir)
out_png = args.out or os.path.join(indir, "el_sweep.png")
out_csv = args.csv or os.path.join(indir, "el_sweep.csv")

files = sorted(glob.glob(os.path.join(indir, "vr_*_rep*.npz")))
if not files:
    raise SystemExit(f"no vr_*_rep*.npz in {indir}")


def ess(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8 or np.ptp(x) == 0:
        return float(x.size)
    return float(az.ess(x[None, :]))


def summarise(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    n_levels = len(meta["levels"])
    top = n_levels - 1
    fine = d[f"qoi__chain_l{top}"]
    q0 = d["qoi__chain_l0"]
    ys = [d[f"qoi__chain_l{l}"] - d[f"qoi__promoted_l{l - 1}"] for l in range(1, n_levels)]
    n_q = fine.shape[1]
    out = {
        "meta": meta,
        "runtime_h": float(d["runtime_seconds"]) / 3600,
        "ess_min": min(meta["ess_fine"].values()),
        "standard": fine.mean(axis=0),
        "multilevel": q0.mean(axis=0) + sum(y.mean(axis=0) for y in ys),
        "se_standard": np.array([np.sqrt(fine[:, k].var() / ess(fine[:, k])) for k in range(n_q)]),
        "se_multilevel": np.array([np.sqrt(q0[:, k].var() / ess(q0[:, k])
                                           + sum(y[:, k].var() / ess(y[:, k]) for y in ys))
                                   for k in range(n_q)]),
    }
    return out


runs = defaultdict(list)
for f in files:
    name = re.match(r"vr_(.+)_rep\d+\.npz", os.path.basename(f)).group(1)
    runs[name].append(summarise(f))

configs = sorted(runs, key=lambda n: (runs[n][0]["meta"]["levels"] != [0, 1, 2], runs[n][0]["meta"]["subchain_length"],
                                      not runs[n][0]["meta"]["randomize"], not runs[n][0]["meta"]["aem"]))
qoi_names = runs[configs[0]][0]["meta"]["qoi_names"]
n_q = len(qoi_names)

print(f"{len(files)} runs in {len(configs)} configurations")
for name in configs:
    if len(runs[name]) < 3:
        print(f"  note: {name} has only {len(runs[name])} repetition(s)")

rows = []


# --- 1. efficiency ---------------------------------------------------------------------------

print("\n=== efficiency: fine-level ESS (worse of E1, E2) per hour " + "=" * 34)
eff = {n: np.array([r["ess_min"] / r["runtime_h"] for r in runs[n]]) for n in configs}
base = eff[args.baseline].mean() if args.baseline in eff else np.nan
print(f"{'config':22s} {'reps':>4s} {'hours':>7s} {'ESS min':>8s} {'ESS/h':>9s} {'vs ' + args.baseline:>22s}")
for n in configs:
    h = np.mean([r["runtime_h"] for r in runs[n]])
    e = np.mean([r["ess_min"] for r in runs[n]])
    print(f"{n:22s} {len(runs[n]):4d} {h:7.2f} {e:8.0f} {eff[n].mean():9.1f} {eff[n].mean() / base:22.2f}")


# --- 2. acceptance and pairing ---------------------------------------------------------------

print("\n=== acceptance per level and pairing " + "=" * 55)
for n in configs:
    acc = defaultdict(list)
    pair = defaultdict(list)
    for r in runs[n]:
        for k, v in r["meta"]["acceptance"].items():
            acc[k].append(v)
        for k, v in r["meta"]["pairing"].items():
            pair[k].append(v)
    print(f"{n:22s} "
          + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(acc.items()))
          + "   pairing " + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(pair.items())))


# --- 3. variance reduction -------------------------------------------------------------------

print("\n=== variance reduction: SE(standard) / SE(multilevel, proposal-paired) " + "=" * 21)
print(f"{'config':22s} " + "".join(f"{q:>18s}" for q in qoi_names) + "   (spread ratio, 3 reps)")
vr = {}
for n in configs:
    pred = np.mean([r["se_standard"] / r["se_multilevel"] for r in runs[n]], axis=0)
    std_est = np.array([r["standard"] for r in runs[n]])
    ml_est = np.array([r["multilevel"] for r in runs[n]])
    spread = (std_est.std(axis=0, ddof=1) / ml_est.std(axis=0, ddof=1)) if len(runs[n]) > 1 else np.full(n_q, np.nan)
    vr[n] = pred
    print(f"{n:22s} " + "".join(f"{v:18.2f}" for v in pred)
          + "   (" + " ".join(f"{s:.2f}" for s in spread) + ")")


# --- 4. consistency --------------------------------------------------------------------------

print("\n=== consistency: estimate - consensus, in combined standard errors " + "=" * 25)
# Consensus = plain mean over configurations of their fine-level estimates. Inverse-variance weighting would
# let one sticky run with an underestimated SE dominate it.
def config_mean_se(vals, ses):
    """Mean over repetitions, and its SE: the larger of the within-run and the across-repetition estimate,
    since within-run SEs are optimistic for sticky chains and 3 repetitions alone are noisy."""
    mean = vals.mean(axis=0)
    se_within = np.sqrt((ses**2).sum(axis=0)) / len(vals)
    se_across = vals.std(axis=0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros_like(mean)
    return mean, np.maximum(se_within, se_across)

std_means = np.array([config_mean_se(np.array([r["standard"] for r in runs[n]]),
                                     np.array([r["se_standard"] for r in runs[n]]))[0] for n in configs])
consensus = std_means.mean(axis=0)
consensus_se = std_means.std(axis=0, ddof=1) / np.sqrt(len(configs)) if len(configs) > 1 else np.zeros(n_q)
print("consensus: " + "  ".join(f"{q}={c:.6g} (+-{s:.2g})" for q, c, s in zip(qoi_names, consensus, consensus_se)))
print(f"{'config':22s} {'estimator':11s} " + "".join(f"{q:>18s}" for q in qoi_names))
for n in configs:
    for est, se_key in (("standard", "se_standard"), ("multilevel", "se_multilevel")):
        mean, se = config_mean_se(np.array([r[est] for r in runs[n]]), np.array([r[se_key] for r in runs[n]]))
        z = (mean - consensus) / np.sqrt(se**2 + consensus_se**2)
        print(f"{n:22s} {est:11s} " + "".join(f"{v:+18.2f}" for v in z))
        for k, q in enumerate(qoi_names):
            rows.append(f"{n},{est},{q},{mean[k]},{se[k]},{z[k]},{vr[n][k]},{eff[n].mean()}")
print("|z| above about 3 means the configuration's estimate is inconsistent with the others.")

with open(out_csv, "w") as fh:
    fh.write("config,estimator,qoi,mean,se,z_vs_consensus,vr_predicted,ess_per_hour\n" + "\n".join(rows) + "\n")


# --- figure ----------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
labels = list(configs)
x = np.arange(len(labels))
axes[0].bar(x, [eff[n].mean() for n in labels], yerr=[eff[n].std(ddof=1) if len(eff[n]) > 1 else 0 for n in labels],
            color=["tab:blue" if runs[n][0]["meta"]["randomize"] else "tab:orange" for n in labels])
axes[0].set_xticks(x, labels, rotation=60, ha="right")
axes[0].set_ylabel("fine-level ESS per hour (worse of E1, E2)")
axes[0].set_title("Efficiency (blue: randomised, orange: fixed)")
for k, q in enumerate(qoi_names):
    axes[1].plot(x, [vr[n][k] for n in labels], "o-", label=q)
axes[1].axhline(1, color="grey", lw=0.8)
axes[1].set_xticks(x, labels, rotation=60, ha="right")
axes[1].set_ylabel("SE(standard) / SE(multilevel)")
axes[1].set_title("Predicted variance reduction, proposal-paired")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print(f"\nwrote {out_png} and {out_csv}")
