"""Aggregate vr_*.npz and compare the standard and variance-reduction
estimators for each quantity of interest.

The primary figure is the ESS-based standard error against sample count. That
needs no reference solution and is far smoother than RMSE against one -- see
the predator-prey study, where a leave-one-out reference visibly bent the
right-hand end of the curve.

The table printed alongside is the diagnostic that actually tells the story:
if Var(Y_l) is comparable to Var(Q_top), no variance reduction is possible,
however the curves happen to look.

    python scripts/plot_vr.py --indir results --out vr_laminate.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(_REPO_ROOT, "results"))
p.add_argument("--out", default=os.path.join(_REPO_ROOT, "results", "vr_laminate.png"))
p.add_argument("--csv", default=os.path.join(_REPO_ROOT, "results", "vr_laminate.csv"))
args = p.parse_args()

files = sorted(glob.glob(os.path.join(os.path.expanduser(args.indir), "vr_*.npz")))
if not files:
    raise SystemExit(f"no vr_*.npz in {args.indir}")

runs = defaultdict(list)
for f in files:
    d = np.load(f, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    tag = os.path.basename(f).rsplit("_rep", 1)[0][len("vr_"):]
    runs[tag].append((d, meta))

print(f"{len(files)} files, {len(runs)} configs: "
      + ", ".join(f"{k} (n={len(v)})" for k, v in sorted(runs.items())))


# ---------------------------------------------------------------------------

def ess(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8 or np.allclose(x, x[0]):
        return float(x.size)
    return float(az.ess(az.convert_to_dataset(x[None, :]))["x"].item())


def se_terms(d, meta, k):
    """Per-term var/ESS for QoI component k.

    Returns (standard_se, vr_se, detail) where detail lists
    (name, variance, ess, var/ess) for the coarsest chain and each correction.
    """
    n_levels = len(meta["levels"])
    top = np.asarray(d[f"qoi__chain_l{n_levels - 1}"], dtype=float)[:, k]
    terms = [("Q_0", np.asarray(d["qoi__chain_l0"], dtype=float)[:, k])]
    for lvl in range(1, n_levels):
        fine = np.asarray(d[f"qoi__chain_l{lvl}"], dtype=float)[:, k]
        prom = np.asarray(d[f"qoi__promoted_l{lvl - 1}"], dtype=float)[:, k]
        n = min(len(fine), len(prom))
        terms.append((f"Y_{lvl}{lvl - 1}", fine[:n] - prom[:n]))

    detail, total = [], 0.0
    for name, x in terms:
        v, e = float(np.var(x)), ess(x)
        detail.append((name, v, e, v / e))
        total += v / e
    v_top, e_top = float(np.var(top)), ess(top)
    detail.append(("Q_top (standard)", v_top, e_top, v_top / e_top))
    return np.sqrt(v_top / e_top), np.sqrt(total), detail


def curves(d, meta, k, n_points=30):
    """Standard and VR standard error against fine-level sample count.

    All levels grow in lockstep with the subchain lengths, so each term is
    truncated at the same *fraction* of its own chain -- ratios are taken from
    the actual array lengths rather than assuming the nominal subchain length,
    which randomised subchains would break.
    """
    n_levels = len(meta["levels"])
    top = np.asarray(d[f"qoi__chain_l{n_levels - 1}"], dtype=float)[:, k]
    n_fine = len(top)
    idx = np.unique(np.logspace(np.log10(50), np.log10(n_fine), n_points).astype(int))

    terms = [np.asarray(d["qoi__chain_l0"], dtype=float)[:, k]]
    for lvl in range(1, n_levels):
        fine = np.asarray(d[f"qoi__chain_l{lvl}"], dtype=float)[:, k]
        prom = np.asarray(d[f"qoi__promoted_l{lvl - 1}"], dtype=float)[:, k]
        n = min(len(fine), len(prom))
        terms.append(fine[:n] - prom[:n])

    std_se, vr_se = [], []
    for i in idx:
        frac = i / n_fine
        std_se.append(np.sqrt(np.var(top[:i]) / ess(top[:i])))
        tot = 0.0
        for x in terms:
            m = max(int(round(frac * len(x))), 8)
            tot += np.var(x[:m]) / ess(x[:m])
        vr_se.append(np.sqrt(tot))
    return idx, np.array(std_se), np.array(vr_se)


# ---------------------------------------------------------------------------

qoi_names = next(iter(runs.values()))[0][1]["qoi_names"]
n_qoi = len(qoi_names)

lines = ["config,qoi,term,variance,ess,var_over_ess,n_reps"]
print()
for tag in sorted(runs):
    reps = runs[tag]
    meta0 = reps[0][1]
    print(f"--- {tag}: levels={meta0['levels']} J={meta0['subchain_length']} "
          f"sampler={meta0['sampler']} n={len(reps)} ---")
    for k, qname in enumerate(qoi_names):
        agg = defaultdict(list)
        ratios = []
        for d, meta in reps:
            s, v, detail = se_terms(d, meta, k)
            ratios.append(s / v)
            for name, var, e, ve in detail:
                agg[name].append((var, e, ve))
        print(f"  {qname}:  SE ratio standard/VR = "
              f"{np.mean(ratios):.2f} +/- {np.std(ratios):.2f}")
        for name, vals in agg.items():
            var = np.mean([x[0] for x in vals])
            e = np.mean([x[1] for x in vals])
            ve = np.mean([x[2] for x in vals])
            print(f"      {name:18s} var={var:.3e}  ESS={e:8.0f}  var/ESS={ve:.3e}")
            lines.append(f"{tag},{qname},{name},{var},{e},{ve},{len(reps)}")

with open(args.csv, "w") as fh:
    fh.write("\n".join(lines) + "\n")

fig, axes = plt.subplots(1, n_qoi, figsize=(5 * n_qoi, 4.2), squeeze=False)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, qname in enumerate(qoi_names):
    ax = axes[0, k]
    for c, tag in enumerate(sorted(runs)):
        stds, vrs = [], []
        for d, meta in runs[tag]:
            idx, s, v = curves(d, meta, k)
            stds.append(s)
            vrs.append(v)
        n = min(len(s) for s in stds)
        idx = idx[:n]
        s = np.mean([x[:n] for x in stds], axis=0)
        v = np.mean([x[:n] for x in vrs], axis=0)
        col = colors[c % len(colors)]
        ax.loglog(idx, s, "--", color=col, label=f"{tag} standard")
        ax.loglog(idx, v, "-", color=col, label=f"{tag} VR")
    ax.set_title(qname)
    ax.set_xlabel("fine-level samples")
    ax.set_ylabel("standard error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"\nwrote {args.out} and {args.csv}")
