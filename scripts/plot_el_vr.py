"""Fig.-9-style variance-reduction plot for the laminate sweep: standard error against fine samples.

For every configuration in results/el_sweep (the vr_<config>_rep<NN>.npz summaries of scripts/vr_run.py)
and a grid of fine-sample counts N, two total sampling errors for one QoI, averaged over repetitions:

  standard    sqrt(Var(Q_L) / ESS(Q_L))                       the plain mean of the fine chain
  multilevel  sqrt(Var(Q_0)/ESS(Q_0) + sum_l Var(Y_l)/ESS(Y_l))  proposal-paired, Y_l = Q_l - Q_{l-1}(promoted)

as in the Fig. 9 replication: every term is cut at the same fraction of its own chain, since all levels
grow in lockstep. One panel per hierarchy, one colour per subchain length J; solid lines are the
multilevel estimator, dashed the standard one of the same runs. No RMSE: there is no reference solution.

Hierarchies are read from the configuration names, <hierarchy>_J<J>_<rand|fixed>[_suffix]
(l012_J5_rand, l02_J10_fixed, l012_J10_rand_noaem, ...). A later GP hierarchy only needs a name of that
form, e.g. gp1_l2_J10_rand, and gets its own panel. Configurations with a suffix (such as _noaem) get
their own panel too, unless --skip-suffixed.

    python scripts/plot_el_vr.py                                  # compliance, randomised promotion
    python scripts/plot_el_vr.py --qoi tip_deflection --promotion both
    python scripts/plot_el_vr.py --configs l012_J10_rand,l012_J10_rand_noaem,l02_J10_rand

Uses no arviz (its API changed in 1.0); the ESS is the usual Geyer initial-monotone estimator.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(_REPO_ROOT, "results", "el_sweep"))
p.add_argument("--qoi", default="compliance", help="tip_deflection, mean_von_mises or compliance")
p.add_argument("--promotion", default="rand", choices=("rand", "fixed", "both"))
p.add_argument("--configs", default=None, help="comma-separated configuration names (default: all)")
p.add_argument("--skip-suffixed", action="store_true", help="leave out e.g. _noaem configurations")
p.add_argument("--points", type=int, default=15, help="points on the N grid")
p.add_argument("--nmin", type=int, default=200, help="smallest N on the grid")
p.add_argument("--out", default=None, help="default: <indir>/el_vr_<qoi>.png")
args = p.parse_args()
indir = os.path.expanduser(args.indir)
out_png = args.out or os.path.join(indir, f"el_vr_{args.qoi}.png")

NAME = re.compile(r"^(?P<hier>.+?)_J(?P<J>\d+)_(?P<prom>rand|fixed)(?P<suffix>_.+)?$")


def ess(x):
    """Effective sample size: FFT autocorrelation, Geyer's initial monotone sequence."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 8 or np.ptp(x) == 0:
        return float(n)
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    acf = np.fft.irfft(f * np.conj(f), m)[:n]
    acf /= acf[0]
    k = (n - 1) // 2
    g = acf[0:2 * k:2] + acf[1:2 * k:2]
    neg = np.where(g < 0)[0]
    g = np.minimum.accumulate(g[: neg[0]] if neg.size else g)
    return float(n / max(-1.0 + 2.0 * g.sum(), 1e-12))


def var_over_ess(x):
    return x.var() / ess(x)


def head(x, fraction):
    return x[: max(8, int(round(fraction * len(x))))]


# --- read ------------------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(indir, "vr_*_rep*.npz")))
if not files:
    raise SystemExit(f"no vr_*_rep*.npz in {indir}")
wanted = set(args.configs.split(",")) if args.configs else None

terms = defaultdict(list)            # config -> list of (Q_top, Q_0, [Y_l]) per repetition
for f in files:
    name = re.match(r"vr_(.+)_rep\d+\.npz", os.path.basename(f)).group(1)
    mt = NAME.match(name)
    if wanted is not None and name not in wanted:
        continue
    if wanted is None and mt:
        if args.promotion != "both" and mt["prom"] != args.promotion:
            continue
        if args.skip_suffixed and mt["suffix"]:
            continue
    d = np.load(f)
    m = json.loads(str(d["meta"]))
    if args.qoi not in m["qoi_names"]:
        raise SystemExit(f"--qoi must be one of {m['qoi_names']}")
    q = m["qoi_names"].index(args.qoi)
    L = len(m["levels"]) - 1
    top = d[f"qoi__chain_l{L}"][:, q]
    q0 = d["qoi__chain_l0"][:, q]
    ys = [d[f"qoi__chain_l{l}"][:, q] - d[f"qoi__promoted_l{l - 1}"][:, q] for l in range(1, L + 1)]
    terms[name].append((top, q0, ys))

if not terms:
    raise SystemExit("no configuration left after the filters")


# --- standard errors along the run -----------------------------------------------------------

n_top = min(len(t[0]) for runs in terms.values() for t in runs)
grid = np.unique(np.geomspace(min(args.nmin, n_top // 2), n_top, args.points).astype(int))

curves = {}                          # config -> (se_standard, se_multilevel), means over repetitions
for name, runs in terms.items():
    std_all, ml_all = [], []
    for top, q0, ys in runs:
        n_this = len(top)
        s, mlv = [], []
        for n in grid:
            frac = n / n_this
            t = head(top, frac)
            s.append(np.sqrt(var_over_ess(t)))
            mlv.append(np.sqrt(var_over_ess(head(q0, frac)) + sum(var_over_ess(head(y, frac)) for y in ys)))
        std_all.append(s)
        ml_all.append(mlv)
    curves[name] = (np.mean(std_all, axis=0), np.mean(ml_all, axis=0), len(runs))


# --- figure ----------------------------------------------------------------------------------

panels = defaultdict(list)           # panel title -> configs
for name in curves:
    mt = NAME.match(name)
    title = (mt["hier"] + (mt["suffix"] or "")) if mt else name
    panels[title].append(name)


def hier_label(title):
    m = re.match(r"^l(\d+)(.*)$", title)
    if m:
        return "levels " + ",".join(m.group(1)) + m.group(2).replace("_", " ")
    return title.replace("_", " ")


Js = sorted({int(NAME.match(n)["J"]) for n in curves if NAME.match(n)})
cmap = plt.get_cmap("viridis")
colour = {J: cmap(i / max(len(Js) - 1, 1) * 0.85) for i, J in enumerate(Js)}

titles = sorted(panels, key=lambda t: (len(t), t))
fig, axes = plt.subplots(1, len(titles), figsize=(4.6 * len(titles), 4.2), sharey=True, squeeze=False)
for ax, title in zip(axes[0], titles):
    for name in sorted(panels[title], key=lambda n: (int(NAME.match(n)["J"]) if NAME.match(n) else 0, n)):
        se_std, se_ml, reps = curves[name]
        mt = NAME.match(name)
        J = int(mt["J"]) if mt else None
        c = colour.get(J, "k")
        prom = f", {mt['prom']}" if (mt and args.promotion == "both") else ""
        ls_ml = "-" if (not mt or mt["prom"] == "rand") else "-."
        ax.loglog(grid, se_ml, ls_ml, color=c, lw=1.8, label=f"J = {J}{prom}: multilevel" if J else name)
        ax.loglog(grid, se_std, "--", color=c, lw=1, alpha=0.8, label=f"J = {J}{prom}: standard" if J else None)
    ax.set_title(f"{hier_label(title)}  ({reps} reps)", fontsize=11)
    ax.set_xlabel("fine samples N")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
axes[0][0].set_ylabel(f"standard error of E[{args.qoi}]")
fig.suptitle("Laminate: total sampling error, multilevel (proposal-paired, solid) vs fine-level mean (dashed)",
             fontsize=11)
fig.tight_layout()
fig.savefig(out_png, dpi=150)

print(f"{'config':24s} {'reps':>4s} {'SE std':>11s} {'SE ML':>11s} {'ratio':>6s}   at N = {grid[-1]}")
for name in sorted(curves):
    s, mlv, reps = curves[name]
    print(f"{name:24s} {reps:4d} {s[-1]:11.4g} {mlv[-1]:11.4g} {s[-1] / mlv[-1]:6.2f}")
print(f"\nwrote {out_png}")
