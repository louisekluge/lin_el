"""Diagnostic plots for the laminate hierarchy sweep (slurm/el_sweep.sbatch).

Two figures, from the vr_<config>_rep<NN>.npz summaries that scripts/vr_run.py writes:

  el_sweep_subchains.png  every AEM configuration against the subchain length J: one line per hierarchy
                          (0,1,2 / 0,2) and promotion (solid randomised, dashed fixed), each repetition
                          as a faint dot, the mean as the line
      (a) efficiency: fine-level ESS per hour, worse of E1 and E2
      (b) acceptance on the finest level (and on level 1 for 0,1,2, dotted)
      (c) predicted variance reduction for one QoI: SE(standard) / SE(multilevel, proposal-paired)
      (d) consistency: multilevel estimate minus the consensus, in standard errors -- fixed promotion
          is expected to drift away from zero, randomised promotion should not

  el_sweep_aem.png        0,1,2, J = 10, randomised, with and without the adaptive error model
      acceptance per level, ESS per hour, predicted variance reduction, and the estimates themselves

    python scripts/plot_el_sweep.py                       # QoI: compliance
    python scripts/plot_el_sweep.py --qoi tip_deflection

Uses no arviz (its API changed in 1.0); the ESS below is the usual Geyer initial-monotone estimator.
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
p.add_argument("--aem-pair", default="l012_J10_rand,l012_J10_rand_noaem",
               help="the two configurations compared in el_sweep_aem.png: with AEM, without AEM")
args = p.parse_args()
indir = os.path.expanduser(args.indir)


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


def se(x):
    return np.sqrt(x.var() / ess(x))


# --- read ------------------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(indir, "vr_*_rep*.npz")))
if not files:
    raise SystemExit(f"no vr_*_rep*.npz in {indir}")

runs = defaultdict(list)
for f in files:
    name = re.match(r"vr_(.+)_rep\d+\.npz", os.path.basename(f)).group(1)
    d = np.load(f)
    m = json.loads(str(d["meta"]))
    if args.qoi not in m["qoi_names"]:
        raise SystemExit(f"--qoi must be one of {m['qoi_names']}")
    q = m["qoi_names"].index(args.qoi)
    L = len(m["levels"]) - 1
    fine = d[f"qoi__chain_l{L}"][:, q]
    q0 = d["qoi__chain_l0"][:, q]
    ys = [d[f"qoi__chain_l{l}"][:, q] - d[f"qoi__promoted_l{l - 1}"][:, q] for l in range(1, L + 1)]
    hours = float(d["runtime_seconds"]) / 3600
    runs[name].append(dict(
        levels=",".join(map(str, m["levels"])), J=m["subchain_length"], rand=m["randomize"], aem=m["aem"],
        acc=m["acceptance"], hours=hours, ess_min=min(m["ess_fine"].values()),
        eff=min(m["ess_fine"].values()) / hours,
        standard=fine.mean(), se_standard=se(fine),
        multilevel=q0.mean() + sum(y.mean() for y in ys),
        se_multilevel=np.sqrt(se(q0) ** 2 + sum(se(y) ** 2 for y in ys)),
    ))
print(f"read {len(files)} runs in {len(runs)} configurations; QoI {args.qoi}")

# consensus: plain mean over configurations of their mean fine-level estimate (no weighting, so one
# sticky run with an underestimated SE cannot dominate it)
consensus = np.mean([np.mean([r["standard"] for r in rs]) for rs in runs.values()])


def z_multilevel(r):
    return (r["multilevel"] - consensus) / r["se_multilevel"]


# --- figure 1: subchain lengths --------------------------------------------------------------

groups = defaultdict(dict)          # (levels, rand) -> J -> runs, AEM configurations only
for rs in runs.values():
    r0 = rs[0]
    if r0["aem"]:
        groups[(r0["levels"], r0["rand"])][r0["J"]] = rs

colour = {"0,1,2": "tab:blue", "0,2": "tab:orange"}
panels = [
    ("eff", "fine-level ESS per hour\n(worse of E1, E2)", True),
    ("acc_fine", "acceptance, finest level", False),
    ("vr", f"SE(standard) / SE(multilevel)\n{args.qoi}", False),
    ("z", f"(multilevel − consensus) / SE\n{args.qoi}", False),
]


def value(r, key):
    if key == "acc_fine":
        return r["acc"][f"level{max(int(k[-1]) for k in r['acc'])}"]
    if key == "vr":
        return r["se_standard"] / r["se_multilevel"]
    if key == "z":
        return z_multilevel(r)
    return r[key]


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (key, ylabel, logy) in zip(axes.ravel(), panels):
    for (levels, rand), byJ in sorted(groups.items()):
        Js = sorted(byJ)
        means = []
        for i, J in enumerate(Js):
            vals = [value(r, key) for r in byJ[J]]
            jitter = (0.06 if rand else -0.06) * J
            ax.scatter([J + jitter] * len(vals), vals, s=12, color=colour.get(levels, "grey"), alpha=0.35)
            means.append(np.mean(vals))
        ax.plot(Js, means, "-" if rand else "--", marker="o", color=colour.get(levels, "grey"),
                label=f"{levels}, {'randomised' if rand else 'fixed'}")
        if key == "acc_fine" and levels == "0,1,2":
            lvl1 = [np.mean([r["acc"]["level1"] for r in byJ[J]]) for J in Js]
            ax.plot(Js, lvl1, ":" , marker=".", color=colour[levels], alpha=0.7,
                    label=f"0,1,2 level 1, {'randomised' if rand else 'fixed'}")
    ax.set_xscale("log")
    ax.set_xticks(sorted({J for g in groups.values() for J in g}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    if key == "vr":
        ax.axhline(1, color="grey", lw=0.8)
    if key == "z":
        ax.axhspan(-2, 2, color="grey", alpha=0.12, lw=0)
        ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("subchain length J")
    ax.set_ylabel(ylabel)
axes[0, 0].legend(fontsize=8)
axes[0, 1].legend(fontsize=7)
for ax, letter in zip(axes.ravel(), "abcd"):
    ax.set_title(f"({letter})", loc="left", fontsize=10)
fig.suptitle("Laminate sweep: subchain length and promotion (all with AEM; dots = repetitions)")
fig.tight_layout()
out1 = os.path.join(indir, "el_sweep_subchains.png")
fig.savefig(out1, dpi=150)


# --- figure 2: AEM vs no AEM -----------------------------------------------------------------

with_aem, without = args.aem_pair.split(",")
if with_aem in runs and without in runs:
    pair = {"with AEM": runs[with_aem], "without AEM": runs[without]}
    cols = {"with AEM": "tab:blue", "without AEM": "tab:red"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))

    # acceptance per level
    ax = axes[0]
    lv = sorted(pair["with AEM"][0]["acc"])
    for i, (lab, rs) in enumerate(pair.items()):
        for k, level in enumerate(lv):
            vals = [r["acc"][level] for r in rs]
            x = k + (i - 0.5) * 0.3
            ax.bar(x, np.mean(vals), width=0.28, color=cols[lab], alpha=0.6, label=lab if k == 0 else None)
            ax.scatter([x] * len(vals), vals, color=cols[lab], s=12, zorder=3)
    ax.set_xticks(range(len(lv)), lv)
    ax.set_ylabel("acceptance")
    ax.set_title("(a) acceptance per level", loc="left", fontsize=10)
    ax.legend(fontsize=8)

    # efficiency and variance reduction
    for ax, key, ylabel, title in ((axes[1], "eff", "fine-level ESS per hour", "(b) efficiency"),
                                   (axes[2], "vr", f"SE(std) / SE(ML), {args.qoi}", "(c) variance reduction")):
        for i, (lab, rs) in enumerate(pair.items()):
            vals = [value(r, key) for r in rs]
            ax.bar(i, np.mean(vals), width=0.6, color=cols[lab], alpha=0.6)
            ax.scatter([i] * len(vals), vals, color=cols[lab], s=12, zorder=3)
        ax.set_xticks([0, 1], list(pair))
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
    axes[2].axhline(1, color="grey", lw=0.8)

    # the estimates themselves
    ax = axes[3]
    for i, (lab, rs) in enumerate(pair.items()):
        for j, (est, mk) in enumerate((("standard", "o"), ("multilevel", "s"))):
            x = i + (j - 0.5) * 0.3
            vals = np.array([r[est] for r in rs])
            ses = np.array([r[f"se_{est}"] for r in rs])
            ax.errorbar([x] * len(vals), vals, yerr=2 * ses, fmt=mk, color=cols[lab], ms=4, capsize=2,
                        alpha=0.8, label=est if i == 0 else None)
    ax.axhline(consensus, color="grey", lw=0.8, ls="--", label="consensus")
    ax.set_xticks([0, 1], list(pair))
    ax.set_ylabel(f"estimate of E[{args.qoi}] (± 2 SE)")
    ax.set_title("(d) estimates per repetition", loc="left", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Adaptive error model: {with_aem} vs {without}")
    fig.tight_layout()
    out2 = os.path.join(indir, "el_sweep_aem.png")
    fig.savefig(out2, dpi=150)
else:
    out2 = None
    print(f"skipping the AEM figure: need both {with_aem} and {without}")


# --- table of what is plotted ----------------------------------------------------------------

print(f"\n{'config':22s} {'reps':>4s} {'ESS/h':>9s} {'acc fine':>9s} {'VR':>6s} {'z(ML)':>7s}")
for name in sorted(runs, key=lambda n: (runs[n][0]["levels"], runs[n][0]["J"], not runs[n][0]["rand"],
                                        not runs[n][0]["aem"])):
    rs = runs[name]
    print(f"{name:22s} {len(rs):4d} {np.mean([r['eff'] for r in rs]):9.1f} "
          f"{np.mean([value(r, 'acc_fine') for r in rs]):9.3f} {np.mean([value(r, 'vr') for r in rs]):6.2f} "
          f"{np.mean([value(r, 'z') for r in rs]):+7.2f}")
print(f"\nwrote {out1}" + (f" and {out2}" if out2 else ""))
