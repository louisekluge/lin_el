"""Building blocks for scripts/analysis.ipynb: load every kind of run output into one format, plot from it.

Every loader returns a list of *runs*. A run is a plain dict:

    source     "el_sweep", "vr", "diagnostics", "fig9", "lr"
    config     configuration name, e.g. "l012_J10_rand", "base", "fig9_state"
    rep        repetition index
    path       the file it came from
    meta       the run's own metadata (dict)
    names      QoI names, one per column
    terms      {"Q_0": (n0, q), "Y_10": (n1, q), ...}   coarsest chain and the multilevel corrections,
                                                       burned in and aligned, time-ordered
    top        (nL, q)                                 the finest chain (the standard estimator)
    reference  (q,) or None                            a known true value, where there is one

so the same plotting functions work for all of them. Raw files stay reachable through load() and
load_chains().

    import analysis_tools as at
    at.index()                                  # what is there
    runs = at.load_el_sweep()
    at.plot_se(at.group(runs, "l012_J._rand$"), "compliance")
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set LIN_EL_RESULTS before importing to read results from elsewhere (e.g. a copy of the cluster results)
RESULTS = os.environ.get("LIN_EL_RESULTS", os.path.join(REPO, "results"))
EL_NAME = re.compile(r"^(?P<hier>.+?)_J(?P<J>\d+)_(?P<prom>rand|fixed)(?P<suffix>_.+)?$")


# =============================================================================================
# Raw access
# =============================================================================================

def load(path):
    """Any .npz as a dict of arrays; a 'meta' entry is parsed from JSON."""
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"]))
    return out


def load_chains(path):
    """A chains_*.npz checkpoint as {level: {"chain": {attr: array}, "promoted": {attr: array}, "accepted": array}}
    plus the top-level keys (done, total, complete, subchain_lengths). No burn-in applied.

    attr is "parameters", "qoi" or "stats" (prior, likelihood, posterior per link).
    """
    d = load(path)
    out = {k: d[k] for k in ("done", "total", "complete", "subchain_lengths", "levels") if k in d}
    pat = re.compile(r"^(chain|promoted)_l(\d+)_0__(\w+)$")
    for key, arr in d.items():
        m = pat.match(key)
        if m:
            kind, level, attr = m.group(1), int(m.group(2)), m.group(3)
            out.setdefault(level, {}).setdefault(kind, {})[attr] = arr
        elif key.startswith("accepted_l"):
            out.setdefault(int(key.split("_")[1][1:]), {})["accepted"] = d[key]
    return out


def index(root=RESULTS):
    """Print what is under results/: per directory and file kind, the configurations and repetitions."""
    found = defaultdict(lambda: defaultdict(set))
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), root)
        stem, rep = re.match(r"^(.*?)(?:_rep(\d+))?$", os.path.basename(path)[:-4]).groups()
        kind, _, cfg = stem.partition("_")
        found[(rel, kind)][cfg].add(rep)
    for (rel, kind), cfgs in sorted(found.items()):
        print(f"{rel}/  {kind}_*")
        for cfg, reps in sorted(cfgs.items()):
            reps = sorted(r for r in reps if r is not None)
            print(f"    {cfg or '-':32s} {len(reps):3d} reps" if reps else f"    {cfg or '-'}")


# =============================================================================================
# Loaders -> runs
# =============================================================================================

def _as2d(x):
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _vr_layout(d, n_levels):
    """Terms from the qoi__chain_l* / qoi__promoted_l* layout of vr_run.py and lr_hierarchy.py."""
    terms = {"Q_0": _as2d(d["qoi__chain_l0"])}
    for l in range(1, n_levels):
        fine, prom = _as2d(d[f"qoi__chain_l{l}"]), _as2d(d[f"qoi__promoted_l{l - 1}"])
        n = min(len(fine), len(prom))
        terms[f"Y_{l}{l - 1}"] = fine[:n] - prom[:n]
    return terms, _as2d(d[f"qoi__chain_l{n_levels - 1}"])


def _configs_ok(config, pattern):
    return pattern is None or re.search(pattern, config) is not None


def load_vr(directory=RESULTS, configs=None, source=None):
    """vr_<config>_rep<NN>.npz summaries of scripts/vr_run.py (laminate). `configs`: regex on the name."""
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "vr_*_rep*.npz"))):
        config, rep = os.path.basename(path)[3:-4].rsplit("_rep", 1)
        if not _configs_ok(config, configs):
            continue
        d = load(path)
        meta = d["meta"]
        terms, top = _vr_layout(d, len(meta["levels"]))
        runs.append(dict(source=source or "vr", config=config, rep=int(rep), path=path, meta=meta,
                         names=list(meta.get("qoi_names", [])), terms=terms, top=top, reference=None))
    return runs


def load_el_sweep(directory=os.path.join(RESULTS, "el_sweep"), configs=None):
    """The laminate sweep (slurm/el_sweep.sbatch): same files as load_vr, in results/el_sweep."""
    return load_vr(directory, configs, source="el_sweep")


def load_lr(directory=os.path.join(RESULTS, "lr_hierarchy"), configs=None):
    """Linear-regression hierarchy (scripts/lr_hierarchy.py). reference = exact posterior mean on the
    finest level, so plot_se(..., rmse=True) works here."""
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "lr_*_rep*.npz"))):
        d = load(path)
        meta = d["meta"]
        if not _configs_ok(meta["tag"], configs):
            continue
        L = len(d["sizes"])
        terms, top = _vr_layout(d, L)
        runs.append(dict(source="lr", config=meta["tag"], rep=meta["rep"], path=path, meta=meta,
                         names=list(meta["qoi_names"]), terms=terms, top=top,
                         reference=d["exact_mean_q"][L - 1], sizes=d["sizes"], work=d["work"]))
    return runs


def load_fig9(directory=os.path.join(RESULTS, "fig9"), pairing="both"):
    """Fig. 9 replication (scripts/fig9_sample.py). pairing: "proposal", "state" or "both"; each pairing
    becomes its own config ("fig9_proposal", "fig9_state"). reference = Q_ref from reference.npz if present."""
    ref_path = os.path.join(directory, "reference.npz")
    reference = np.atleast_1d(float(load(ref_path)["Q_ref"])) if os.path.exists(ref_path) else None
    kinds = ("proposal", "state") if pairing == "both" else (pairing,)
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "fig9_rep*.npz"))):
        d = load(path)
        for kind in kinds:
            terms = {"Q_0": _as2d(d["Q_0"]), "Y_10": _as2d(d[f"Y_{kind}_1"]), "Y_21": _as2d(d[f"Y_{kind}_2"])}
            runs.append(dict(source="fig9", config=f"fig9_{kind}", rep=d["meta"]["rep"], path=path,
                             meta=d["meta"], names=["Q"], terms=terms, top=_as2d(d["Q_top"]),
                             reference=reference))
    return runs


def _aligned(chains, promoted, burnin):
    """Burn in and align full chains as mlda_diagnostics.vr_terms does: the top chain carries its initial
    state (one longer than its promoted partner), and every term is cut at the same point in the run."""
    n_levels = len(chains)
    top = chains[n_levels - 1]
    n_top = len(top)
    terms = {"Q_0": chains[0][int(round(burnin * len(chains[0]) / n_top)):]}
    for l in range(1, n_levels):
        fine, prom = chains[l], promoted[l - 1]
        if len(fine) == len(prom) + 1:
            fine = fine[1:]
        n = min(len(fine), len(prom))
        cut = int(round(burnin * n / n_top))
        terms[f"Y_{l}{l - 1}"] = fine[cut:n] - prom[cut:n]
    return terms, top[min(burnin, n_top - 1):]


def load_diagnostics(directory=os.path.join(RESULTS, "diagnostics"), configs=None, reps=None):
    """Predator-prey sweep (scripts/mlda_diagnostics.py). The diag_*.npz summaries have no per-sample
    terms, so this reads the chains_*.npz next to them (slow for the big configs: use `configs`).

    Columns: "mean_predator" (the model's QoI, if the run has one) followed by the six parameters
    (log_P0 ... log_d). The parameters are the honest probe of variance reduction here: the QoI averages
    over a different time window on each level (see mlda_diagnostics.py).
    reps: load only repetitions 0 .. reps-1 (the full sweep's chain files are ~8 GB; base with 20 reps
    alone is a few GB in memory). For final numbers without chains, use diag_table().
    The single-level config comes back with terms = {"Q_0": top}, so it plots as standard = multilevel.
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "diag_*_rep*.npz"))):
        config, rep = os.path.basename(path)[5:-4].rsplit("_rep", 1)
        if not _configs_ok(config, configs) or (reps is not None and int(rep) >= reps):
            continue
        summary = load(path)
        meta = summary["meta"]
        chain_path = os.path.join(directory, f"chains_{config}_rep{rep}.npz")
        if not os.path.exists(chain_path):
            print(f"skipping {os.path.basename(path)}: no chain file")
            continue
        c = load_chains(chain_path)
        attrs = (["qoi"] if meta.get("has_qoi") else []) + ["parameters"]
        n_levels = meta["n_levels"]

        def cols(entry):
            return np.hstack([_as2d(entry[a]) for a in attrs])

        chains = [cols(c[l]["chain"]) for l in range(n_levels)]
        promoted = [cols(c[l]["promoted"]) for l in range(n_levels - 1)]
        terms, top = _aligned(chains, promoted, meta["burnin"])
        names = (["mean_predator"] if meta.get("has_qoi") else []) + list(meta["parameter_names"])
        runs.append(dict(source="diagnostics", config=config, rep=int(rep), path=chain_path, meta=meta,
                         names=list(names), terms=terms, top=top, reference=None, summary=summary))
    return runs


# =============================================================================================
# Selecting and grouping
# =============================================================================================

def group(runs, pattern=None, key=None):
    """{label: [runs]} grouped by config (or by key(run)), keeping only configs matching the regex."""
    out = defaultdict(list)
    for r in runs:
        if _configs_ok(r["config"], pattern):
            out[key(r) if key else r["config"]].append(r)
    return dict(sorted(out.items()))


def el_parts(config):
    """'l012_J10_rand_noaem' -> dict(hier='l012', J=10, prom='rand', suffix='_noaem'); None if it doesn't parse."""
    m = EL_NAME.match(config)
    return None if m is None else dict(hier=m["hier"], J=int(m["J"]), prom=m["prom"], suffix=m["suffix"] or "")


def _col(run, qoi):
    if isinstance(qoi, (int, np.integer)):
        return int(qoi)
    return run["names"].index(qoi)


# =============================================================================================
# Statistics
# =============================================================================================

def ess(x):
    """Effective sample size: FFT autocorrelation, Geyer's initial monotone sequence. No arviz."""
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


def _head(x, frac):
    return x[:max(8, int(round(frac * len(x))))]


def estimates(run, qoi, frac=1.0):
    """(standard, multilevel, se_standard, se_multilevel) using the first `frac` of every term."""
    k = _col(run, qoi)
    top = _head(run["top"][:, k], frac)
    ts = [_head(t[:, k], frac) for t in run["terms"].values()]
    return (top.mean(), sum(t.mean() for t in ts),
            np.sqrt(top.var() / ess(top)), np.sqrt(sum(t.var() / ess(t) for t in ts)))


def se_curve(run, qoi, points=15, nmin=200):
    """Along the run: N fine samples -> standard and multilevel estimate and SE. Every term is cut at the
    same fraction of its own chain (all levels grow in lockstep)."""
    n_top = len(run["top"])
    N = np.unique(np.geomspace(min(nmin, n_top // 2), n_top, points).astype(int))
    rows = np.array([estimates(run, qoi, n / n_top) for n in N])
    return dict(N=N, standard=rows[:, 0], multilevel=rows[:, 1], se_standard=rows[:, 2], se_multilevel=rows[:, 3])


def term_table(groups, qoi, show=True):
    """Per group: Var, ESS, Var/ESS of every term and of the fine chain, averaged over repetitions, and the
    SE ratio standard / multilevel. If a correction's Var is close to Var(Q_top), no reduction is possible."""
    out = {}
    for label, runs in groups.items():
        rows = defaultdict(list)
        for r in runs:
            k = _col(r, qoi)
            for name, t in list(r["terms"].items()) + [("Q_top", r["top"])]:
                x = t[:, k]
                e = ess(x)
                rows[name].append((x.var(), e, x.var() / e))
        rows = {n: np.mean(v, axis=0) for n, v in rows.items()}
        ml = np.sqrt(sum(v[2] for n, v in rows.items() if n != "Q_top"))
        out[label] = dict(terms=rows, ratio=np.sqrt(rows["Q_top"][2]) / ml, reps=len(runs))
        if show:
            print(f"--- {label}  ({len(runs)} reps)   SE standard / multilevel = {out[label]['ratio']:.2f}")
            for n, (v, e, ve) in rows.items():
                print(f"    {n:8s} Var {v:10.3e}   ESS {e:8.0f}   Var/ESS {ve:10.3e}")
    return out


def run_table(runs):
    """One line per config: reps, acceptance per level, pairing, runtime -- from the runs' metadata."""
    for label, rs in group(runs).items():
        m = rs[0]["meta"]
        acc = defaultdict(list)
        for r in rs:
            for k, v in (r["meta"].get("acceptance") or {}).items():
                acc[k].append(v)
        acc_s = "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(acc.items())) or "-"
        pair = m.get("pairing") or {}
        pair_s = "  ".join(f"{k}={np.mean([r['meta']['pairing'][k] for r in rs]):.2f}" for k in sorted(pair)) or "-"
        rt = [r["meta"].get("runtime_seconds") for r in rs if r["meta"].get("runtime_seconds")]
        rt_s = f"{np.mean(rt) / 60:7.1f} min" if rt else ""
        print(f"{label:28s} {len(rs):3d} reps  acc: {acc_s}   pairing: {pair_s}   {rt_s}")


def diag_table(directory=os.path.join(RESULTS, "diagnostics"), block="theta", configs=None):
    """Predator-prey sweep from the small diag_*.npz summaries only (fast, all repetitions): per config the
    SE ratio standard / multilevel for every column, mean over repetitions, plus acceptance on the finest
    level and its ESS. block: "theta" (parameters) or "qoi"."""
    rows = defaultdict(list)
    names = None
    for path in sorted(glob.glob(os.path.join(directory, "diag_*_rep*.npz"))):
        config = os.path.basename(path)[5:-4].rsplit("_rep", 1)[0]
        if not _configs_ok(config, configs):
            continue
        d = load(path)
        m = d["meta"]
        if f"{block}__top_var" not in d:
            continue
        names = m["parameter_names"] if block == "theta" else ["mean_predator"]
        se_std = np.sqrt(d[f"{block}__top_var"] / d[f"{block}__top_ess"])
        if f"{block}__term_var" in d:
            se_ml = np.sqrt((d[f"{block}__term_var"] / d[f"{block}__term_ess"]).sum(axis=0))
        else:
            se_ml = se_std
        top = m["n_levels"] - 1
        rows[config].append((se_std / se_ml, m["acceptance"].get(f"level{top}", np.nan), m["runtime_seconds"]))
    if not rows:
        print("no diag files")
        return {}
    print(f"{'config':16s} {'reps':>4s} {'acc_top':>7s} {'min/run':>7s}   SE ratio standard / multilevel")
    print(f"{'':39s}" + "".join(f"{n:>9s}" for n in names))
    out = {}
    for config, rs in rows.items():
        ratio = np.mean([r[0] for r in rs], axis=0)
        out[config] = ratio
        print(f"{config:16s} {len(rs):4d} {np.mean([r[1] for r in rs]):7.3f} "
              f"{np.mean([r[2] for r in rs]) / 60:7.1f}  " + "".join(f"{v:9.2f}" for v in ratio))
    return out


# =============================================================================================
# Plots
# =============================================================================================

def plot_se(groups, qoi, ax=None, standard=True, rmse=False, colors=None, label=None, points=15, nmin=200,
            standard_style="--", multilevel_style="-", title=None):
    """Fig.-9-style plot: SE against fine samples, mean over repetitions. Solid = multilevel, dashed =
    standard of the same runs. rmse=True adds the RMSE against run['reference'] as open circles
    (only where a reference exists: lr, fig9).

    groups  {label: [runs]}, e.g. group(runs, "l02_")
    label   function label -> legend text (default: the label itself)
    """
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (lab, runs) in enumerate(groups.items()):
        c = colors[i % len(colors)] if isinstance(colors, (list, tuple)) else colors[lab]
        curves = [se_curve(r, qoi, points, nmin) for r in runs]
        n = min(len(cv["N"]) for cv in curves)
        N = curves[0]["N"][:n]
        text = label(lab) if label else lab
        ax.loglog(N, np.mean([cv["se_multilevel"][:n] for cv in curves], axis=0), multilevel_style, color=c,
                  label=f"{text}, multilevel")
        if standard:
            ax.loglog(N, np.mean([cv["se_standard"][:n] for cv in curves], axis=0), standard_style, color=c,
                      label=f"{text}, standard")
        if rmse and runs[0].get("reference") is not None:
            ref = runs[0]["reference"][_col(runs[0], qoi)]
            for key, marker in (("multilevel", "o"), ("standard", "s")):
                if key == "standard" and not standard:
                    continue
                err = np.sqrt(np.mean([(cv[key][:n] - ref) ** 2 for cv in curves], axis=0))
                ax.loglog(N, err, marker, color=c, mfc="none", ms=5, label=f"{text}, {key} RMSE")
    ax.set_xlabel("fine samples $N$")
    ax.set_ylabel("standard error")
    ax.set_title(title if title is not None else str(qoi))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    return ax


def plot_el_vr(runs, qoi="compliance", promotion="rand", skip_suffixed=False, configs=None, points=15, nmin=200,
               sharey=True):
    """plot_el_vr.py as a function: one panel per hierarchy (suffixed configs like _noaem get their own),
    one colour per J, solid multilevel, dashed standard; with promotion='both', fixed promotion is
    dash-dot (multilevel) and dotted (standard)."""
    panels = defaultdict(dict)
    for lab, rs in group(runs, configs).items():
        p = el_parts(lab)
        if p is None or (promotion != "both" and p["prom"] != promotion) or (skip_suffixed and p["suffix"]):
            continue
        panels[p["hier"] + p["suffix"]][lab] = rs
    if not panels:
        raise ValueError("no configurations left to plot")
    Js = sorted({el_parts(l)["J"] for g in panels.values() for l in g})
    cmap = plt.get_cmap("viridis")
    jcol = {J: cmap(i / max(1, len(Js) - 1) * 0.85) for i, J in enumerate(Js)}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3), squeeze=False, sharey=sharey)
    for ax, (hier, g) in zip(axes[0], sorted(panels.items())):
        for lab, rs in sorted(g.items(), key=lambda kv: (el_parts(kv[0])["J"], kv[0])):
            p = el_parts(lab)
            plot_se({lab: rs}, qoi, ax=ax, colors=[jcol[p["J"]]], points=points, nmin=nmin,
                    label=lambda l, p=p: f"J={p['J']} {p['prom']}" if promotion == "both" else f"J={p['J']}",
                    multilevel_style="-" if p["prom"] == "rand" else "-.",
                    standard_style="--" if p["prom"] == "rand" else ":", title=hier)
    fig.suptitle(f"{qoi}: standard error, solid multilevel, dashed standard (mean over repetitions)")
    fig.tight_layout()
    return fig
