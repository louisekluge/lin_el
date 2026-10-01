"""Forward-model smoke test: does the FEM hierarchy support multilevel MCMC?

No sampling. Evaluates every level at the true parameters and reports the QoI,
the inter-level differences and the cost per solve. Those three numbers decide
whether MLDA and variance reduction can pay off here:

* the QoI must converge under refinement, or the telescoping sum has no
  structure to exploit;
* the inter-level QoI difference is what the correction terms estimate;
* the cost ratio between levels is what MLDA buys its speedup with -- if the
  coarse level is not much cheaper, subchains only cost.

    pixi run python tests/smoke_fem.py
    pixi run python tests/smoke_fem.py --levels 4 --with-reference
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lin_el.models import QOI_NAMES, build_hierarchy  # noqa: E402
from lin_el.problem import TRUE_PARAMETERS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--levels", type=int, default=3)
p.add_argument("--repeats", type=int, default=3, help="solves to time per level")
p.add_argument("--with-reference", action="store_true",
               help="also evaluate the data-generating level (slow: ~1e6 cells)")
args = p.parse_args()

print(f"building hierarchy ({args.levels} levels)...")
t0 = time.time()
levels, data_model = build_hierarchy(n_levels=args.levels)
print(f"  {time.time() - t0:.1f} s\n")

models = list(enumerate(levels))
if args.with_reference:
    models.append(("ref", data_model))

rows = []
for name, model in models:
    model(TRUE_PARAMETERS)  # warm up: first solve pays setup costs
    t0 = time.time()
    for _ in range(args.repeats):
        obs, qoi = model(TRUE_PARAMETERS)
    dt = (time.time() - t0) / args.repeats
    rows.append((name, model.ndofs, dt, np.asarray(obs), np.asarray(qoi, dtype=float)))

w = 16
print(f"{'level':>6} {'ndofs':>9} {'s/solve':>9} {'|obs|':>12} "
      + "".join(f"{n:>{w}}" for n in QOI_NAMES))
for name, ndofs, dt, obs, qoi in rows:
    print(f"{str(name):>6} {ndofs:>9} {dt:>9.3f} {np.linalg.norm(obs):>12.6g} "
          + "".join(f"{v:>{w}.6g}" for v in qoi))

print("\ninter-level QoI differences (what the correction terms estimate)")
print(f"{'pair':>10} " + "".join(f"{n:>{w}}" for n in QOI_NAMES) + "   (relative)")
for (n0, _, _, _, q0), (n1, _, _, _, q1) in zip(rows, rows[1:]):
    d = q1 - q0
    rel = np.where(q1 != 0, np.abs(d / q1), np.nan)
    print(f"{f'{n0}->{n1}':>10} " + "".join(f"{v:>{w}.4g}" for v in d)
          + "   " + "  ".join(f"{r:.2%}" for r in rel))

print("\ncost ratio relative to the finest level")
t_fine = rows[len(levels) - 1][2]
for name, _, dt, _, _ in rows:
    print(f"  level {str(name):>3}: {t_fine / dt:6.1f}x cheaper"
          if dt > 0 else f"  level {name}: ?")

print("\nchecks")
tip = rows[-1][4][0]
vm = rows[-1][4][1]
print(f"  tip deflection negative : {tip < 0}   ({tip:.4g})")
print(f"  von Mises positive      : {vm > 0}   ({vm:.4g})")
print(f"  QoI differences shrink  : "
      f"{all(abs(rows[i+1][4][0] - rows[i][4][0]) >= abs(rows[i+2][4][0] - rows[i+1][4][0]) for i in range(len(rows)-2))}"
      if len(rows) > 2 else "  (needs >=3 levels)")