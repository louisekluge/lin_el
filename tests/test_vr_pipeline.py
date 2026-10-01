"""Smoke test for the variance-reduction pipeline without dolfinx.

Swaps the FEM hierarchy for an analytic stand-in with the same interface, so
the extraction, pairing and plotting code can be exercised on a laptop. Does
NOT test the FEM model -- only the machinery around it.

    python tests/test_vr_pipeline.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(outdir, levels, rep):
    """Run scripts/vr_run.py with the mock problem patched in."""
    src = open(os.path.join(ROOT, "scripts", "vr_run.py")).read()
    src = src.replace("from lin_el.problem import build_problem",
                      "from mock_problem import build_problem")
    src = src.replace("from lin_el.models import QOI_NAMES",
                      "QOI_NAMES = ('tip_deflection','mean_von_mises','compliance')")
    src = src.replace(
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        f"sys.path.insert(0, {ROOT!r}); sys.path.insert(0, {HERE!r})")
    patched = os.path.join(outdir, "vr_run_mock.py")
    open(patched, "w").write(src)
    return subprocess.run(
        [sys.executable, patched, "--rep", str(rep), "--levels", levels,
         "--iterations", "400", "--subchain-length", "5", "--outdir", outdir],
        capture_output=True, text=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for levels in ("0,2", "0,1,2"):
            for rep in (0, 1):
                r = run(tmp, levels, rep)
                assert r.returncode == 0, r.stderr[-2000:]
            print(f"  levels={levels}: ok")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "plot_vr.py"),
             "--indir", tmp, "--out", os.path.join(tmp, "x.png"),
             "--csv", os.path.join(tmp, "x.csv")],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        print("  plot_vr: ok")
    print("pipeline smoke test passed")
