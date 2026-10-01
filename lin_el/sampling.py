"""Proposal construction.

Adaptive Metropolis learns the posterior covariance as it goes, but it can
only learn it from samples the chain has actually visited: started from an
isotropic C0 on an anisotropic posterior, it mixes badly, and mixing badly is
exactly what stops it estimating the covariance. The way out is to hand it a
reasonable anisotropic C0 up front.

A Laplace approximation at the MAP does that deterministically for ~9 model
evaluations, which on the coarsest level is cheaper than a pilot chain and has
no tuning of its own.
"""

from __future__ import annotations

import numpy as np
import tinyDA as tda

_DEFAULT_T0 = 100
_DEFAULT_EPSILON = 1e-6


def laplace_covariance(
    posterior,
    theta: np.ndarray,
    rel_step: float = 1e-3,
    abs_step: float = 1e-4,
) -> np.ndarray:
    """Inverse Hessian of the negative log-posterior at `theta`.

    Central differences. With a uniform prior the log-prior is constant in the
    interior, so this is effectively the likelihood's curvature -- which is
    what sets the posterior's shape here.

    Raises ValueError if the result is not positive definite, which happens if
    `theta` is not really a mode or the step size is badly scaled.
    """
    theta = np.asarray(theta, dtype=float)
    d = theta.size
    h = np.maximum(np.abs(theta) * rel_step, abs_step)

    def nlp(t):
        return -posterior.create_link(np.asarray(t, dtype=float)).posterior

    f0 = nlp(theta)
    if not np.isfinite(f0):
        raise ValueError(f"log-posterior is not finite at {theta}")

    hessian = np.zeros((d, d))
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h[i]
        hessian[i, i] = (nlp(theta + ei) - 2.0 * f0 + nlp(theta - ei)) / h[i] ** 2
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = h[j]
            hessian[i, j] = hessian[j, i] = (
                nlp(theta + ei + ej)
                - nlp(theta + ei - ej)
                - nlp(theta - ei + ej)
                + nlp(theta - ei - ej)
            ) / (4.0 * h[i] * h[j])

    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(eigenvalues > 0):
        raise ValueError(
            f"Hessian is not positive definite (eigenvalues {eigenvalues}); "
            f"{theta} may not be a mode, or rel_step needs adjusting"
        )
    return np.linalg.inv(hessian)


def make_adaptive_metropolis(
    problem,
    c0: np.ndarray | None = None,
    laplace_level: int | None = 0,
    t0: int = _DEFAULT_T0,
    epsilon: float = _DEFAULT_EPSILON,
    verbose: bool = True,
):
    """Adaptive Metropolis with a sensibly scaled starting covariance.

    Precedence: an explicit `c0`, else the Laplace covariance at the MAP on
    level `laplace_level` (the coarsest by default -- cheapest, and the
    posterior shape is similar enough across levels to seed a proposal), else
    the identity.

    A fresh proposal object must be built for every run: these are stateful
    and adapt in place.
    """
    d = problem.true_parameters.size

    if c0 is None and laplace_level is not None:
        posterior = problem.posteriors([laplace_level], adaptive_coarse=False)[0]
        try:
            c0 = laplace_covariance(posterior, problem.map_estimate)
            if verbose:
                sd = np.sqrt(np.diag(c0))
                corr = c0[0, 1] / (sd[0] * sd[1]) if d == 2 else np.nan
                print(f"  Laplace C0: sd={np.array2string(sd, precision=3)}"
                      + (f" corr={corr:+.3f}" if d == 2 else ""))
        except (ValueError, np.linalg.LinAlgError) as exc:
            if verbose:
                print(f"  Laplace C0 failed ({exc}); falling back to identity")
            c0 = None

    if c0 is None:
        c0 = np.eye(d)

    return tda.AdaptiveMetropolis(C0=np.asarray(c0, dtype=float),
                                  t0=t0, sd=None, epsilon=epsilon)