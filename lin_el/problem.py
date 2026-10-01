"""Inverse problem setup: synthetic data, prior, likelihoods, posteriors.

Everything that defines *which* posterior we are sampling lives here, so a
runner script only has to choose a hierarchy and a sampler.

The problem is constructed under a FIXED seed. Repetitions of an experiment
must differ only in the sampler's randomness, never in the data -- otherwise
they are not repetitions of the same problem and pooling them is meaningless.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as stats
import tinyDA as tda

from .models import LinearElasticity, build_hierarchy

PROBLEM_SEED = 678
TRUE_PARAMETERS = np.array([80.0, 40.0])
NOISE_SD = 0.005


@dataclass
class LaminateProblem:
    """Everything needed to assemble tinyDA posteriors for any sub-hierarchy."""

    levels: list[LinearElasticity]
    data: np.ndarray
    prior: object
    true_parameters: np.ndarray
    noise_sd: float
    qoi_true: np.ndarray
    # MAP on the coarsest level, filled in by build_problem. Used both as the
    # chain's starting point and as the expansion point for the Laplace
    # proposal covariance in lin_el.sampling.
    map_estimate: np.ndarray = field(default=None)
    map_neg_log_posterior: float = field(default=None)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def posteriors(self, level_indices: list[int], adaptive_coarse: bool = True):
        """tinyDA posteriors for the given levels, ordered coarse -> fine.

        The finest level always gets an exact `GaussianLogLike`. Coarse levels
        get `AdaptiveGaussianLogLike` when `adaptive_coarse` is set, which is
        what tinyDA's adaptive error model needs.

        A fresh likelihood object is built on every call: the adaptive ones
        accumulate bias statistics, so reusing them across runs would leak
        state between repetitions.
        """
        cov = self.noise_sd**2 * np.eye(self.data.size)
        out = []
        for position, idx in enumerate(level_indices):
            is_finest = position == len(level_indices) - 1
            if is_finest or not adaptive_coarse:
                loglike = tda.GaussianLogLike(self.data, cov)
            else:
                loglike = tda.AdaptiveGaussianLogLike(self.data, cov)
            out.append(tda.Posterior(self.prior, loglike, self.levels[idx]))
        return out


def _compute_map(problem: LaminateProblem) -> tuple[np.ndarray, float]:
    """Deterministic MAP on the coarsest level.

    tda.get_MAP defaults to starting the optimiser at prior.rvs(). On this
    posterior -- a thin ridge, corr(E1, E2) ~ -0.97 -- a simplex started
    somewhere random stops at a different point along the ridge every time,
    which makes both the chain's starting point and the Laplace proposal
    covariance non-reproducible. Start from a fixed point and restart the
    simplex until it stops moving, since a single pass tends to stall crawling
    along the ridge rather than converging across it.
    """
    posterior = problem.posteriors([0], adaptive_coarse=False)[0]

    def nlp(t):
        return -posterior.create_link(np.asarray(t, dtype=float)).posterior

    theta = np.array([75.0, 25.1])  # prior midpoints
    for _ in range(5):
        new = np.asarray(
            tda.get_MAP(
                posterior,
                initial_parameters=theta,
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
            ),
            dtype=float,
        )
        settled = np.allclose(new, theta, rtol=0.0, atol=1e-7)
        theta = new
        if settled:
            break
    else:
        warnings.warn(
            f"MAP restarts did not settle; using last point {theta}", RuntimeWarning
        )

    return theta, float(nlp(theta))


def build_problem(n_levels: int = 3, seed: int = PROBLEM_SEED) -> LaminateProblem:
    """Build the laminate inverse problem. Deterministic given `seed`."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        levels, data_model = build_hierarchy(n_levels=n_levels)

        x_true, qoi_true = data_model(TRUE_PARAMETERS)
        noise = np.random.normal(scale=NOISE_SD, size=x_true.shape[0])
        data = x_true + noise

        prior = tda.JointPrior(
            [
                stats.uniform(loc=50, scale=50),   # E1 in [50, 100]
                stats.uniform(loc=0.1, scale=50),  # E2 in [0.1, 50.1]
            ]
        )
    finally:
        np.random.set_state(rng_state)

    problem = LaminateProblem(
        levels=levels,
        data=data,
        prior=prior,
        true_parameters=TRUE_PARAMETERS,
        noise_sd=NOISE_SD,
        qoi_true=np.asarray(qoi_true, dtype=float),
    )

    problem.map_estimate, problem.map_neg_log_posterior = _compute_map(problem)
    return problem