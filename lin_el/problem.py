"""Inverse problem setup: synthetic data, prior, likelihoods, posteriors.

Everything that defines *which* posterior we are sampling lives here, so a
runner script only has to choose a hierarchy and a sampler.

The problem is constructed under a FIXED seed. Repetitions of an experiment
must differ only in the sampler's randomness, never in the data -- otherwise
they are not repetitions of the same problem and pooling them is meaningless.
"""

from __future__ import annotations

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
    map_estimate: np.ndarray = field(default=None)

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

    # MAP on the coarsest level: cheap, and only used as a starting point.
    coarse_posterior = problem.posteriors([0], adaptive_coarse=False)[0]
    problem.map_estimate = tda.get_MAP(coarse_posterior)
    return problem
