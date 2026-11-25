"""Gaussian-process surrogate of one FEM level, usable as a level in the hierarchy.

The GP maps the parameters (E1, E2) to the same outputs as LinearElasticity: the observation vector and the
QoIs, so it plugs into tda.Posterior unchanged and the multilevel estimator gets Q_0 from the GP.

Inputs are whitened with the Laplace approximation at the level-0 MAP, z = L^-1 (theta - MAP): the posterior
is a thin ridge (corr ~ -0.97), and in z the training design and the kernel's length scales line up with it.
Trained by scripts/train_gp.py; the pickle holds the fitted sklearn model plus the whitening.
"""

from __future__ import annotations

import pickle

import numpy as np


class GPLevel:
    """Callable like LinearElasticity: theta -> (observations, qoi), from the GP posterior mean."""

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self.gp = d["gp"]
        self.centre = np.asarray(d["centre"], dtype=float)
        self.L_inv = np.linalg.inv(np.asarray(d["L"], dtype=float))
        self.n_obs = int(d["n_obs"])
        self.info = {k: v for k, v in d.items() if k not in ("gp", "X", "Y")}

    def _z(self, theta):
        return (self.L_inv @ (np.atleast_2d(theta) - self.centre).T).T

    def predict(self, theta, return_std=False):
        """Batch prediction, rows of theta -> rows of [observations, qoi]."""
        return self.gp.predict(self._z(theta), return_std=return_std)

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = self.gp.predict(self._z(state))[0]
        return y[: self.n_obs], y[self.n_obs:]


class GPLogLikeLevel:
    """Bottom level from two GPs trained on one FEM level: one for its log-likelihood, one for its QoIs.

    Its "model output" is the log-likelihood itself, so it goes with IdentityLogLike, and it cannot take
    part in the adaptive error model (there is no output vector to correct): run with --no-aem.
    """

    def __init__(self, gp_loglike, gp_qoi, info=None):
        self.gp_loglike = gp_loglike
        self.gp_qoi = gp_qoi
        self.info = info or {}

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(state)
        return np.atleast_1d(self.gp_loglike.predict(X)[0]), np.atleast_1d(self.gp_qoi.predict(X)[0])


class IdentityLogLike:
    """Likelihood for GPLogLikeLevel: the model output already is the log-likelihood."""

    def loglike(self, x):
        return float(np.ravel(x)[0])
