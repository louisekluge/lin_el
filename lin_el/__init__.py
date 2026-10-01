"""Bayesian inversion for a 2D laminate cantilever, with multilevel MCMC.

Top-level names are resolved lazily (PEP 562) so that modules with no FEM
dependency -- `sampling`, and anything added later for analysis -- can be
imported on a machine without dolfinx. Touching `LinearElasticity`,
`build_hierarchy` or `build_problem` is what pulls FEniCSx in.
"""

from __future__ import annotations

__all__ = [
    "LinearElasticity",
    "build_hierarchy",
    "QOI_NAMES",
    "LaminateProblem",
    "build_problem",
    "laplace_covariance",
    "make_adaptive_metropolis",
]

_LOCATIONS = {
    "LinearElasticity": "models",
    "build_hierarchy": "models",
    "QOI_NAMES": "models",
    "LaminateProblem": "problem",
    "build_problem": "problem",
    "laplace_covariance": "sampling",
    "make_adaptive_metropolis": "sampling",
}


def __getattr__(name: str):
    module_name = _LOCATIONS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so later lookups skip this path
    return value


def __dir__():
    return sorted(__all__)