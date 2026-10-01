"""Forward models for the 2D laminate cantilever.

A single `LinearElasticity` class, used for every level of the hierarchy, plus
a helper that builds a hierarchy of uniformly refined meshes sharing one
observation mesh.

Changes relative to the notebook / `testing_mlda.py`:

* `__call__` returns a real quantity of interest instead of the placeholder
  `0`, so multilevel variance reduction has something to estimate.
* The stress expression used for the QoI is the *same* `sigma()` used in the
  weak form, so the QoI is always consistent with the operator being solved
  (see SHEAR_FACTOR below).
* QoI forms are compiled once in `__init__`; recompiling them per call would
  dominate the runtime over tens of thousands of samples.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import dx, inner

from .helpers import interpolate_nonmatching

# ---------------------------------------------------------------------------
# Constitutive law
#
# The standard isotropic law is  sigma = lambda*tr(eps)*I + 2*mu*eps.
# The notebook and testing_mlda.py use a factor of 1 on the deviatoric term,
# which halves the effective shear modulus: the inferred "E" is therefore not
# a Young's modulus in the usual sense. The inverse problem is still
# self-consistent (the synthetic data comes from the same operator), so this
# constant is left at the legacy value to keep earlier runs comparable.
#
# Set to 2.0 for the textbook law -- but then regenerate the data and redo any
# reference runs, because the posterior moves.
# ---------------------------------------------------------------------------
SHEAR_FACTOR = 1.0

# QoI component names, in the order returned by LinearElasticity.__call__.
QOI_NAMES = ("tip_deflection", "mean_von_mises", "compliance")


class LinearElasticity:
    """2D cantilever laminate: clamped at x=0, body force, two-layer Young's
    modulus (E1 in the lower half, E2 in the upper half).

    Parameters
    ----------
    solve_mesh
        The mesh this level solves on.
    observation_mesh
        Shared across levels, so every level produces an observation vector of
        the same length. tinyDA's adaptive error model requires this.
    tip_point
        Where the tip-deflection QoI is evaluated, in the solve mesh's
        coordinates. Default is the free end at mid-height.
    """

    def __init__(
        self,
        solve_mesh: mesh.Mesh,
        observation_mesh: mesh.Mesh,
        tip_point: tuple[float, float] = (3.0, 0.5),
    ) -> None:
        self._observation_mesh = observation_mesh
        self._mesh = solve_mesh
        gdim = self._mesh.geometry.dim
        self._V = fem.functionspace(self._mesh, ("Lagrange", 1, (gdim,)))
        self._V_observation = fem.functionspace(
            self._observation_mesh, ("Lagrange", 1, (gdim,))
        )

        # --- boundary conditions: clamped at x = 0 --------------------------
        facets_left = mesh.locate_entities_boundary(
            self._mesh,
            dim=(self._mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0.0),
        )
        dofs_left = fem.locate_dofs_topological(
            V=self._V, entity_dim=self._mesh.topology.dim - 1, entities=facets_left
        )
        bc_left = fem.dirichletbc(
            value=default_scalar_type((0, 0)), dofs=dofs_left, V=self._V
        )

        # --- coefficients ---------------------------------------------------
        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)
        f = fem.Constant(self._mesh, default_scalar_type((0, -0.1)))
        x = ufl.SpatialCoordinate(self._mesh)

        nu = 0.3
        self._E1 = fem.Constant(self._mesh, default_scalar_type(20.0))
        self._E2 = fem.Constant(self._mesh, default_scalar_type(10.0))

        def lam(E):
            return E * nu / ((1 + nu) * (1 - 2 * nu))

        def mu(E):
            return E / (2 * (1 + nu))

        def E_field(xx):
            bottom = np.min(self._mesh.geometry.x[:, 1])
            top = np.max(self._mesh.geometry.x[:, 1])
            height = top - bottom
            return ufl.conditional(xx[1] <= bottom + height / 2, self._E1, self._E2)

        def epsilon(w):
            return 0.5 * (ufl.grad(w) + ufl.grad(w).T)

        def sigma(w):
            E = E_field(x)
            return (
                lam(E) * ufl.nabla_div(w) * ufl.Identity(len(w))
                + SHEAR_FACTOR * mu(E) * epsilon(w)
            )

        # --- weak form ------------------------------------------------------
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(f, v) * dx

        self._problem = LinearProblem(
            a,
            L,
            bcs=[bc_left],
            petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
            petsc_options_prefix="lp",
        )

        # --- interpolation onto the shared observation mesh -----------------
        self._cells_V_to = np.arange(
            self._V_observation.mesh.topology.index_map(
                self._V_observation.mesh.topology.dim
            ).size_local,
            dtype=np.int32,
        )
        self._interpolation_data = fem.create_interpolation_data(
            self._V_observation, self._V, self._cells_V_to
        )

        # --- QoI forms, compiled once ---------------------------------------
        # dolfinx reuses the same Function object on every solve, so forms
        # built against it here stay valid for the life of the object.
        uh = self._problem.u
        s = sigma(uh)
        # plane strain: eps_zz = 0, so the deviatoric term drops out of s_zz
        # whatever SHEAR_FACTOR is.
        s_zz = lam(E_field(x)) * ufl.nabla_div(uh)
        von_mises = ufl.sqrt(
            0.5
            * (
                (s[0, 0] - s[1, 1]) ** 2
                + (s[1, 1] - s_zz) ** 2
                + (s_zz - s[0, 0]) ** 2
                + 6.0 * s[0, 1] ** 2
            )
        )

        one = fem.Constant(self._mesh, default_scalar_type(1.0))
        self._area = fem.assemble_scalar(fem.form(one * dx))
        self._form_von_mises = fem.form(von_mises * dx)
        self._form_compliance = fem.form(inner(f, uh) * dx)

        # --- tip point lookup, done once ------------------------------------
        self._tip_pt = np.array([[tip_point[0], tip_point[1], 0.0]], dtype=np.float64)
        bb_tree = geometry.bb_tree(self._mesh, self._mesh.topology.dim)
        candidates = geometry.compute_collisions_points(bb_tree, self._tip_pt)
        colliding = geometry.compute_colliding_cells(
            self._mesh, candidates, self._tip_pt
        )
        self._tip_cells = colliding.links(0)[:1]
        if len(self._tip_cells) == 0:
            raise ValueError(
                f"tip_point {tip_point} is not inside the solve mesh; "
                "the tip-deflection QoI cannot be evaluated"
            )

    # -----------------------------------------------------------------------

    def _evaluate(self, state: np.ndarray):
        self._E1.value = state[0]
        self._E2.value = state[1]
        return self._problem.solve()

    def _compute_qoi(self) -> np.ndarray:
        """Scalar QoIs on this level's own mesh. Serial only -- under MPI each
        term needs an allreduce and the tip point lives on one rank."""
        uh = self._problem.u
        return np.array(
            [
                float(np.ravel(uh.eval(self._tip_pt, self._tip_cells))[1]),
                fem.assemble_scalar(self._form_von_mises) / self._area,
                fem.assemble_scalar(self._form_compliance),
            ]
        )

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Observation operator. Returns (observation vector, QoI vector)."""
        pde_sol = self._evaluate(state)
        observed = interpolate_nonmatching(
            self._V_observation,
            self._V,
            pde_sol,
            self._interpolation_data,
            self._cells_V_to,
        )
        return observed.x.array, self._compute_qoi()

    # -----------------------------------------------------------------------

    @property
    def gdim(self) -> int:
        return self._mesh.geometry.dim

    @property
    def ndofs(self) -> int:
        return self._V.dofmap.index_map.size_global


def build_hierarchy(
    n_levels: int = 3,
    base_divisions: tuple[int, int] = (50, 10),
    obs_divisions: tuple[int, int] = (15, 5),
    data_refinements: int = 5,
    corners: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (3.0, 1.0)),
) -> tuple[list[LinearElasticity], LinearElasticity]:
    """Uniformly refined mesh hierarchy sharing one observation mesh.

    Returns ``(levels, data_model)`` with ``levels`` ordered coarse -> fine.
    ``data_model`` lives on a more refined mesh than any level and is used only
    to generate the synthetic data, which avoids the inverse crime.
    """
    (x0, y0), (x1, y1) = corners
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([x0, y0]), np.array([x1, y1])],
        list(base_divisions),
    )
    msh.topology.create_entities(1)

    meshes = [msh]
    for _ in range(max(n_levels, data_refinements + 1) - 1):
        nxt = mesh.refine(meshes[-1])[0]
        nxt.topology.create_entities(1)
        meshes.append(nxt)

    # inset slightly so observation points stay strictly inside every level
    obs_msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[x0 + 0.05, y0 + 0.05], [x1 - 0.05, y1 - 0.05]],
        list(obs_divisions),
    )

    tip = (x1, 0.5 * (y0 + y1))
    levels = [LinearElasticity(m, obs_msh, tip_point=tip) for m in meshes[:n_levels]]
    data_model = LinearElasticity(meshes[data_refinements], obs_msh, tip_point=tip)
    return levels, data_model
