import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import solve_ivp
import tinyDA as tda
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import dx, grad, inner
from helpers import interpolate_nonmatching, plot_function
import pyvista
import pickle
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import arviz as az
import corner
import math

class LinearElasticity:
    def __init__(self, solve_mesh: mesh, observation_mesh: mesh) -> None:
        self._observation_mesh = observation_mesh
        self._mesh = solve_mesh
        self._V = fem.functionspace(self._mesh, ("Lagrange", 1, (self._mesh.geometry.dim, )))
        self._V_observation = fem.functionspace(self._observation_mesh, ("Lagrange", 1, (self._mesh.geometry.dim, )))

        # boundary conditions
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

        # coefficient setup
        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)
        f = fem.Constant(self._mesh, default_scalar_type((0, -0.1)))
        x = ufl.SpatialCoordinate(self._mesh)

        # ── Inferred parameters: E1, E2 (Young's moduli left/right half) ──
        nu = 0.3  # fixed Poisson's ratio
        self._E1 = fem.Constant(self._mesh, default_scalar_type(20.0))
        self._E2 = fem.Constant(self._mesh, default_scalar_type(10.0))

        # Lamé parameters derived from E and nu
        # lambda = E*nu / ((1+nu)*(1-2*nu)),  mu = E / (2*(1+nu))
        def lam(E):
            return E * nu / ((1 + nu) * (1 - 2 * nu))

        def mu(E):
            return E / (2 * (1 + nu))

        def E_field(x):
            bottom = np.min(self._mesh.geometry.x[:, 1])
            top    = np.max(self._mesh.geometry.x[:, 1])
            height = top - bottom
            return ufl.conditional(x[1] <= bottom + height / 2, self._E1, self._E2)

        def epsilon(u):
            return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

        def sigma(u):
            E = E_field(x)
            return lam(E) * ufl.nabla_div(u) * ufl.Identity(len(u)) + mu(E) * epsilon(u)

        # weak formulation
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(f, v) * dx

        self._problem = LinearProblem(
            a,
            L,
            bcs=[bc_left],
            petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
            petsc_options_prefix="lp",
        )

        # performance optimization
        self._cells_V_to = np.arange(
            self._V_observation.mesh.topology.index_map(
                self._V_observation.mesh.topology.dim
            ).size_local,
            dtype=np.int32,
        )
        self._interpolation_data = fem.create_interpolation_data(
            self._V_observation, self._V, self._cells_V_to
        )

    def _evaluate(self, state: np.ndarray) -> np.ndarray:
        # state = [log(E1), log(E2)]
        self._E1.value = state[0]
        self._E2.value = state[1]
        return self._problem.solve()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluation of the observation operator"""
        pde_sol = self._evaluate(state)
        observed = interpolate_nonmatching(
            self._V_observation,
            self._V,
            pde_sol,
            self._interpolation_data,
            self._cells_V_to,
        )
        return observed.x.array, 0  # placeholder for qoi

    @property
    def gdim(self):
        return self._mesh.geometry.dim

    @property
    def ndofs(self):
        return self._V.dofmap.index_map.size_global

    def plot(self, solution):
        """Plot forward model output, indicate regions of different E."""

        def E_numpy(x):
            bottom = np.min(self._mesh.geometry.x[:, 1])
            top    = np.max(self._mesh.geometry.x[:, 1])
            height = top - bottom
            return np.where(x[:, 1] <= bottom + height / 2, self._E1.value, self._E2.value)

        p = pyvista.Plotter()
        topology, cell_types, geometry = plot.vtk_mesh(self._V_observation)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        u_2d = solution.reshape((geometry.shape[0], 2))
        u_3d = np.column_stack([u_2d, np.zeros(u_2d.shape[0])])

        E_vals = E_numpy(geometry)

        grid["u"] = u_3d
        grid["E"] = E_vals
        warped = grid.warp_by_vector("u", factor=1.5)

        boundary = grid.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        p.add_mesh(boundary, color="k", line_width=2)
        p.add_mesh(warped, scalars="E", cmap="viridis", show_edges=True)
        p.add_scalar_bar(title="E (Young's modulus)")
        p.show_axes()
        if not pyvista.OFF_SCREEN:
            p.show()
        else:
            p.screenshot("deflection.png")

_msh = mesh.create_rectangle(MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([3, 1])],  # x goes from 0 → 3 instead of 0 → 1
    [50, 10])
_msh.topology.create_entities(1)

_msh2 = mesh.refine(_msh)[0]
_msh2.topology.create_entities(1)

_msh3 = mesh.refine(_msh2)[0]
_msh3.topology.create_entities(1)
_msh4 = mesh.refine(_msh3)[0]
_msh4.topology.create_entities(1)
_msh5 = mesh.refine(_msh4)[0]
_msh5.topology.create_entities(1)
_msh_data = mesh.refine(_msh5)[0]

_obs_msh = mesh.create_rectangle(
    MPI.COMM_WORLD, 
    [[0.05, 0.05], [2.95, 0.95]],  # slightly inset from [0,0] → [3,1]
    [15, 5]
)

model1 = LinearElasticity(_msh, _obs_msh)
model2 = LinearElasticity(_msh2, _obs_msh)
model3 = LinearElasticity(_msh3, _obs_msh)
model_data = LinearElasticity(_msh_data, _obs_msh)

E1 = 80
E2 = 40

true_parameters = np.array([E1, E2]) # no log
x_true = model_data(true_parameters)[0] # true displacement
x_true.shape

sigma = 0.005
noise = np.random.normal(scale=sigma, size=(x_true.shape[0]))
data = x_true+noise

prior_E1 = stats.uniform(loc=50, scale=50)
prior_E2 = stats.uniform(loc=0.1, scale=50)

my_prior = tda.CompositePrior([prior_E1, prior_E2])
cov_likelihood = sigma**2*np.eye(data.size)
cov_loglike_gp = sigma**2*np.eye(data.size)

my_loglike_1 = tda.AdaptiveGaussianLogLike(data, cov_likelihood)
my_loglike_2 = tda.AdaptiveGaussianLogLike(data, cov_likelihood)
my_loglike_3 = tda.GaussianLogLike(data, cov_likelihood)

my_posterior_l2 = tda.Posterior(my_prior, my_loglike_2, model2)
my_posterior_l1 = tda.Posterior(my_prior, my_loglike_1, model1)
my_posterior_l3 = tda.Posterior(my_prior, my_loglike_3, model3)

my_posteriors_ml = [my_posterior_l1, my_posterior_l3]

my_posterior = tda.Posterior(my_prior, my_loglike_2, model2)

MAP = tda.get_MAP(my_posterior_l1)

# Adaptive Metropolis
am_cov = np.eye(true_parameters.size)
am_t0 = 100
am_sd = None
am_epsilon = 1e-6
am_adaptive = True
my_proposal = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)

my_chain_l1 = tda.sample(my_posterior_l1, my_proposal, iterations=20000, n_chains=1, initial_parameters=true_parameters) # 3 levels
idata1 = tda.to_inference_data(my_chain_l1, burnin=5000, parameter_names = ['E1', 'E2'])

my_chain_l2 = tda.sample(my_posterior_l2, my_proposal, iterations=20000, n_chains=1, initial_parameters=true_parameters) # 3 levels
idata2 = tda.to_inference_data(my_chain_l2, burnin=5000, parameter_names = ['E1', 'E2'])

my_chain_l3 = tda.sample(my_posterior_l3, my_proposal, iterations=20000, n_chains=1, initial_parameters=true_parameters) # 3 levels
idata3 = tda.to_inference_data(my_chain_l3, burnin=5000, parameter_names = ['E1', 'E2'])

my_chain_ml = tda.sample([my_posterior_l1, my_posterior_l3], my_proposal, iterations=3500, n_chains=1, initial_parameters=MAP, subchain_length=10, adaptive_error_model='state-independent') # 2 levels
idata4 = tda.to_inference_data(my_chain_ml, level='fine', burnin=500, parameter_names = ['E1', 'E2'])

my_chain_m3l = tda.sample([my_posterior_l1, my_posterior_l2, my_posterior_l3], my_proposal, iterations=2500, n_chains=1, initial_parameters=MAP, subchain_length=[5,5], adaptive_error_model='state-independent') # 2 levels
idata5 = tda.to_inference_data(my_chain_m3l, level='2', burnin=500, parameter_names = ['E1', 'E2'])




