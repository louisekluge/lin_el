import pyvista as pv
from dolfinx import fem, mesh, plot, default_scalar_type
import numpy as np

#from ls_mcmc.algorithms import MCMCAlgorithm, _CachedArgs
#from ls_mcmc import model
from dataclasses import dataclass

# plotting help
# pv.set_jupyter_backend("trame")
# pv.global_theme.trame.server_proxy_enabled=True
def plot_function(V: fem.FunctionSpace, v: fem.Function):
    _v = fem.Function(V)
    _v.x.array[:]=v
    p = pv.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    
    # Attach vector values to grid and warp grid by vector
    u_2d = _v.x.array.reshape((geometry.shape[0], 2))
    u_3d = np.column_stack((u_2d, np.zeros(u_2d.shape[0])))
    grid["u"] = u_3d
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if not pv.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("deflection.png")

    #plotter.show()

def plot_function_3d(V: fem.FunctionSpace, v: fem.Function):
    _v = fem.Function(V)
    _v.x.array[:]=v
    p = pv.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    
    # Attach vector values to grid and warp grid by vector
    u_3d = _v.x.array.reshape((geometry.shape[0], 3))
    #u_3d = np.column_stack((u_2d, np.zeros(u_2d.shape[0])))
    grid["u"] = u_3d
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if not pv.OFF_SCREEN:
        p.show()
    else:
        figure_as_array = p.screenshot("deflection.png")

    #plotter.show()



# interpolation from calculation mesh to observation mesh
def interpolate_nonmatching(
    V_to: fem.FunctionSpace,
    V_from: fem.FunctionSpace,
    func: fem.Function,
    interpolation_data=None,
    cells_V_to=None,
) -> fem.Function:
    func_V_to = fem.Function(V_to)
    if (interpolation_data is None) or (cells_V_to is None):
        cells_V_to = np.arange(
            V_to.mesh.topology.index_map(
                V_to.mesh.topology.dim
            ).size_local,
            dtype=np.int32,
        )
        interpolation_data = fem.create_interpolation_data(
            V_to, V_from, cells_V_to
        )
    func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
    func_V_to.x.scatter_forward()
    return func_V_to

