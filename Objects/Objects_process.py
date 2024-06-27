import pandas as pd


class WellPosition:
    def __init__(self, positionwell, radius, pressure, flow, typewell):
        self.line = positionwell[0]
        self.column = positionwell[1]
        self.radius = radius
        self.pressure = pressure
        self.flow = flow
        self.type = typewell


class ResultsOneDimFlow:
    def __init__(self, data_analitical, data_explicit, data_implicit, data_ana_exp, data_ana_imp, n_cell_explicit,
                 n_cell_implicit, dx_explicit, dx_implicit, t_process_explicit, t_process_implicit, mesh_explicit,
                 mesh_implicit, time_mesh_explicit, time_mesh_implicit):
        self.analitical = data_analitical
        self.explicit = data_explicit
        self.implicit = data_implicit
        self.analitical_explicit = data_ana_exp
        self.analitical_implicit = data_ana_imp
        self.ncell_explicit = n_cell_explicit
        self.ncell_implicit = n_cell_implicit
        self.dx_explicit = dx_explicit
        self.dx_implicit = dx_implicit
        self.time_explicit = time_mesh_explicit
        self.time_implicit = time_mesh_implicit
        self.mesh_explicit = mesh_explicit
        self.mesh_implicit = mesh_implicit
        self.timeprocess_explicit = t_process_explicit
        self.timeprocess_implicit = t_process_implicit


class ResultsTwoDimFlow:
    def __init__(self, name: str, mesh: dict, cells: int, time: float, x_points: list, y_points: list, welldata: dict,
                 time_process: float):
        self.name = name
        self.mesh = mesh
        self.n_cells = cells
        self.time = time
        self.welldata = welldata
        self.time_process = time_process
        self.set_index_mesh(x=x_points, y=y_points)

    def set_index_mesh(self, x: list, y: list):
        for _, grid in self.mesh.items():
            grid = grid.set_index(pd.Index(y))
            grid.columns = x
            self.mesh[_] = grid
