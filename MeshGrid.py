import Functions
import numpy as np


class OneDimensionalFlowMesh:
    def __init__(self):
        pass

    @staticmethod
    def set_mesh_grid(time_explicit: np.ndarray, n_cells_explicit: int,
                      time_implicit: np.ndarray, n_cells_implicit: int,
                      wellclass: object):

        wellclass.explicit_mesh, wellclass.dx_explicit = Functions.create_mesh_1d(time_values=time_explicit,
                                                                                  n_cells=n_cells_explicit,
                                                                                  wellclass=wellclass,
                                                                                  method='Explicit')

        wellclass.implicit_mesh, wellclass.dx_implicit = Functions.create_mesh_1d(time_values=time_implicit,
                                                                                  n_cells=n_cells_implicit,
                                                                                  wellclass=wellclass,
                                                                                  method='Implicit')


class TwoDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}


class ThreeDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}
