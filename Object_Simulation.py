import Functions
import numpy as np
from abc import ABC, abstractmethod
from Simuladores_1DFlow import Analitical, Numerical_EXPLICIT, Numerical_IMPLICIT


class Simulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self):
        """
        Method to start the simulation
        :return: The results in pressure for the mesh created in any flow dinamics set
        """
        pass


class OneDimensionalFlowMesh(Simulator):
    def __init__(self, wellcase):
        super().__init__()
        self.wellclass = wellcase

    def set_mesh_grid(self, time_explicit: np.ndarray, n_cells_explicit: int,
                      time_implicit: np.ndarray, n_cells_implicit: int):
        self.wellclass.explicit_mesh, self.wellclass.dx_explicit = Functions.create_mesh_1d(time_values=time_explicit,
                                                                                            n_cells=n_cells_explicit,
                                                                                            wellclass=self.wellclass,
                                                                                            method='Explicit')

        self.wellclass.implicit_mesh, self.wellclass.dx_implicit = Functions.create_mesh_1d(time_values=time_implicit,
                                                                                            n_cells=n_cells_implicit,
                                                                                            wellclass=self.wellclass,
                                                                                            method='Implicit')

    def simulate(self):
        self.wellclass.analitical_mesh = self.wellclass.explicit_mesh
        return


class TwoDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}


class ThreeDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}
