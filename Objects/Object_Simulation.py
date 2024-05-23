import Functions
import numpy as np
from abc import ABC, abstractmethod
from Objects import Object_Result


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

        self.wellclass.n_cells_explicit = n_cells_explicit
        self.wellclass.n_cells_implicit = n_cells_implicit
        self.wellclass.n_cells_analitical = 1000
        self.wellclass.analitical_mesh, self.wellclass.dx_analitical = Functions.create_mesh_1d(
            time_values=time_implicit, n_cells=1000, wellclass=self.wellclass, method='Analitical'
        )

    def simulate(self):
        analitical, explicit, implicit = Functions.set_object_simulation(boundaries=self.wellclass.condiction)
        # --------------------------------------------------------------------------------------------------------------
        analitical.set_parameters(t=self.wellclass.time_implicit, well_class=self.wellclass, name_file='Analitical')
        analitical.start_simulate()
        result_analitical = analitical.dataframe

        # --------------------------------------------------------------------------------------------------------------
        explicit.set_parameters(t=self.wellclass.time_explicit, well_class=self.wellclass, name_file='explicit')
        explicit.start_simulate()
        result_explicit = explicit.dataframe

        self.wellclass.analitical_mesh = self.wellclass.explicit_mesh
        self.wellclass.n_cells_analitical = self.wellclass.n_cells_explicit
        analitical.set_parameters(t=self.wellclass.time_explicit, well_class=self.wellclass,
                                  name_file='AnaliticalForExplicit')
        analitical.start_simulate()
        result_ana_exp = analitical.dataframe

        # --------------------------------------------------------------------------------------------------------------
        implicit.set_parameters(t=self.wellclass.time_implicit, well_class=self.wellclass, name_file='Implicit')
        implicit.start_simulate()
        result_implicit = implicit.dataframe

        self.wellclass.analitical_mesh = self.wellclass.implicit_mesh
        self.wellclass.n_cells_analitical = self.wellclass.n_cells_implicit
        analitical.set_parameters(t=self.wellclass.time_implicit, well_class=self.wellclass,
                                  name_file='AnaliticalForImplicit')
        analitical.start_simulate()
        result_ana_imp = analitical.dataframe

        # --------------------------------------------------------------------------------------------------------------
        self.wellclass.results = Object_Result.ResultsOneDimFlow(
            data_analitical=result_analitical,
            data_explicit=result_explicit,
            data_implicit=result_implicit,
            data_ana_exp=result_ana_exp,
            data_ana_imp=result_ana_imp,
            n_cell_explicit=self.wellclass.n_cells_explicit,
            n_cell_implicit=self.wellclass.n_cells_implicit,
            dx_explicit=self.wellclass.dx_explicit,
            dx_implicit=self.wellclass.dx_implicit,
            t_process_explicit=self.wellclass.time_TO_explicit,
            t_process_implicit=self.wellclass.time_TO_implicit,
            mesh_explicit=self.wellclass.explicit_mesh,
            mesh_implicit=self.wellclass.implicit_mesh,
            time_mesh_explicit=self.wellclass.time_explicit,
            time_mesh_implicit=self.wellclass.time_implicit
        )

        # --------------------------------------------------------------------------------------------------------------
        del (self.wellclass.dx_implicit, self.wellclass.dx_explicit, self.wellclass.dx_analitical,
             self.wellclass.n_cells_implicit, self.wellclass.n_cells_explicit, self.wellclass.n_cells_analitical,
             self.wellclass.time_TO_implicit, self.wellclass.time_TO_explicit, self.wellclass.explicit_mesh,
             self.wellclass.implicit_mesh, self.wellclass.analitical_mesh, self.wellclass.time_explicit,
             self.wellclass.time_implicit)


class TwoDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}


class ThreeDimensionalFlowMesh:
    def __init__(self):
        self.mesh = {}
