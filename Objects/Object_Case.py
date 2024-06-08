from abc import ABC, abstractmethod
import numpy as np
import Functions
import os


class ObjectCase(ABC):
    def __init__(self):
        self.deltaPressure = None
        self.eta = None
        self.injectflow = None
        self.well_flow = None
        self.res_thickness = None
        self.res_area = None
        self.compressibility = None
        self.porosity = None
        self.viscosity = None
        self.permeability = None
        self.res_length = None
        self.well_pressure = None
        self.initial_pressure = None
        self.results = None

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure

    def set_case_parameters(self, initial_press: int or float, well_press: int or float, res_len: int or float,
                            permeability: float, viscosity: float, porosity: float, compressibility: float,
                            res_area: int or float, res_thick: float or int, wellflow=0, injectflow=0):
        self.initial_pressure = initial_press
        self.well_pressure = well_press
        self.res_length = res_len
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compressibility
        self.res_area = res_area
        self.res_thickness = res_thick
        self.well_flow = wellflow
        self.injectflow = injectflow
        self.eta, self.deltaPressure = self.calc()

    @abstractmethod
    def compute(self):
        """
        Method to compute the field pressure error for the problem methods used compared with analitical method.
        ----------------------------------------------------------------------------------------------------------------
        """
        pass


class OneDimensionalFlowCase(ObjectCase):
    def __init__(self, condiction):
        super().__init__()
        self.condiction = condiction
        self.rootpath = f'results\\OneDimensionalFlow\\{self.condiction}'
        self.createpath()

    def createpath(self):
        if not os.path.isdir(f'.\\{self.rootpath}'):
            os.makedirs(f'.\\{self.rootpath}')

    def compute(self):
        results = self.results
        extract_columns = [col for col in results.implicit.columns if col != 0.0 and col in results.explicit.columns]

        # --------------------------------------------------------------------------------------------------------------
        erro_explicit_dx, erro_explicit_dt = Functions.fo_erro(data_analitical=results.analitical_explicit,
                                                               data_method=results.explicit,
                                                               columns=extract_columns, n_cell=results.ncell_explicit)
        erro_implicit_dx, erro_implicit_dt = Functions.fo_erro(data_analitical=results.analitical_implicit,
                                                               data_method=results.implicit,
                                                               columns=extract_columns, n_cell=results.ncell_implicit)
        # --------------------------------------------------------------------------------------------------------------
        list_explicit = [results.dx_explicit, results.time_explicit[1] - results.time_explicit[0],
                         results.ncell_explicit, results.timeprocess_explicit]
        for i in erro_explicit_dx:
            list_explicit.append(i)
        for i in erro_explicit_dt:
            list_explicit.append(i)

        list_implicit = [results.dx_implicit, results.time_implicit[1] - results.time_implicit[0],
                         results.ncell_implicit, results.timeprocess_implicit]
        for i in erro_implicit_dx:
            list_implicit.append(i)
        for i in erro_implicit_dt:
            list_implicit.append(i)
        # --------------------------------------------------------------------------------------------------------------
        data = Functions.create_errordataframe_1d(explit_list=list_explicit, implicit_list=list_implicit,
                                                  columns=extract_columns)
        data.to_excel(f'{self.rootpath}\\Error_analysis.xlsx')

        Functions.plot_graphs(dataclass=results, columns=[10, 50, 100], root=self.rootpath)
        Functions.plot_animation_results(data=results, root=self.rootpath)
        Functions.plot_pressuremap_animation(data=results, root=self.rootpath)


class TwoDimensionalFlowCase(ObjectCase):
    def __init__(self, condiction):
        super().__init__()
        self.condiction = condiction
        self.rootpath = f'results\\TwoDimensionalFlow\\{self.condiction}'
        self.createpath()

    def createpath(self):
        if not os.path.isdir(f'.\\{self.rootpath}'):
            os.makedirs(f'.\\{self.rootpath}')

    def compute(self):
        return
