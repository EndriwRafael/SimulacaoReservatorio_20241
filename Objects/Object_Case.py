from abc import ABC, abstractmethod
import Functions
import os
from Objects import Objects_process as Obj


class ObjectCase(ABC):
    def __init__(self):
        self.rightflow = None
        self.leftflow = None
        self.topflow = None
        self.baseflow = None
        self.leftpress = None
        self.rightpress = None
        self.toppress = None
        self.basepress = None
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
        self.res_width = None
        self.well_pressure = None
        self.initial_pressure = None
        self.wellpositions = None
        self.results = None

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure

    @staticmethod
    def set_well_position(position):
        wells_p = {}
        for key, item in position.items():
            wells_p[key] = Obj.WellPosition(positionwell=item['Position'], radius=item['Radius'],
                                            pressure=item['Pressure'],
                                            flow=item['Flow'], typewell=item['Type'])
        return wells_p

    def set_case_parameters(self, initial_press: int or float, res_len: int or float,
                            viscosity: float, porosity: float, compressibility: float,
                            res_area: int or float, res_thick: float or int, well_press=None, permeability=None,
                            res_width=None,
                            wellflow=None, injectflow=None, leftflow=None, rightflow=None, topflow=None, baseflow=None,
                            left_press=None, right_press=None, top_press=None, base_press=None, well_position=None):
        self.initial_pressure = initial_press
        self.well_pressure = well_press
        self.res_length = res_len
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compressibility
        self.res_area = res_area
        self.res_thickness = res_thick
        self.res_width = res_width
        self.well_flow = wellflow
        self.injectflow = injectflow
        self.leftflow = leftflow
        self.rightflow = rightflow
        self.topflow = topflow
        self.baseflow = baseflow
        self.leftpress = left_press
        self.rightpress = right_press
        self.toppress = top_press
        self.basepress = base_press
        self.wellpositions = well_position if well_position is None else self.set_well_position(well_position)
        self.eta, self.deltaPressure = self.calc() if type(self.permeability) == float else (None, None)

    @abstractmethod
    def compute(self):
        """
        Method to compute the post process of the simulation for field pressure analysis.
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
        Functions.plot_graphs_2d(welldata=self.results.welldata, time_values=self.results.time, path=self.rootpath)
        Functions.plot_animation_map_2d(grid=self.results.mesh, name=self.results.name, path=self.rootpath)
