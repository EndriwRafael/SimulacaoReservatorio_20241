from abc import ABC, abstractmethod
from Simuladores import Analitical_OneDimensional, Explicit_OneDimensional, Explicit_OneDimensional


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
        Method to compute the field pressure for the problem
        Returns the results simulation for all methods (explicit and implicit)
        """
        pass


class PressureBoundaries(ObjectCase):
    def __init__(self):
        super().__init__()
        self.condiction = 'PP'

    def compute(self):
        pass


class FlowPressureBoundaries(ObjectCase):
    def __init__(self):
        super().__init__()
        self.condiction = 'FP'

    def compute(self):
        pass


class FlowBoundaries(ObjectCase):
    def __init__(self):
        super().__init__()
        self.condiction = 'FF'

    def compute(self):
        pass
