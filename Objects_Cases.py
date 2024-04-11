class PressureBoundaries:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = int(reserlength)
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.eta, self.deltaPressure = self.calc()

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure


class FlowBoundaries:
    def __init__(self, initial_press: float, well_press: float, res_len: float, permeability: float, viscosity: float,
                 porosity: float, compressibility: float, res_area: float, res_thick: float or int, wellflow: float,
                 injectflow: float):
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
        self.eta = self.calc_eta()

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta


class FlowPressureBoundaries:
    def __init__(self, initial_press: float, well_press: float, res_len: float, permeability: float, viscosity: float,
                 porosity: float, compressibility: float, res_area: float, res_thick: float or int, wellflow: float):
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
        self.eta = self.calc_eta()

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta
