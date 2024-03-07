import matplotlib.pyplot as plt
import numpy as np


class SimuladorPressPress:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = reserlength
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.result = self.calc()
        self.plot_result()

    def calc(self) -> int:
        delta_pressure = self.initial_pressure - self.well_pressure
        return delta_pressure

    def plot_result(self):
        print(self.result)
