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
        self.eta, self.deltaPressure = self.calc()

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure

    def calc_sum(self, time, point):
        if time == 0 or point == 0:
            sum_value = 0
        else:
            sum_value = 0
            n = 1
            erro = 100
            eppara = 0.001
            while erro >= eppara:
                sum_old = sum_value
                sum_value += ((np.exp(-1 * (((n * np.pi) / self.res_length) ** 2) * (self.eta * time)) / n) * np.sin(
                    (n * np.pi * point)
                    / self.res_length))
                erro = np.fabs(sum_value - sum_old)/sum_value * 100
                n += 1
        return sum_value

    def start_simulation(self, x: np.ndarray, t: np.ndarray, deltapressure: float, wellpressure: float):
        pressure = {}
        for time in t:
            vetor_for_time = []

            for point in x:
                suma = self.calc_sum(time, point)
                vetor_for_time.append(deltapressure * suma + wellpressure)

            pressure['time'] = vetor_for_time

        return pressure
