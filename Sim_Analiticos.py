"""
Simuladores para fluxo linear e radial - Métodos analiticos
"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as Df
import os
from scipy.special import erfc, expn


class SimuladorPressPressLinear:
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
        if point == 0:
            sum_value = 0
            return sum_value
        else:
            sum_value = round(((np.exp(-1 * (((1 * np.pi) / self.res_length) ** 2) * (self.eta * time)) / 1) *
                               np.sin((1 * np.pi * point) / self.res_length)), ndigits=5)
            n = 2
            erro = 100
            eppara = 0.001
            while erro >= eppara:
                sum_old = sum_value
                sum_value += round(((np.exp(-1 * (((n * np.pi) / self.res_length) ** 2) * (self.eta * time)) / n) *
                                    np.sin((n * np.pi * point) / self.res_length)), ndigits=5)
                erro = np.fabs(sum_value - sum_old) / sum_value
                n += 1
            return sum_value

    @staticmethod
    def plot_results(data: Df):
        if not os.path.isdir(f'results\\Simulador_Pressao-Pressao'):
            os.makedirs(f'results\\Simulador_Pressao-Pressao')

        for column in data.columns:
            time_num = float(column.split(' ')[1])
            plt.plot(data.index, data[column], label=f't = {int(time_num)}')
        # plt.plot(data.index, data['time 50.0'])
        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_analitico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_analitico.xlsx')

    def start_simulation(self, x: np.ndarray, t: np.ndarray):
        pressure = {}
        for time in t:
            vetor_for_time = []

            for point in x:
                suma = self.calc_sum(time, point)
                vetor_for_time.append((self.deltaPressure * (point / self.res_length + ((2 / np.pi) * suma))) +
                                      self.well_pressure)

            pressure[f'time {time}'] = vetor_for_time

        data_pressure = Df(pressure).set_index(x)
        return self.plot_results(data_pressure)


class SimuladorFlowPressureLinear:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility,
                 reserarea, reserthick, wellflow):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = reserlength
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.eta = self.calc_eta()
        self.area = reserarea
        self.thickness = reserthick
        self.flow = wellflow

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results\\Simulador_Fluxo-Pressao'):
            os.makedirs(f'results\\Simulador_Fluxo-Pressao')

        for column in data.columns:
            time_num = float(column.split(' ')[1])
            plt.plot(data.index, data[column], label=f't = {int(time_num)}')
        # plt.plot(data.index, data['time 50.0'])
        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.xlsx')

    def start_simulation(self, x: np.ndarray, t: np.ndarray):
        pressure = {}
        for time in t:
            vetor_in_time = []

            for position in x:
                a = ((self.flow * self.viscosity) / (self.permeability * self.area))
                b = np.sqrt((4 * self.eta * time) / np.pi)
                c = np.exp(position ** 2 / (-4 * self.eta * time))
                d = position
                e = erfc(position / np.sqrt(4 * self.eta * time))
                vetor_in_time.append(self.initial_pressure - (a * ((b * c) - (d * e))))

            pressure[f'time {time}'] = vetor_in_time

        datapressure = Df(pressure).set_index(x)

        return self.plot_results(data=datapressure)


class SimuladorFlowPressureRadial:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility,
                 reserthick, wellflow):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = reserlength
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.eta = self.calc_eta()
        self.thickness = reserthick
        self.flow = wellflow

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results\\Simulador_Fluxo-Pressao_Radial'):
            os.makedirs(f'results\\Simulador_Fluxo-Pressao_Radial')

        for column in data.columns:
            time_num = float(column.split(' ')[1])
            plt.plot(data.index, data[column], label=f't = {int(time_num)}')
        # plt.plot(data.index, data['time 50.0'])
        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao_Radial\\fluxo-pressao_radial_analitico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao_Radial\\fluxo-pressao_radial_analitico.xlsx')

    def start_simulation(self, x: np.ndarray, t: np.ndarray):
        pressure = {}

        for time in t:
            vetor_in_time = []

            for position in x:
                a = (self.flow * self.viscosity) / (4 * np.pi * self.permeability * self.thickness)
                b = expn(1, (self.porosity * self.viscosity * self.compressibility * position ** 2) /
                         (4 * self.permeability * time))
                vetor_in_time.append(self.initial_pressure - (a * b))

            pressure[f'time {time}'] = vetor_in_time

        datapressure = Df(pressure).set_index(x)

        return self.plot_results(data=datapressure)
