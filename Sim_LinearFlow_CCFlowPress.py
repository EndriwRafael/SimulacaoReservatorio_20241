"""
Simuladores para fluxo linear e radial - Métodos analitico e numérico
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e Pressão na fronteira.

Método FTCS -- Forward in Time Centered Space
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os
from scipy.special import erfc
import sys


class NumericalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.wellclass = well_class
        self.pressure = None
        self.start_simulate()

    def start_simulate(self):
        pass


class AnaliticalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.pressure = None
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def plot_results(self):
        if not os.path.isdir(f'results\\Simulador_Fluxo-Pressao'):
            os.makedirs(f'results\\Simulador_Fluxo-Pressao')

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in self.pressure.columns:
            if column in time_to_plot:
                plt.plot(self.pressure.index, self.pressure[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.png')
        plt.close()

        self.pressure.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.xlsx')

    def start_simulate(self):
        pressure = {}
        for time in self.time:
            vetor_to_time = []
            for key in self.well_class.mesh.keys():
                if time == 0:
                    vetor_to_time.append(self.well_class.initial_pressure)
                else:
                    a = ((self.well_class.well_flow * self.well_class.viscosity) /
                         (self.well_class.permeability * self.well_class.res_area))
                    b = np.sqrt((4 * self.well_class.eta * time) / np.pi)
                    c = np.exp(self.well_class.mesh[key] ** 2 / (-4 * self.well_class.eta * time))
                    d = self.well_class.mesh[key]
                    e = erfc(self.well_class.mesh[key] / np.sqrt(4 * self.well_class.eta * time))
                    vetor_to_time.append(self.well_class.initial_pressure - (a * ((b * c) - (d * e))))

            pressure[time] = vetor_to_time

            # Setting the mesh points as the dataframe index
            index_for_dataframe = [round(self.well_class.mesh[key], ndigits=3) for key in self.well_class.mesh.keys()]
            pressure = {'x': index_for_dataframe, **pressure}
            # Creating dataframe from the variable 'pressure'
            pressure = Df(pressure).set_index('x')

        self.pressure = pressure
        self.plot_results()


class InitializeData:
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
        self.mesh, self.deltax, self.n_cells, self.rx = None, None, None, None

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta
