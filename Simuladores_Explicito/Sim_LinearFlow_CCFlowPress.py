"""
Simuladores para fluxo linear e radial - Métodos analitico e numérico
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e Pressão na fronteira.

Método FTCS -- Forward in Time Centered Space
"""
import Functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os
from scipy.special import erfc


class NumericalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.wellclass = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Fluxo-Pressao'):
            os.makedirs(f'../results/Simulador_Fluxo-Pressao')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.wellclass.mesh[key], ndigits=3) for key in self.wellclass.mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Flow-Pressão [Numérico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.wellclass.n_cells)

        last_column = None
        for i_col in col_idx:

            if i_col == 0.0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.wellclass.initial_pressure
                last_column = i_col

            else:
                for j_row in row_idx:

                    if j_row == 0:
                        p0_t = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto 1, tempo anterior.
                        pressure_df.loc[j_row, i_col] = p0_t - (((self.wellclass.well_flow * self.wellclass.viscosity)
                                                                 / (self.wellclass.permeability *
                                                                    self.wellclass.res_area)) * self.wellclass.deltax/2)

                    elif j_row == self.wellclass.n_cells + 1:
                        pressure_df.loc[j_row, i_col] = self.wellclass.initial_pressure

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = self.wellclass.eta * self.wellclass.rx * p2_t
                            b = (1 - (self.wellclass.eta * self.wellclass.rx)) * p1_t
                            c = self.wellclass.eta * self.wellclass.rx * self.wellclass.well_flow * \
                                self.wellclass.viscosity * self.wellclass.deltax / \
                                (self.wellclass.permeability * self.wellclass.res_area)
                            pressure_df.loc[j_row, i_col] = a + b - c

                        elif j_row == self.wellclass.n_cells:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = (4 / 3) * self.wellclass.eta * self.wellclass.rx * p_n_1_t
                            b = (1 - (4 * self.wellclass.eta * self.wellclass.rx)) * p_n_t
                            c = (8 / 3) * self.wellclass.eta * self.wellclass.rx * self.wellclass.initial_pressure
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.wellclass.eta * self.wellclass.rx * pi_1
                            b = (1 - (2 * self.wellclass.eta * self.wellclass.rx)) * pi_t
                            c = self.wellclass.eta * self.wellclass.rx * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c
                last_column = i_col
        self.plot_results(data=pressure_df)


class AnaliticalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.pressure = None
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Fluxo-Pressao'):
            os.makedirs(f'../results/Simulador_Fluxo-Pressao')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.mesh[key], ndigits=3) for key in self.well_class.mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Fluxo-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_analitico.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.well_class.n_cells)

        for time in col_idx:
            if time == 0:
                for x in row_idx:
                    pressure_df.loc[x, time] = self.well_class.initial_pressure

            else:
                for x in row_idx:
                    a = ((self.well_class.well_flow * self.well_class.viscosity) /
                         (self.well_class.permeability * self.well_class.res_area))
                    b = np.sqrt((4 * self.well_class.eta * time) / np.pi)
                    c = np.exp(self.well_class.mesh[x] ** 2 / (-4 * self.well_class.eta * time))
                    d = self.well_class.mesh[x]
                    e = erfc(self.well_class.mesh[x] / np.sqrt(4 * self.well_class.eta * time))
                    value = (self.well_class.initial_pressure - (a * ((b * c) - (d * e))))
                    pressure_df.loc[x, time] = value

        self.plot_results(data=pressure_df)


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

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta
