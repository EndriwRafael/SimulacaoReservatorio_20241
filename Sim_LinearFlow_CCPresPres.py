"""
Simuladores para fluxo linear e radial - Métodos analitico e numérico
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.

Método FTCS -- Forward in Time Centered Space
"""
import Functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os


class NumericalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results\\Simulador_Pressao-Pressao'):
            os.makedirs(f'results\\Simulador_Pressao-Pressao')

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
        plt.title('Pressão-Pressão [Numérico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.well_class.n_cells)

        last_column = None
        for i_col in col_idx:

            if i_col == 0.0:

                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_column = i_col

            else:

                for j_row in row_idx:

                    if j_row == 0:
                        pressure_df.loc[j_row, i_col] = self.well_class.well_pressure

                    elif j_row == self.well_class.n_cells + 1:
                        pressure_df.loc[j_row, i_col] = self.well_class.initial_pressure

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = (8 / 3) * self.well_class.eta * self.well_class.rx * self.well_class.well_pressure
                            b = (1 - (4 * self.well_class.eta * self.well_class.rx)) * p1_t
                            c = (4 / 3) * self.well_class.eta * self.well_class.rx * p2_t
                            pressure_df.loc[j_row, i_col] = a + b + c

                        elif j_row == self.well_class.n_cells:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = (4 / 3) * self.well_class.eta * self.well_class.rx * p_n_1_t
                            b = (1 - (4 * self.well_class.eta * self.well_class.rx)) * p_n_t
                            c = (8 / 3) * self.well_class.eta * self.well_class.rx * self.well_class.initial_pressure
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.well_class.eta * self.well_class.rx * pi_1
                            b = (1 - (2 * self.well_class.eta * self.well_class.rx)) * pi_t
                            c = self.well_class.eta * self.well_class.rx * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c

                last_column = i_col
        self.plot_results(data=pressure_df)


class AnaliticalAnalysis:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def calc_sum(self, t_value, point_value):
        """
        Function that will calculate the sum that is required to plot the pressure field in the analitical solution.

        :param t_value: Value of time. Must be an integer of float.
        :param point_value: Position in the mesh. Must be an integer or float.
        :return: The sum needed to calculate the analitical solution.
        """

        # Value of the sum for n = 1:
        sum_value = ((np.exp(-1 * (((1 * np.pi) / self.well_class.res_length) ** 2) * (self.well_class.eta *
                                                                                       t_value)) / 1) *
                     np.sin((1 * np.pi * point_value) / self.well_class.res_length))

        # Now the iterative process begins:
        n = 2
        erro = 100
        eppara = 0.000001
        while erro >= eppara:
            sum_old = sum_value
            sum_value += ((np.exp(-1 * (((n * np.pi) / self.well_class.res_length) ** 2) * (self.well_class.eta *
                                                                                            t_value)) / n) *
                          np.sin((n * np.pi * point_value) / self.well_class.res_length))
            erro = np.fabs(sum_value - sum_old) / sum_value
            n += 1

        return sum_value

    def plot_result(self, data: Df):
        if not os.path.isdir(f'results\\Simulador_Pressao-Pressao'):
            os.makedirs(f'results\\Simulador_Pressao-Pressao')

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
        plt.title('Pressão-Pressão [Analítico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_analitico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_analitico.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.well_class.n_cells)

        for time in col_idx:
            if time == 0:
                for x in row_idx:
                    pressure_df.loc[x, time] = self.well_class.initial_pressure
            else:
                for x in row_idx:
                    if x == 0:
                        pressure_df.loc[x, time] = self.well_class.well_pressure
                    else:
                        suma = self.calc_sum(time, self.well_class.mesh[x])
                        value = ((self.well_class.deltaPressure * (self.well_class.mesh[x] /
                                                                   self.well_class.res_length +
                                                                   ((2 / np.pi) * suma))) +
                                 self.well_class.well_pressure)
                        pressure_df.loc[x, time] = value

        self.plot_result(data=pressure_df)


class InitializeData:
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
