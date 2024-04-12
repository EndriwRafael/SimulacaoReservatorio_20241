"""
------------------------------------------------------------------------------------------------------------------------
Simuladores para fluxo linear e radial - Métodos numéricos.
------------------------------------------------------------------------------------------------------------------------
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e pressão na fronteira.
Condições de Contorno de Neumann - Fluxo no poço e na fronteira.
------------------------------------------------------------------------------------------------------------------------
Método BTCS -- Backward in Time Centered Space
------------------------------------------------------------------------------------------------------------------------
"""
import Functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os
from scipy.linalg import solve


class PressureBoundaries:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Pressao-Pressao'):
            os.makedirs(f'../results/Simulador_Pressao-Pressao')

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
        plt.title('Pressão-Pressão [Numérico - Implicita]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico_Implicit.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico_Implicit.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.well_class.n_cells)

        param = {
            'a': 1 + (4 * self.well_class.rx * self.well_class.eta),
            'b': - (4 / 3) * self.well_class.rx * self.well_class.eta,
            'c': - self.well_class.rx * self.well_class.eta,
            'd': 1 + (2 * self.well_class.rx * self.well_class.eta),
            'f1': (8 / 3) * self.well_class.rx * self.well_class.eta * self.well_class.well_pressure,
            'fn': (8 / 3) * self.well_class.rx * self.well_class.eta * self.well_class.initial_pressure
        }

        pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_pressureboundaries(
            n_cells=self.well_class.n_cells, param_values=param)
        last_col = None
        for i_col in col_idx:

            if i_col == 0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_col = i_col
            else:
                vetor_last_pressure = [pressure_df.loc[i, last_col] for i in row_idx if i != 0
                                       and i != self.well_class.n_cells + 1]
                b = vetor_last_pressure + const_matrix
                vetor_next_pressure = solve(pressure_matrix_, b)

                vetor_next_pressure = np.insert(vetor_next_pressure, 0, self.well_class.well_pressure)
                vetor_next_pressure = np.append(vetor_next_pressure, self.well_class.initial_pressure)

                pressure_df[i_col] = vetor_next_pressure

                last_col = i_col

        self.plot_results(data=pressure_df)


class WellFlowAndPressureBoundaries:
    def __init__(self, t: np.ndarray, well_class: object):
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
        plt.title('Flow-Pressão [Numérico - Implicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico_Implicit.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico_Implicit.xlsx')

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time, n_cells=self.well_class.n_cells)

        param = {
            'a': 1 + (self.well_class.rx * self.well_class.eta),
            'b': - self.well_class.rx * self.well_class.eta,
            'c': 1 + (2 * self.well_class.rx * self.well_class.eta),
            'd': - (4 / 3) * self.well_class.rx * self.well_class.eta,
            'e': 1 + (4 * self.well_class.rx * self.well_class.eta),
            'f1': - (self.well_class.rx * self.well_class.eta * self.well_class.well_flow * self.well_class.viscosity *
                     self.well_class.deltax) / (self.well_class.permeability * self.well_class.res_area),
            'fn': (8 / 3) * self.well_class.rx * self.well_class.eta * self.well_class.initial_pressure
        }

        pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_flowboundaries(
            n_cells=self.well_class.n_cells, param_values=param)
        last_col = None
        for i_col in col_idx:

            if i_col == 0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_col = i_col
            else:
                vetor_last_pressure = [pressure_df.loc[i, last_col] for i in row_idx if i != 0
                                       and i != self.well_class.n_cells + 1]
                b = vetor_last_pressure + const_matrix
                vetor_next_pressure = solve(pressure_matrix_, b)

                p1_t = pressure_df[last_col][1]  # pressão no ponto 1, tempo anterior.
                value_to_add = p1_t - (((self.well_class.well_flow * self.well_class.viscosity) /
                                        (self.well_class.permeability * self.well_class.res_area)) *
                                       (self.well_class.deltax / 2))

                vetor_next_pressure = np.insert(vetor_next_pressure, 0, value_to_add)
                vetor_next_pressure = np.append(vetor_next_pressure, self.well_class.initial_pressure)

                pressure_df[i_col] = vetor_next_pressure

                last_col = i_col

        self.plot_results(data=pressure_df)
