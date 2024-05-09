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
import time
from abc import ABC, abstractmethod


class Implicit(ABC):
    def __init__(self):
        self.dataframe = None
        self.time = None
        self.well_class = None
        self.name = None

    def set_parameters(self, t: np.ndarray, well_class: object, name_file: str):
        self.time = t
        self.well_class = well_class
        self.name = name_file

    @abstractmethod
    def start_simulate(self):
        pass


class PressureBoundaries(Implicit):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results/OneDimensionalFlow/PressurePressure_Simulator'):
            os.makedirs(f'results/OneDimensionalFlow/PressurePressure_Simulator')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.implicit_mesh[key], ndigits=3)
                               for key in self.well_class.implicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title('Pressão-Pressão [Implicita]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.png')
        plt.close()

        data.to_excel(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.xlsx')
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_implicit)

        param = {
            'a': 1 + (4 * self.well_class.rx_implicit * self.well_class.eta),
            'b': - (4 / 3) * self.well_class.rx_implicit * self.well_class.eta,
            'c': - self.well_class.rx_implicit * self.well_class.eta,
            'd': 1 + (2 * self.well_class.rx_implicit * self.well_class.eta),
            'f1': (8 / 3) * self.well_class.rx_implicit * self.well_class.eta * self.well_class.well_pressure,
            'fn': (8 / 3) * self.well_class.rx_implicit * self.well_class.eta * self.well_class.initial_pressure
        }

        pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_pressureboundaries(
            n_cells=self.well_class.n_cells_implicit, param_values=param)

        start_time = time.time()
        last_col = None
        for i_col in col_idx:

            if i_col == 0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_col = i_col
            else:
                vetor_last_pressure = [pressure_df.loc[i, last_col] for i in row_idx if i != 0
                                       and i != self.well_class.n_cells_implicit + 1]
                b = vetor_last_pressure + const_matrix
                vetor_next_pressure = solve(pressure_matrix_, b)

                vetor_next_pressure = np.insert(vetor_next_pressure, 0, self.well_class.well_pressure)
                vetor_next_pressure = np.append(vetor_next_pressure, self.well_class.initial_pressure)

                pressure_df[i_col] = vetor_next_pressure

                last_col = i_col

        end_time = time.time()
        self.well_class.time_TO_implicit = end_time - start_time

        self.plot_results(data=pressure_df)


class WellFlowAndPressureBoundaries(Implicit):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results/OneDimensionalFlow/FlowPressure_Simulator'):
            os.makedirs(f'results/OneDimensionalFlow/FlowPressure_Simulator')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.implicit_mesh[key], ndigits=3)
                               for key in self.well_class.implicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title('Flow-Pressão [Implicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.png')
        plt.close()

        data.to_excel(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.xlsx')
        # self.well_class.dataframe_to_implicit = data
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_implicit)

        param = {
            'a': 1 + (self.well_class.rx_implicit * self.well_class.eta),
            'b': - self.well_class.rx_implicit * self.well_class.eta,
            'c': 1 + (2 * self.well_class.rx_implicit * self.well_class.eta),
            'd': - (4 / 3) * self.well_class.rx_implicit * self.well_class.eta,
            'e': 1 + (4 * self.well_class.rx_implicit * self.well_class.eta),
            'f1': - (self.well_class.rx_implicit * self.well_class.eta * self.well_class.well_flow *
                     self.well_class.viscosity *
                     self.well_class.dx_implicit) / (self.well_class.permeability * self.well_class.res_area),
            'fn': (8 / 3) * self.well_class.rx_implicit * self.well_class.eta * self.well_class.initial_pressure
        }

        pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_flowboundaries(
            n_cells=self.well_class.n_cells_implicit, param_values=param)

        start_time = time.time()
        last_col = None
        for i_col in col_idx:

            if i_col == 0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_col = i_col
            else:
                vetor_last_pressure = [pressure_df.loc[i, last_col] for i in row_idx if i != 0
                                       and i != self.well_class.n_cells_implicit + 1]
                b = vetor_last_pressure + const_matrix
                vetor_next_pressure = solve(pressure_matrix_, b)

                # p1_t = pressure_df[last_col][1]  # pressão no ponto 1, tempo anterior.
                # p1_t = pressure_df.loc[1, last_col]  # pressão no ponto 1, tempo anterior.
                value_to_add = vetor_next_pressure[0] - (((self.well_class.well_flow * self.well_class.viscosity) /
                                                          (self.well_class.permeability * self.well_class.res_area)) *
                                                         (self.well_class.dx_implicit / 2))

                vetor_next_pressure = np.insert(vetor_next_pressure, 0, value_to_add)
                vetor_next_pressure = np.append(vetor_next_pressure, self.well_class.initial_pressure)

                pressure_df[i_col] = vetor_next_pressure

                last_col = i_col

        end_time = time.time()
        self.well_class.time_TO_implicit = end_time - start_time

        self.plot_results(data=pressure_df)
