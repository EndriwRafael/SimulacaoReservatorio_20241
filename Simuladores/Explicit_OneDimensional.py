"""
------------------------------------------------------------------------------------------------------------------------
Simuladores para fluxo linear e radial - Métodos numéricos.
------------------------------------------------------------------------------------------------------------------------
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e pressão na fronteira.
Condições de Contorno de Neumann - Fluxo no poço e na fronteira.
------------------------------------------------------------------------------------------------------------------------
Método FTCS -- Forward in Time Centered Space
------------------------------------------------------------------------------------------------------------------------
"""
import Functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os
import time
from abc import ABC, abstractmethod


class Explicit(ABC):
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


class PressureBoundaries(Explicit):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results/OneDimensionalFlow/PressurePressure_Simulator'):
            os.makedirs(f'results/OneDimensionalFlow/PressurePressure_Simulator')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.explicit_mesh[key], ndigits=3)
                               for key in self.well_class.explicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title('Pressão-Pressão [Explicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.png')
        plt.close()

        data.to_excel(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.xlsx')
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_explicit)

        start_time = time.time()
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

                    elif j_row == self.well_class.n_cells_explicit + 1:
                        pressure_df.loc[j_row, i_col] = self.well_class.initial_pressure

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * \
                                self.well_class.well_pressure
                            b = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit)) * p1_t
                            c = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit * p2_t
                            pressure_df.loc[j_row, i_col] = a + b + c

                        elif j_row == self.well_class.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit * p_n_1_t
                            b = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit)) * p_n_t
                            c = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * \
                                self.well_class.initial_pressure
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.well_class.eta * self.well_class.rx_explicit * pi_1
                            b = (1 - (2 * self.well_class.eta * self.well_class.rx_explicit)) * pi_t
                            c = self.well_class.eta * self.well_class.rx_explicit * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c

                last_column = i_col
        end_time = time.time()
        self.well_class.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)


class WellFlowAndPressureBoundaries(Explicit):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results/OneDimensionalFlow/FlowPressure_Simulator'):
            os.makedirs(f'results/OneDimensionalFlow/FlowPressure_Simulator')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.explicit_mesh[key], ndigits=3)
                               for key in self.well_class.explicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title('Flow-Pressão [Explicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.png')
        plt.close()

        data.to_excel(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.xlsx')
        # self.well_class.dataframe_to_explicit = data
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_explicit)

        start_time = time.time()
        last_column = None
        for i_col in col_idx:

            if i_col == 0.0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_column = i_col

            else:
                for j_row in row_idx:

                    if j_row == 0:
                        p0_t = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto 1, tempo anterior.
                        pressure_df.loc[j_row, i_col] = p0_t - (((self.well_class.well_flow * self.well_class.viscosity)
                                                                 / (self.well_class.permeability *
                                                                    self.well_class.res_area)) *
                                                                self.well_class.dx_explicit/2)

                    elif j_row == self.well_class.n_cells_explicit + 1:
                        pressure_df.loc[j_row, i_col] = self.well_class.initial_pressure

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = self.well_class.eta * self.well_class.rx_explicit * p2_t
                            b = (1 - (self.well_class.eta * self.well_class.rx_explicit)) * p1_t
                            c = self.well_class.eta * self.well_class.rx_explicit * self.well_class.well_flow * \
                                self.well_class.viscosity * self.well_class.dx_explicit / \
                                (self.well_class.permeability * self.well_class.res_area)
                            pressure_df.loc[j_row, i_col] = a + b - c

                        elif j_row == self.well_class.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit * p_n_1_t
                            b = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit)) * p_n_t
                            c = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * \
                                self.well_class.initial_pressure
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.well_class.eta * self.well_class.rx_explicit * pi_1
                            b = (1 - (2 * self.well_class.eta * self.well_class.rx_explicit)) * pi_t
                            c = self.well_class.eta * self.well_class.rx_explicit * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c
                last_column = i_col
        end_time = time.time()
        self.well_class.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)


class FlowBoundaries(Explicit):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'results/OneDimensionalFlow/FlowFlow_Simulator'):
            os.makedirs(f'results/OneDimensionalFlow/FlowFlow_Simulator')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.explicit_mesh[key], ndigits=3)
                               for key in self.well_class.explicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title('Flow-Flow [Explicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\OneDimensionalFlow\\FlowFlow_Simulator\\FlowFlow_{self.name}.png')
        plt.close()

        data.to_excel(f'results\\OneDimensionalFlow\\FlowFlow_Simulator\\FlowFlow_{self.name}.xlsx')
        # self.wellclass.dataframe_to_explicit = data
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_explicit)

        start_time = time.time()
        last_column = None
        for i_col in col_idx:

            if i_col == 0.0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_column = i_col

            else:
                for j_row in row_idx:

                    if j_row == 0:  # Poço
                        p0_t = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto 1, tempo anterior.
                        pressure_df.loc[j_row, i_col] = p0_t - (((self.well_class.well_flow * self.well_class.viscosity)
                                                                 / (self.well_class.permeability *
                                                                    self.well_class.res_area)) *
                                                                (self.well_class.dx_explicit/2))

                    elif j_row == self.well_class.n_cells_explicit + 1:  # Fronteira externa
                        a = (self.well_class.injectflow * self.well_class.viscosity) / \
                            (self.well_class.permeability * self.well_class.res_area)
                        b = self.well_class.dx_explicit / 2
                        p_n = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N, no tempo anterior.
                        pressure_df.loc[j_row, i_col] = p_n - (a * b)

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = self.well_class.eta * self.well_class.rx_explicit * p2_t
                            b = (1 - (self.well_class.eta * self.well_class.rx_explicit)) * p1_t
                            c = self.well_class.eta * self.well_class.rx_explicit * self.well_class.well_flow * \
                                self.well_class.viscosity * self.well_class.dx_explicit / \
                                (self.well_class.permeability * self.well_class.res_area)
                            pressure_df.loc[j_row, i_col] = a + b - c

                        elif j_row == self.well_class.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = self.well_class.eta * self.well_class.rx_explicit * p_n_1_t
                            b = (1 - (self.well_class.eta * self.well_class.rx_explicit)) * p_n_t
                            c = - (self.well_class.eta * self.well_class.rx_explicit * self.well_class.injectflow *
                                   self.well_class.viscosity * self.well_class.dx_explicit) / \
                                (self.well_class.permeability * self.well_class.res_area)
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.well_class.eta * self.well_class.rx_explicit * pi_1
                            b = (1 - (2 * self.well_class.eta * self.well_class.rx_explicit)) * pi_t
                            c = self.well_class.eta * self.well_class.rx_explicit * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c
                last_column = i_col
        end_time = time.time()
        self.well_class.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)
