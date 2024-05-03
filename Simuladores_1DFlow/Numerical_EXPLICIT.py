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


class PressureBoundaries:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.well_class = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Pressao-Pressao'):
            os.makedirs(f'../results/Simulador_Pressao-Pressao')

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
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Numérico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico_Explicit.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Pressao-Pressao\\pressao-pressao_numerico_Explicit.xlsx')
        self.well_class.dataframe_to_explicit = data

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


class WellFlowAndPressureBoundaries:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.wellclass = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Fluxo-Pressao'):
            os.makedirs(f'../results/Simulador_Fluxo-Pressao')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.wellclass.explicit_mesh[key], ndigits=3)
                               for key in self.wellclass.explicit_mesh.keys()]
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
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico_Explicit.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico_Explicit.xlsx')
        self.wellclass.dataframe_to_explicit = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.wellclass.n_cells_explicit)

        start_time = time.time()
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
                                                                    self.wellclass.res_area)) *
                                                                self.wellclass.deltax_explicit/2)

                    elif j_row == self.wellclass.n_cells_explicit + 1:
                        pressure_df.loc[j_row, i_col] = self.wellclass.initial_pressure

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = self.wellclass.eta * self.wellclass.rx_explicit * p2_t
                            b = (1 - (self.wellclass.eta * self.wellclass.rx_explicit)) * p1_t
                            c = self.wellclass.eta * self.wellclass.rx_explicit * self.wellclass.well_flow * \
                                self.wellclass.viscosity * self.wellclass.deltax_explicit / \
                                (self.wellclass.permeability * self.wellclass.res_area)
                            pressure_df.loc[j_row, i_col] = a + b - c

                        elif j_row == self.wellclass.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = (4 / 3) * self.wellclass.eta * self.wellclass.rx_explicit * p_n_1_t
                            b = (1 - (4 * self.wellclass.eta * self.wellclass.rx_explicit)) * p_n_t
                            c = (8 / 3) * self.wellclass.eta * self.wellclass.rx_explicit * \
                                self.wellclass.initial_pressure
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.wellclass.eta * self.wellclass.rx_explicit * pi_1
                            b = (1 - (2 * self.wellclass.eta * self.wellclass.rx_explicit)) * pi_t
                            c = self.wellclass.eta * self.wellclass.rx_explicit * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c
                last_column = i_col
        end_time = time.time()
        self.wellclass.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)


class FlowBoundaries:
    def __init__(self, t: np.ndarray, well_class: object):
        self.time = t
        self.wellclass = well_class
        self.start_simulate()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'../results/Simulador_Fluxo-Fluxo'):
            os.makedirs(f'../results/Simulador_Fluxo-Fluxo')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.wellclass.explicit_mesh[key], ndigits=3)
                               for key in self.wellclass.explicit_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

        for column in data.columns:
            if column in time_to_plot:
                plt.plot(data.index, data[column], label=f't = {int(column)}h')

        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Flow-Flow [Numérico]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Fluxo\\fluxo-fluxo_numerico_Explicit.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Fluxo\\fluxo-fluxo_numerico_Explicit.xlsx')
        self.wellclass.dataframe_to_explicit = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.wellclass.n_cells_explicit)

        start_time = time.time()
        last_column = None
        for i_col in col_idx:

            if i_col == 0.0:
                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.wellclass.initial_pressure
                last_column = i_col

            else:
                for j_row in row_idx:

                    if j_row == 0:  # Poço
                        p0_t = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto 1, tempo anterior.
                        pressure_df.loc[j_row, i_col] = p0_t - (((self.wellclass.well_flow * self.wellclass.viscosity)
                                                                 / (self.wellclass.permeability *
                                                                    self.wellclass.res_area)) *
                                                                (self.wellclass.deltax_explicit/2))

                    elif j_row == self.wellclass.n_cells_explicit + 1:  # Fronteira externa
                        a = (self.wellclass.injectflow * self.wellclass.viscosity) / \
                            (self.wellclass.permeability * self.wellclass.res_area)
                        b = self.wellclass.deltax_explicit / 2
                        p_n = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N, no tempo anterior.
                        pressure_df.loc[j_row, i_col] = p_n - (a * b)

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no
                            # tempo anterior
                            a = self.wellclass.eta * self.wellclass.rx_explicit * p2_t
                            b = (1 - (self.wellclass.eta * self.wellclass.rx_explicit)) * p1_t
                            c = self.wellclass.eta * self.wellclass.rx_explicit * self.wellclass.well_flow * \
                                self.wellclass.viscosity * self.wellclass.deltax_explicit / \
                                (self.wellclass.permeability * self.wellclass.res_area)
                            pressure_df.loc[j_row, i_col] = a + b - c

                        elif j_row == self.wellclass.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no
                            # tempo anterior
                            a = self.wellclass.eta * self.wellclass.rx_explicit * p_n_1_t
                            b = (1 - (self.wellclass.eta * self.wellclass.rx_explicit)) * p_n_t
                            c = - (self.wellclass.eta * self.wellclass.rx_explicit * self.wellclass.injectflow *
                                   self.wellclass.viscosity * self.wellclass.deltax_explicit) / \
                                (self.wellclass.permeability * self.wellclass.res_area)
                            pressure_df.loc[j_row, i_col] = a + b + c

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no
                            # tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no
                            # tempo anterior.
                            a = self.wellclass.eta * self.wellclass.rx_explicit * pi_1
                            b = (1 - (2 * self.wellclass.eta * self.wellclass.rx_explicit)) * pi_t
                            c = self.wellclass.eta * self.wellclass.rx_explicit * pi_2
                            pressure_df.loc[j_row, i_col] = a + b + c
                last_column = i_col
        end_time = time.time()
        self.wellclass.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)
