"""
------------------------------------------------------------------------------------------------------------------------
Simuladores para fluxo linear - Métodos numéricos.
------------------------------------------------------------------------------------------------------------------------
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e pressão na fronteira.
Condições de Contorno de Neumann - Fluxo no poço e na fronteira.
------------------------------------------------------------------------------------------------------------------------
Método FTCS -- Forward in Time Centered Space [Explicit Method]
------------------------------------------------------------------------------------------------------------------------
"""
import sys
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

    @abstractmethod
    def set_objectivefunctions(self):
        """
        Function to create the approximate objective functions that will solve the EDH equation, for all type of flux
        [1D, 2D or 3D].
        :return:
        """
        pass

    @abstractmethod
    def start_simulate(self):
        """
        Function that will start the simulation for all type of flux [1D, 2D or 3D]
        :return: The result of the field pressure in all the mesh points.
        """
        pass

    def set_parameters(self, t: np.ndarray, well_class: object, name_file: str):
        self.time = t
        self.well_class = well_class
        self.name = name_file
        self.set_objectivefunctions()


class OneDimensionalExplicitMethod(Explicit):
    def __init__(self):
        super().__init__()
        self.external_boundary_func = None
        self.well_boundary_func = None
        self.intern_cells_func = None
        self.last_cell_func = None
        self.first_cell_func = None

    def plot_results(self, data: Df):
        if not os.path.isdir(f'{self.well_class.rootpath}'):
            os.makedirs(f'{self.well_class.rootpath}')

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
        plt.title(f'{self.well_class.condiction} [Explicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{self.well_class.rootpath}\\{self.well_class.condiction}_{self.name}.png')
        plt.close()

        data.to_excel(f'{self.well_class.rootpath}\\{self.well_class.condiction}_{self.name}.xlsx')

        self.dataframe = data

    def set_objectivefunctions(self):

        if self.well_class.condiction == 'PressurePressure':
            # First cell equations -------------------------------------------------------------------------------------
            a = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * self.well_class.well_pressure
            b = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit))
            c = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit
            self.first_cell_func = lambda x1, x2: a + b * x1 + c * x2

            # Last cell equations --------------------------------------------------------------------------------------
            d = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit
            e = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit))
            f = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * self.well_class.initial_pressure
            self.last_cell_func = lambda x1, x2: d * x1 + e * x2 + f

            # Well boundary equation -----------------------------------------------------------------------------------
            self.well_boundary_func = lambda x: x / x * self.well_class.well_pressure

            # External boundary equation -------------------------------------------------------------------------------
            self.external_boundary_func = self.well_class.initial_pressure

        elif self.well_class.condiction == 'FlowPressure':
            # First cell equations -------------------------------------------------------------------------------------
            a = self.well_class.eta * self.well_class.rx_explicit
            b = (1 - (self.well_class.eta * self.well_class.rx_explicit))
            c = self.well_class.eta * self.well_class.rx_explicit * self.well_class.well_flow * \
                self.well_class.viscosity * self.well_class.dx_explicit / \
                (self.well_class.permeability * self.well_class.res_area)
            self.first_cell_func = lambda x1, x2: a * x2 + b * x1 - c

            # Last cell equations --------------------------------------------------------------------------------------
            d = (4 / 3) * self.well_class.eta * self.well_class.rx_explicit
            e = (1 - (4 * self.well_class.eta * self.well_class.rx_explicit))
            f = (8 / 3) * self.well_class.eta * self.well_class.rx_explicit * self.well_class.initial_pressure
            self.last_cell_func = lambda x1, x2: d * x1 + e * x2 + f

            # Well boundary equation -----------------------------------------------------------------------------------
            g = (((self.well_class.well_flow * self.well_class.viscosity)
                  / (self.well_class.permeability *
                     self.well_class.res_area)) *
                 self.well_class.dx_explicit / 2)
            self.well_boundary_func = lambda x: x - g

            # External boundary equation -------------------------------------------------------------------------------
            self.external_boundary_func = self.well_class.initial_pressure

        elif self.well_class.condiction == 'FlowFlow':
            print('Error! The condition of Flow in the two boundaries was not implemented yet.')
            sys.exit()

        else:
            print('Error! You must set only a valid condition boundaries. To see what are the valid conditions, '
                  'please check the Readme.md')
            sys.exit()

        # Intern cells equations ---------------------------------------------------------------------------------------
        j = self.well_class.eta * self.well_class.rx_explicit
        h = (1 - (2 * self.well_class.eta * self.well_class.rx_explicit))
        i = self.well_class.eta * self.well_class.rx_explicit
        self.intern_cells_func = lambda x1, x2, x3: j * x1 + h * x2 + i * x3

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_explicit)

        start_time = time.time()
        last_column = None
        for i_col in col_idx:

            # Initial condition ----------------------------------------------------------------------------------------
            if i_col == 0.0:

                for i_row in row_idx:
                    pressure_df.loc[i_row, i_col] = self.well_class.initial_pressure
                last_column = i_col

            # Contour condition ----------------------------------------------------------------------------------------
            else:

                for j_row in row_idx:

                    if j_row == 0:
                        if self.well_class.condiction == 'PressurePressure':
                            pressure_df.loc[j_row, i_col] = self.well_boundary_func(1)
                        else:  # FlowPressure condition
                            p0_t = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto 1, tempo anterior.
                            pressure_df.loc[j_row, i_col] = self.well_boundary_func(p0_t)

                    elif j_row == self.well_class.n_cells_explicit + 1:
                        pressure_df.loc[j_row, i_col] = self.external_boundary_func

                    else:
                        if j_row == 1:  # i = 1. Ponto central da primeira célula.
                            p1_t = pressure_df.loc[j_row, last_column]  # pressão no ponto 1, tempo anterior.
                            p2_t = pressure_df.loc[2, last_column]  # pressão no ponto 2, no tempo anterior

                            pressure_df.loc[j_row, i_col] = self.first_cell_func(p1_t, p2_t)

                        elif j_row == self.well_class.n_cells_explicit:  # i = N. Ponto central da última célula.
                            p_n_t = pressure_df.loc[j_row, last_column]  # pressão no ponto N, no tempo anterior.
                            p_n_1_t = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto N-1, no tempo anterior

                            pressure_df.loc[j_row, i_col] = self.last_cell_func(p_n_1_t, p_n_t)

                        else:
                            pi_t = pressure_df.loc[j_row, last_column]  # pressão no ponto i, no tempo anterior.
                            pi_1 = pressure_df.loc[j_row - 1, last_column]  # pressão no ponto i-1, no tempo anterior.
                            pi_2 = pressure_df.loc[j_row + 1, last_column]  # pressão no ponto i+1, no tempo anterior.

                            pressure_df.loc[j_row, i_col] = self.intern_cells_func(pi_1, pi_t, pi_2)

                last_column = i_col
        end_time = time.time()
        self.well_class.time_TO_explicit = end_time - start_time

        self.plot_results(data=pressure_df)


class TwoDimensionalExplicitMethod(Explicit):
    def __init__(self):
        super().__init__()
        pass

    def set_objectivefunctions(self):
        pass

    def start_simulate(self):
        pass
