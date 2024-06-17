"""
------------------------------------------------------------------------------------------------------------------------
Simuladores para fluxo linear - Métodos numéricos.
------------------------------------------------------------------------------------------------------------------------
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e pressão na fronteira.
Condições de Contorno de Neumann - Fluxo no poço e na fronteira.
------------------------------------------------------------------------------------------------------------------------
Método BTCS -- Backward in Time Centered Space [Implicit Method]
------------------------------------------------------------------------------------------------------------------------
"""
import sys

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

    @abstractmethod
    def set_matrixparameters(self):
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
        self.set_matrixparameters()


class OneDimensionalImplicitMethod(Implicit):
    def __init__(self):
        super().__init__()
        self.parameters = None

    def plot_results(self, data: Df):
        if not os.path.isdir(f'{self.well_class.rootpath}'):
            os.makedirs(f'{self.well_class.rootpath}')

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
        plt.title(f'{self.well_class.condiction} [Implicito]')
        plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        # plt.savefig(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.png')
        plt.savefig(f'{self.well_class.rootpath}\\{self.well_class.condiction}_{self.name}.png')
        plt.close()

        # data.to_excel(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.xlsx')
        data.to_excel(f'{self.well_class.rootpath}\\{self.well_class.condiction}_{self.name}.xlsx')
        # self.well_class.dataframe_to_implicit = data
        self.dataframe = data

    def set_matrixparameters(self):
        if self.well_class.condiction == 'PressurePressure':
            self.parameters = {
                'a': 1 + (4 * self.well_class.rx_implicit * self.well_class.eta),
                'b': - (4 / 3) * self.well_class.rx_implicit * self.well_class.eta,
                'c': - self.well_class.rx_implicit * self.well_class.eta,
                'd': 1 + (2 * self.well_class.rx_implicit * self.well_class.eta),
                'f1': (8 / 3) * self.well_class.rx_implicit * self.well_class.eta * self.well_class.well_pressure,
                'fn': (8 / 3) * self.well_class.rx_implicit * self.well_class.eta * self.well_class.initial_pressure
            }

        elif self.well_class.condiction == 'FlowPressure':
            self.parameters = {
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

        else:
            print('Error! Other condition besides PressurePressure or FlowPressure was not implemented yet.')
            sys.exit()

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_implicit)

        if self.well_class.condiction == 'PressurePressure':
            pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_pressureboundaries(
                n_cells=self.well_class.n_cells_implicit, param_values=self.parameters)
        else:
            pressure_matrix_, const_matrix = Functions.create_pressurecoeficients_flowboundaries(
                n_cells=self.well_class.n_cells_implicit, param_values=self.parameters)

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

                if self.well_class.condiction == 'PressurePressure':
                    value_to_add = self.well_class.well_pressure
                else:
                    value_to_add = vetor_next_pressure[0] - (((self.well_class.well_flow * self.well_class.viscosity) /
                                                              (self.well_class.permeability *
                                                               self.well_class.res_area)) *
                                                             (self.well_class.dx_implicit / 2))

                vetor_next_pressure = np.insert(vetor_next_pressure, 0, value_to_add)
                vetor_next_pressure = np.append(vetor_next_pressure, self.well_class.initial_pressure)

                pressure_df[i_col] = vetor_next_pressure

                last_col = i_col

        end_time = time.time()
        self.well_class.time_TO_implicit = end_time - start_time

        self.plot_results(data=pressure_df)


class TwoDimensionalImplicitMethod(Implicit):
    def __init__(self):
        super().__init__()
        self.beta = None
        self.ry = None
        self.rx = None
        self.parameters = None
        self.permeability_map = None

    def set_matrixparameters(self):
        self.beta = 1/(self.well_class.compressibility * self.well_class.viscosity * self.well_class.porosity)
        self.rx = self.well_class.rx_implicit
        self.ry = self.well_class.ry_implicit

        with open(self.well_class.permeability, 'r') as file:
            if file.readline()[:-1] != '# Permeability':
                print('Error: The file is not a file of permeability map!')
                sys.exit()
            else:
                pass

        index_for_dataframe = np.linspace(1, self.well_class.n_cells_implicit, self.well_class.n_cells_implicit)
        index_for_dataframe = [int(i) for i in index_for_dataframe]

        perm = pd.read_csv(self.well_class.permeability, delim_whitespace=True, header=None, skiprows=1)
        x = np.shape(perm)
        perm = Df(perm)

        self.permeability_map = Df(perm, columns=index_for_dataframe, index=index_for_dataframe)

    def start_simulate(self):
        pass

