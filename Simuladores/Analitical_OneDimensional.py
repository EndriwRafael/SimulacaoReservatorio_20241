"""
------------------------------------------------------------------------------------------------------------------------
Simuladores para fluxo linear e radial - Métodos analiticos.
------------------------------------------------------------------------------------------------------------------------
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
Condições de Contorno de Neumann e Dirichlet - Fluxo no poço e pressão na fronteira.
------------------------------------------------------------------------------------------------------------------------
"""
import Functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as Df
import os
from scipy.special import erfc
from abc import ABC, abstractmethod


class Analitical(ABC):
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


class PressureBoundaries(Analitical):
    def __init__(self):
        super().__init__()

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
        if not os.path.isdir(f'{self.well_class.rootpath}'):
            os.makedirs(f'{self.well_class.rootpath}')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.analitical_mesh[key], ndigits=3)
                               for key in self.well_class.analitical_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        if self.name == 'Analitical':
            time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

            for column in data.columns:
                if column in time_to_plot:
                    plt.plot(data.index, data[column], label=f't = {int(column)}h')

            plt.ticklabel_format(axis='y', style='plain')
            plt.xlabel('Comprimento (m)')
            plt.ylabel('Pressão (Pa)')
            plt.title(f'Pressão-Pressão [{self.name}]')
            plt.legend(framealpha=1)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'{self.well_class.rootpath}\\PressurePressure_{self.name}.png')
            # plt.savefig(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.png')
            plt.close()
        else:
            pass

        data.to_excel(f'{self.well_class.rootpath}\\PressurePressure_{self.name}.xlsx')
        # data.to_excel(f'results\\OneDimensionalFlow\\PressurePressure_Simulator\\PressurePressure_{self.name}.xlsx')
        # self.well_class.dataframe_to_analitical = data
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_analitical)

        for time in col_idx:
            if time == 0:
                for x in row_idx:
                    pressure_df.loc[x, time] = self.well_class.initial_pressure
            else:
                for x in row_idx:
                    if x == 0:
                        pressure_df.loc[x, time] = self.well_class.well_pressure
                    else:
                        suma = self.calc_sum(time, self.well_class.analitical_mesh[x])
                        value = ((self.well_class.deltaPressure * (self.well_class.analitical_mesh[x] /
                                                                   self.well_class.res_length +
                                                                   ((2 / np.pi) * suma))) +
                                 self.well_class.well_pressure)
                        pressure_df.loc[x, time] = value

        self.plot_result(data=pressure_df)


class WellFlowAndPressureBoundaries(Analitical):
    def __init__(self):
        super().__init__()

    def plot_results(self, data: Df):
        if not os.path.isdir(f'{self.well_class.rootpath}'):
            os.makedirs(f'{self.well_class.rootpath}')

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.well_class.analitical_mesh[key], ndigits=3)
                               for key in self.well_class.analitical_mesh.keys()]
        data = data.set_index(pd.Index(index_for_dataframe, name='x'))

        if self.name == 'Analitical':
            time_to_plot = np.linspace(self.time[0], self.time[-1], 11)

            for column in data.columns:
                if column in time_to_plot:
                    plt.plot(data.index, data[column], label=f't = {int(column)}h')

            plt.ticklabel_format(axis='y', style='plain')
            plt.xlabel('Comprimento (m)')
            plt.ylabel('Pressão (Pa)')
            plt.title(f'Fluxo-Pressão [{self.name}]')
            plt.legend(framealpha=1)
            plt.grid()
            plt.tight_layout()
            # plt.savefig(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.png')
            plt.savefig(f'{self.well_class.rootpath}\\FlowPressure_{self.name}.png')
            plt.close()
        else:
            pass

        # data.to_excel(f'results\\OneDimensionalFlow\\FlowPressure_Simulator\\FlowPressure_{self.name}.xlsx')
        data.to_excel(f'{self.well_class.rootpath}\\FlowPressure_{self.name}.xlsx')
        # self.well_class.dataframe_to_analitical = data
        self.dataframe = data

    def start_simulate(self):
        pressure_df, col_idx, row_idx = Functions.create_dataframe(time=self.time,
                                                                   n_cells=self.well_class.n_cells_analitical)

        for time in col_idx:
            if time == 0:
                for x in row_idx:
                    pressure_df.loc[x, time] = self.well_class.initial_pressure

            else:
                for x in row_idx:
                    a = ((self.well_class.well_flow * self.well_class.viscosity) /
                         (self.well_class.permeability * self.well_class.res_area))
                    b = np.sqrt((4 * self.well_class.eta * time) / np.pi)
                    c = np.exp(self.well_class.analitical_mesh[x] ** 2 / (-4 * self.well_class.eta * time))
                    d = self.well_class.analitical_mesh[x]
                    e = erfc(self.well_class.analitical_mesh[x] / np.sqrt(4 * self.well_class.eta * time))
                    value = (self.well_class.initial_pressure - (a * ((b * c) - (d * e))))
                    pressure_df.loc[x, time] = value

        self.plot_results(data=pressure_df)
