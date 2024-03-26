"""
Simuladores para fluxo linear e radial - Métodos numéricos
MDF - Expansão em série de Tylor
"""
import numpy as np
from pandas import DataFrame as Df
import os
import matplotlib.pyplot as plt


class SimuladorPressPressLinear:
    def __init__(self, initial_pressure, well_pressure, permeability, viscosity, compressibility,
                 porosity, res_length):
        self.initial_pressure = initial_pressure
        self.well_pressure = well_pressure
        self.permeability = permeability
        self.viscosity = viscosity
        self.compressibility = compressibility
        self.porosity = porosity
        self.res_length = res_length
        self.eta = self.calc_eta()

    def calc_eta(self) -> float:
        eta_num = self.permeability / (self.porosity * self.viscosity * self.compressibility)
        return eta_num

    def calc_deltax(self, n_cells: int, t_len: np.ndarray) -> tuple:
        deltax = self.res_length/n_cells
        deltat = t_len[1] - t_len[0]
        rx_num = deltat/(deltax ** 2)
        x = np.linspace(0+(deltax/2), self.res_length-(deltax/2), n_cells)
        return deltax, deltat, rx_num, x

    def calc_estab(self, n_cells: int, t: np.ndarray):
        self.deltax = self.res_length / n_cells
        self.deltat = t[1] - t[0]
        self.rx = self.deltat / (self.deltax ** 2)
        self.x = np.linspace(0 + (self.deltax / 2), self.res_length - (self.deltax / 2), n_cells)
        value = self.rx * self.eta
        return value

    @staticmethod
    def plot_results(data: Df):
        if not os.path.isdir(f'results\\Simulador_Pressao-Pressao'):
            os.makedirs(f'results\\Simulador_Pressao-Pressao')

        columns_to_plot = ['0', '100', '200', '300', '400', '500']
        for column in data.columns:
            # time_num = float(column.split(' ')[1])
            # if column in columns_to_plot:
            plt.plot(data.index, data[column], label=f't = {int(column)}')
        # plt.plot(data.index, data['time 50.0'])
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

    def start_simulation(self, cell_number: int, t: np.ndarray):
        # delta_x, delta_t, rx, x = self.calc_deltax(cell_number, t)

        mesh = {f'{int(time)}': [] for time in t}
        chave_anterior = None
        for key in mesh.keys():
            pressure_in_x = []
            if key == '0':
                for position in self.x:
                    pressure_in_x.append(self.initial_pressure)
                mesh[key] = pressure_in_x
                chave_anterior = key
            else:
                n = len(self.x) - 1
                for i in range(n+1):

                    if i == 0:
                        p1_t = mesh[chave_anterior][i]
                        p2_t = mesh[chave_anterior][i+1]
                        pressure_value = ((8/3 * self.eta * self.rx * self.well_pressure) +
                                          ((1 - (4 * self.eta * self.rx)) * p1_t) + (4/3 * self.eta * self.rx * p2_t))
                        pressure_in_x.append(pressure_value)
                    elif i == n:
                        pn_t = mesh[chave_anterior][i]
                        pn_1_t = mesh[chave_anterior][i-1]
                        pressure_value = ((4/3 * self.eta * self.rx * pn_1_t) +
                                          ((1 - (4 * self.eta * self.rx)) * pn_t) +
                                          (8/3 * self.eta * self.rx * self.initial_pressure))
                        pressure_in_x.append(pressure_value)
                    else:
                        pi_1_t = mesh[chave_anterior][i-1]
                        pi_t = mesh[chave_anterior][i]
                        pi_2_t = mesh[chave_anterior][i+1]
                        pressure_value = ((self.eta * self.rx * pi_1_t) + ((1 - (2 * self.eta * self.rx)) * pi_t) +
                                          (self.eta * self.rx * pi_2_t))
                        pressure_in_x.append(pressure_value)

                mesh[key] = pressure_in_x
                chave_anterior = key

        self.x = np.insert(self.x, 0, 0)
        self.x = np.append(self.x, self.res_length)
        for key in mesh:
            if key == '0':
                mesh[key].insert(0, self.initial_pressure)
                mesh[key].append(self.initial_pressure)
            else:
                mesh[key].insert(0, self.well_pressure)
                mesh[key].append(self.initial_pressure)

        mesh = Df(mesh).set_index(self.x)
        self.plot_results(data=mesh)


class SimuladorFlowPressLinear:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility,
                 reserarea, reserthick, wellflow):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = reserlength
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.eta = self.calc_eta()
        self.area = reserarea
        self.thickness = reserthick
        self.flow = wellflow

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta

    def calc_deltax(self, n_cells: int, t_len: np.ndarray) -> tuple:
        deltax = self.res_length/n_cells
        deltat = t_len[1] - t_len[0]
        rx_num = deltat/(deltax ** 2)
        x = np.linspace(0+(deltax/2), self.res_length-(deltax/2), n_cells)
        return deltax, deltat, rx_num, x

    def calc_estab(self, n_cells: int, t: np.ndarray):
        self.deltax = self.res_length / n_cells
        self.deltat = t[1] - t[0]
        self.rx = self.deltat / (self.deltax ** 2)
        self.x = np.linspace(0 + (self.deltax / 2), self.res_length - (self.deltax / 2), n_cells)
        value = self.rx * self.eta
        return value

    @staticmethod
    def plot_results(data: Df):
        if not os.path.isdir(f'results\\Simulador_Fluxo-Pressao'):
            os.makedirs(f'results\\Simulador_Fluxo-Pressao')

        for column in data.columns:
            # time_num = float(column.split(' ')[1])
            plt.plot(data.index, data[column], label=f't = {int(column)}')
        # plt.plot(data.index, data['time 50.0'])
        plt.ticklabel_format(axis='y', style='plain')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.title('Pressão-Pressão [Numérico]')
        # plt.legend(framealpha=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico.png')
        plt.close()

        data.to_excel(f'results\\Simulador_Fluxo-Pressao\\fluxo-pressao_numerico.xlsx')

    def start_simulation(self, cell_number: int, t: np.ndarray):
        # delta_x, delta_t, rx, x = self.calc_deltax(cell_number, t)

        mesh = {f'{int(time)}': [] for time in t}
        chave_anterior = None
        for key in mesh.keys():
            pressure_in_x = []
            if key == '0':
                for position in self.x:
                    pressure_in_x.append(self.initial_pressure)
                mesh[key] = pressure_in_x
                chave_anterior = key
            else:
                n = len(self.x) - 1
                for i in range(n+1):

                    if i == 0:
                        p1_t = mesh[chave_anterior][i]
                        p2_t = mesh[chave_anterior][i+1]
                        pressure_value = ((self.eta * self.rx * p2_t) + ((1 - (self.eta * self.rx)) * p1_t) +
                                          (self.eta * self.rx * self.flow * (self.viscosity * self.deltax /
                                                                             (self.permeability * self.area))))
                        pressure_in_x.append(pressure_value)
                    elif i == n:
                        pn_t = mesh[chave_anterior][i]
                        pn_1_t = mesh[chave_anterior][i-1]
                        pressure_value = ((4/3 * self.eta * self.rx * pn_1_t) +
                                          ((1 - (4 * self.eta * self.rx)) * pn_t) +
                                          (8/3 * self.eta * self.rx * self.initial_pressure))
                        pressure_in_x.append(pressure_value)
                    else:
                        pi_1_t = mesh[chave_anterior][i-1]
                        pi_t = mesh[chave_anterior][i]
                        pi_2_t = mesh[chave_anterior][i+1]
                        pressure_value = ((self.eta * self.rx * pi_1_t) + ((1 - (2 * self.eta * self.rx)) * pi_t) +
                                          (self.eta * self.rx * pi_2_t))
                        pressure_in_x.append(pressure_value)

                mesh[key] = pressure_in_x
                chave_anterior = key

        self.x = np.insert(self.x, 0, 0)
        self.x = np.append(self.x, self.res_length)
        for key in mesh:
            if key == '0':
                mesh[key].insert(0, self.initial_pressure)
                mesh[key].append(self.initial_pressure)
            else:
                mesh[key].insert(0, (self.flow * self.viscosity)/(self.permeability * self.area))
                mesh[key].append(self.initial_pressure)

        mesh = Df(mesh).set_index(self.x)
        self.plot_results(data=mesh)
