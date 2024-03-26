"""
Simuladores para fluxo linear e radial - Métodos analitico e numérico
Condições de Contorno de Dirichlet - Pressão no poço e Pressão na fronteira.
"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as Df
import os
from scipy.special import erfc, expn
import sys


class NumericalAnalysis:
    def __init__(self, grid: dict, t: np.ndarray, eta: float, well_pressure, initial_pressure, deltax: float or int):
        self.mesh = grid
        self.time = t
        self.eta = eta
        self.well_pressure = well_pressure
        self.initial_pressure = initial_pressure
        self.delta_x = deltax
        self.rx = self.calc_rx()

    def calc_rx(self):
        """
        The function calculate the parameter Rx (delta in time / delta in length ²) and verify the convergence
        criterium of the numerical solution.
        :return: The rx value and a message error if the convergence criterium is not
        """
        delta_t = self.time[1] - self.time[0]
        r_x = delta_t / (self.delta_x ** 2)

        if r_x * self.eta >= 0.25:
            print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "rx * eta > 0.25".')
            print(f'rx = {r_x} // eta = {self.eta}  // rx * eta = {r_x * self.eta}')
            sys.exit()

        return r_x


class AnaliticalAnalysis:
    def __init__(self, grid: dict, t: np.ndarray, eta: float, res_length, delta_press: float, well_pressure,
                 initial_pressure):
        self.mesh = grid
        self.time = t
        self.eta = eta
        self.res_length = res_length
        self.delta_pressure = delta_press
        self.well_pressure = well_pressure
        self.initial_pressure = initial_pressure
        self.start_simulate()

    def calc_sum(self, t_value, point_value):
        """
        Function that will calculate the sum that is required to plot the pressure field in the analitical solution.

        :param t_value: Value of time. Must be an integer of float.
        :param point_value: Position in the mesh. Must be an integer or float.
        :return: The sum needed to calculate the analitical solution.
        """

        # Value of the sum for n = 1:
        sum_value = ((np.exp(-1 * (((1 * np.pi) / self.res_length) ** 2) * (self.eta * t_value)) / 1) *
                     np.sin((1 * np.pi * point_value) / self.res_length))

        # Now the iterative process begins:
        n = 2
        erro = 100
        eppara = 0.000001
        while erro >= eppara:
            sum_old = sum_value
            sum_value += ((np.exp(-1 * (((n * np.pi) / self.res_length) ** 2) * (self.eta * t_value)) / n) *
                          np.sin((n * np.pi * point_value) / self.res_length))
            erro = np.fabs(sum_value - sum_old) / sum_value
            n += 1

        return sum_value

    @staticmethod
    def plot_result(data: Df):
        if not os.path.isdir(f'results\\Simulador_Pressao-Pressao'):
            os.makedirs(f'results\\Simulador_Pressao-Pressao')

        for column in data.columns:
            # time_num = float(column.split(' ')[1])
            plt.plot(data.index, data[column], label=f't = {int(column)}')
        # plt.plot(data.index, data['time 50.0'])
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
        pressure = {}
        for t in self.time:
            vector_for_time = []

            if t == 0:
                for x in self.mesh.keys():
                    vector_for_time.append(self.initial_pressure * (self.mesh[x] + 1) / (self.mesh[x] + 1))
            else:
                for key in self.mesh.keys():
                    if self.mesh[key] == 0:
                        vector_for_time.append(self.well_pressure)
                    elif self.mesh[key] == self.res_length:
                        vector_for_time.append(self.initial_pressure)
                    else:
                        suma = self.calc_sum(t, self.mesh[key])
                        vector_for_time.append((self.delta_pressure * (self.mesh[key] / self.res_length +
                                                                       ((2 / np.pi) * suma))) + self.well_pressure)

            pressure[int(t)] = vector_for_time

        # Setting the mesh points as the dataframe index
        index_for_dataframe = [round(self.mesh[key], ndigits=3) for key in self.mesh.keys()]
        pressure = {'x': index_for_dataframe, **pressure}
        # Creating dataframe from the variable 'pressure'
        pressure = Df(pressure).set_index('x')
        self.plot_result(data=pressure)


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
        self.mesh, self.deltax = None, None

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure

    def create_mesh(self, n_cells: int = 0, deltax: float or int = 0):
        """
        Function to generate the grid for simulating pressure field. You can pass the value of cells that you wish your
        grid to have or the distance between the points. If you set one of then, you must set the other value equal to
        zero.

        :param n_cells: Number of cells that you wish to divide your grid. Must be an integer value. Set n_cells = 0 if
        you pass deltax!
        :param deltax: The distance between the points in the grid. It can be an integer or a float value. Set deltax =
        0 if you pass n_cells!
        :return: The mesh dict of your problem with the internal points and the two contour points!
        """
        if type(n_cells) is not int:
            print(f'Error!!! The parameter "n_cells" must be set if an integer value. Type passed: {type(n_cells)}.')
            sys.exit()

        if n_cells != 0 and deltax != 0:
            print(f'Error!!! Both parameters were set non-zero. You must set at least you of then different from zero.'
                  f'Or you can set just one of then.')

        if n_cells == 0 and deltax == 0:
            print(f'Error! Both parameters were set if value zero. '
                  f'You must set at least one of then non-zero. Or you can set just one of then.')
            sys.exit()

        if deltax == 0:  # the creation of the grid depends on the number of cells
            self.deltax = self.res_length / n_cells
            initial_point = deltax / 2
            final_point = self.res_length - deltax / 2
            x_array = np.linspace(initial_point, final_point, n_cells)  # internal points of the grid
            x_array = np.insert(x_array, 0, '0')  # insert the initial contour point
            x_array = np.append(x_array, int(self.res_length))  # insert the final contour point
            self.mesh = {i: x_array[i] for i in range(len(x_array))}
        else:  # the creation of the grid depends on the value of deltax
            self.deltax = deltax
            n_cells = int(self.res_length / deltax)
            initial_point = deltax / 2
            final_point = self.res_length - deltax / 2
            x_array = np.linspace(initial_point, final_point, n_cells)  # internal points of the grid
            x_array = np.insert(x_array, 0, '0')  # insert the initial contour point
            x_array = np.append(x_array, int(self.res_length))  # insert the final contour point
            self.mesh = {i: x_array[i] for i in range(len(x_array))}
