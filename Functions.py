import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from pandas import DataFrame as Df
import numpy as np
import random as rm
# from matplotlib.lines import Line2D
import sys
from Objects import Object_Simulation, Object_Case
from Simuladores import ExplicitMethod, ImplicitMethod, AnaliticalMethod


def set_color(list_color: list):
    while True:
        a = rm.randint(0, 9)
        if f'C{a}' not in list_color:
            break
    return f'C{a}'


def plot_graphs(dataclass: object, columns, root: str):
    # ------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    # legend_elements = []
    color_list = []
    color_label = []

    for col in columns:
        if len(color_list) == 10:
            color_list = []
        color_list.append(set_color(list_color=color_list))

        ax.plot(dataclass.analitical.index, dataclass.analitical[col], color=color_list[-1])
        ax.scatter(dataclass.explicit.index, dataclass.explicit[col], marker='o', s=5, color=color_list[-1],
                   label='_nolegend_')
        ax.scatter(dataclass.implicit.index, dataclass.implicit[col], marker='^', s=.5, color=color_list[-1],
                   label='_nolegend_')

        # legend_elements.append(Line2D([0], [0], linestyle='-', color=color_list[-1], label=f'{col}'))
        color_label.append(color_list[-1])

    patches = [Patch(color=c, label=l) for c, l in zip(color_list, columns)]

    plt.ticklabel_format(axis='y', style='plain')
    plt.xlabel('Comprimento (m)')
    plt.ylabel('Pressão (psia)')
    plt.title("Comparação das Curvas (Analítico, Explicito e Implicito")
    ax.legend(framealpha=1, handles=patches)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{root}\\CompareAnalysis.png')
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------


def plot_animation_results(data: object, root: str):
    df_ana = data.analitical
    df_exp = data.explicit
    df_imp = data.implicit
    columns = [coll for coll in df_imp.columns if coll != 0.0 and coll in df_exp.columns]

    # Função para atualizar o gráfico a cada quadro da animação
    def update(frame):
        pressure_ana = [j for _, j in enumerate(df_ana.loc[:, columns[frame]])]
        pressure_exp = [j for _, j in enumerate(df_exp.loc[:, columns[frame]])]
        pressure_imp = [j for _, j in enumerate(df_imp.loc[:, columns[frame]])]

        plt.cla()  # Limpa o eixo atual para atualizar o gráfico
        ax.plot(df_ana.index, pressure_ana, label=f'Analítica t={columns[frame]}', color='red')
        ax.scatter(df_exp.index, pressure_exp, marker='o', s=8, label=f'Explicita t={columns[frame]}', color='green')
        ax.scatter(df_imp.index, pressure_imp, marker='^', s=.5, label=f'Implitia t={columns[frame]}', color='blue')
        plt.ticklabel_format(axis='y', style='plain')
        plt.ylim([df_ana.min().min(), df_ana.max().max()])
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (Pa)')
        plt.title("Comparação das Curvas (Analítico, Explicito e Implicito)")
        ax.legend(framealpha=1)
        ax.grid()
        plt.tight_layout()

    # Configuração do gráfico
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(columns) - 1, interval=1000)  # Intervalo de 1000ms entre frames

    # Salvar a animação como GIF
    ani.save(f'{root}\\animacao_dataframe.gif', writer='pillow', fps=3)  # 1 frame por segundo
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------


def plot_pressuremap_animation(data: object, root: str):
    df_ana = data.analitical
    df_exp = data.explicit
    df_imp = data.implicit
    columns = [coll for coll in df_imp.columns if coll != 0.0 and coll in df_exp.columns]

    def update(frame):
        pressure_ana = [[j for _, j in enumerate(df_ana.loc[:, columns[frame]])],
                        [j for _, j in enumerate(df_ana.loc[:, columns[frame]])]]
        pressure_exp = [[j for _, j in enumerate(df_exp.loc[:, columns[frame]])],
                        [j for _, j in enumerate(df_exp.loc[:, columns[frame]])]]
        pressure_imp = [[j for _, j in enumerate(df_imp.loc[:, columns[frame]])],
                        [j for _, j in enumerate(df_imp.loc[:, columns[frame]])]]

        plt.cla()  # Limpa o eixo atual para atualizar o gráfico

        def plot_map(axis, dframe, label):
            axis.imshow(dframe, cmap='rainbow', interpolation='bicubic', origin='lower',
                        extent=(0, df_ana.index.max(), 0, 5),
                        norm=colors.Normalize(vmin=df_imp.min().min(), vmax=df_imp.max().max()))
            axis.set_yticks([])
            axis.set_ylabel(label)

        for nn, ax in enumerate(axs):

            if nn == 0:
                plot_map(ax, pressure_ana, 'Analitica')
            elif nn == 1:
                plot_map(ax, pressure_exp, 'Explícita')
            else:
                plot_map(ax, pressure_imp, 'Implícita')

    # Configuração do gráfico
    fig = plt.figure()
    subfigs = fig.subfigures(1, 1)
    axs = subfigs.subplots(3, 1, sharex=True)

    df_map = [[j for _, j in enumerate(df_imp.loc[:, :])],
              [j for _, j in enumerate(df_imp.loc[:, :])]]
    map_to_plot = plt.imshow(df_map, cmap='rainbow', interpolation='bicubic', origin='lower',
                             extent=(0, df_ana.index.max(), 0, 5),
                             norm=colors.Normalize(vmin=df_imp.min().min(), vmax=df_imp.max().max()))
    fig.colorbar(mappable=map_to_plot, ax=axs)

    ani = FuncAnimation(fig, update, frames=len(columns) - 1, interval=500)  # Intervalo de 1000ms entre frames

    # Salvar a animação como GIF
    ani.save(f'{root}\\animacao_map.gif', writer='pillow', fps=60)  # 1 frame por segundo
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------


def fo_erro(data_analitical: Df, data_method: Df, columns: list, n_cell: int) -> tuple:
    erro_list_dx = []  # Erro em relação ao n e dx
    erro_list_dt = []  # Erro em relação ao dt
    for col in data_method.columns:
        suma = 0
        if col in columns:
            # Erro em relação ao n e dx
            for index in data_method.index:
                suma += abs(data_analitical.loc[index, col] -
                            data_method.loc[index, col]) / data_analitical.loc[index, col]
            erro_coldx = np.sqrt((1 / n_cell) * suma)
            erro_list_dx.append(erro_coldx)

            # Erro em relação ao dt
            erro_list_dt.append(max((data_analitical.loc[:, col] - data_method.loc[:, col])))
    return erro_list_dx, erro_list_dt


def create_errordataframe_1d(explit_list: list, implicit_list: list, columns: list):
    data_erro = {'Explicit': explit_list, 'Implicit': implicit_list}
    data_columns_mesh = ['dx', 'dt', 'n', 'tempo']
    data_columns_error = []
    for i in columns:
        data_columns_error.append(f'Erro {i}')

    iterables_for_mesh = [['Mesh'], data_columns_mesh]
    series_mesh = pd.MultiIndex.from_product(iterables=iterables_for_mesh)
    iterables_for_error = [['L2 for dx', 'L3 for dt'], data_columns_error]
    series_error = pd.MultiIndex.from_product(iterables=iterables_for_error)

    s = series_mesh.union(series_error, sort=False)

    dataframe = Df([], columns=s, index=['Explicit', 'Implicit'])
    dataframe.loc['Explicit', :] = data_erro['Explicit']
    dataframe.loc['Implicit', :] = data_erro['Implicit']
    return dataframe


def create_mesh_1d(time_values: np.ndarray, n_cells: int, wellclass: object, method: str) -> tuple:
    """
    Function to generate the grid for simulating pressure field. You should pass the value of cells that you wish your
    grid to have.

    :param time_values:
    :param method:
    :param wellclass:
    :param n_cells: Number of cells that you wish to divide your grid. Must be an integer value.
    :return: The mesh dict of your problem with the internal points and the two contour points!
    """
    if type(n_cells) is not int:
        print(f'Error!!! The parameter "n_cells" must be set as an integer value. Type passed: {type(n_cells)}.')
        sys.exit()

    deltax = wellclass.res_length / n_cells
    initial_point = deltax / 2
    final_point = wellclass.res_length - deltax / 2
    x_array = np.linspace(initial_point, final_point, n_cells)  # internal points of the grid
    x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
    x_array = np.append(x_array, int(wellclass.res_length))  # insert the final contour point
    x_array = [round(i, ndigits=3) for i in x_array]
    grid = {i: x_array[i] for i in range(len(x_array))}

    delta_t = time_values[1] - time_values[0]
    if method == 'Explicit':
        wellclass.rx_explicit = delta_t / (deltax ** 2)
        wellclass.time_explicit = time_values
        if wellclass.rx_explicit * wellclass.eta >= 0.25:
            print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "(rx * eta) > 0.25".')
            print(f'rx = {wellclass.rx_explicit} // eta = {wellclass.eta}  // (rx * eta) = '
                  f'{wellclass.rx_explicit * wellclass.eta}')
            sys.exit()
        else:
            pass
    else:
        wellclass.rx_implicit = delta_t / (deltax ** 2)
        wellclass.time_implicit = time_values

    return grid, deltax


def create_mesh_2d(time_values: np.ndarray, n_cells: int, wellclass: object, method: str) -> tuple:
    """
        Function to generate the grid for simulating pressure field. You should pass the value of cells that you wish
        your grid to have.

        :param time_values: time discretization.
        :param method: numerical method used.
        :param wellclass: object case containing the case parameters.
        :param n_cells: Number of cells that you wish to divide your grid. Must be an integer value.
        :return: The mesh dict of your problem with the internal points and the all contour points!
    """
    if type(n_cells) is not int:
        print(f'Error!!! The parameter "n_cells" must be set as an integer value. Type passed: {type(n_cells)}.')
        sys.exit()

    deltax, deltay = wellclass.res_length / n_cells, wellclass.res_width / n_cells

    if deltax != deltay:
        pass

    initial_point_x = deltax / 2
    final_point_x = wellclass.res_length - deltax / 2
    x_array = np.linspace(initial_point_x, final_point_x, n_cells)  # internal points of the grid
    x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
    x_array = np.append(x_array, int(wellclass.res_length))  # insert the final contour point
    x_array = [round(i, ndigits=3) for i in x_array]
    wellclass.x_values = x_array

    initial_point_y = deltay / 2
    final_point_y = wellclass.res_width - deltax / 2
    y_array = np.linspace(initial_point_y, final_point_y, n_cells)  # internal points of the grid
    y_array = np.insert(y_array, 0, 0)  # insert the initial contour point
    y_array = np.append(y_array, int(wellclass.res_width))  # insert the final contour point
    y_array = [round(i, ndigits=3) for i in y_array]
    y_array.reverse()
    wellclass.y_values = y_array

    index_for_dataframe = np.linspace(0, n_cells + 1, n_cells + 2)
    index_column_x = [int(i) for i in index_for_dataframe]
    index_row_y = [int(i) for i in index_for_dataframe]

    grid = {i: Df([], columns=index_column_x, index=index_row_y) for i in time_values}

    delta_t = time_values[1] - time_values[0]
    if method == 'Explicit':
        wellclass.rx_explicit = delta_t / (deltax ** 2)
        wellclass.ry_explicit = delta_t / (deltay ** 2)
        wellclass.time_explicit = time_values
        wellclass.deltat = delta_t
        if wellclass.rx_explicit * wellclass.eta >= 0.25:
            print(f'Error: O critério de convergência não foi atingido para o eixo x. Parâmetro "(rx * eta) > 0.25".')
            print(f'rx = {wellclass.rx_explicit} // eta = {wellclass.eta}  // (rx * eta) = '
                  f'{wellclass.rx_explicit * wellclass.eta}')
            sys.exit()
        elif wellclass.ry_explicit * wellclass.eta >= 0.25:
            print(f'Error: O critério de convergência não foi atingido para o eixo y. Parâmetro "(ry * eta) > 0.25".')
            print(f'rx = {wellclass.ry_explicit} // eta = {wellclass.eta}  // (ry * eta) = '
                  f'{wellclass.ry_explicit * wellclass.eta}')
            sys.exit()
        else:
            pass
    else:
        wellclass.rx_implicit = delta_t / (deltax ** 2)
        wellclass.ry_implicit = delta_t / (deltay ** 2)
        wellclass.time_implicit = time_values
        wellclass.deltat = delta_t

    return grid, deltax, deltay


def create_dataframe(time: np.ndarray, n_cells: int) -> tuple:
    """
    Function that will create the dataframe table for the pressure field with relative mesh grid created.
    :return: A dataframe table that contains the mesh grid that was set.
    """
    # Setting the time points as the dataframe columns
    time_to_columns = [float(t) for t in time]
    # Setting the mesh points as the dataframe index. The points are not the positions in x, they are just the
    # equivalent cell for the positions.
    index_for_dataframe = np.linspace(0, n_cells + 1, n_cells + 2)
    index_for_dataframe = [int(i) for i in index_for_dataframe]
    # Creating the dataframe table with columns and index
    pressure = Df(float(0), index=index_for_dataframe, columns=time_to_columns)

    return pressure, time_to_columns, index_for_dataframe


def create_pressurecoeficients_pressureboundaries(n_cells: int, param_values: dict):
    field_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        if i == 0:
            field_matrix[i, 0], field_matrix[i, 1] = param_values['a'], param_values['b']
        elif i == n_cells - 1:
            field_matrix[i, n_cells - 2], field_matrix[i, n_cells - 1] = param_values['b'], param_values['a']
        else:
            field_matrix[i, i - 1], field_matrix[i, i], field_matrix[i, i + 1] = param_values['c'], param_values['d'], \
                param_values['c']

    constant_matrix = np.zeros(n_cells)
    constant_matrix[0] = param_values['f1']
    constant_matrix[-1] = param_values['fn']

    return field_matrix, constant_matrix


def create_pressurecoeficients_flowboundaries(n_cells: int, param_values: dict):
    field_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        if i == 0:
            field_matrix[i, 0], field_matrix[i, 1] = param_values['a'], param_values['b']
        elif i == n_cells - 1:
            field_matrix[i, n_cells - 2], field_matrix[i, n_cells - 1] = param_values['d'], param_values['e']
        else:
            field_matrix[i, i - 1], field_matrix[i, i], field_matrix[i, i + 1] = param_values['b'], param_values['c'], \
                param_values['b']

    constant_matrix = np.zeros(n_cells)
    constant_matrix[0] = param_values['f1']
    constant_matrix[-1] = param_values['fn']

    return field_matrix, constant_matrix


def calc_permeability(xi, xj):
    """
    Function to calculate the equivalent permeability between two cells in the mesh grid.
    :param xi:
    :param xj:
    :return: The equivalent permeability between the two cells [float].
    """

    # if xi == 0 and xj != 0:
    #     k_eq = 2 / (0 + (1 / xj))
    # elif xi != 0 and xj == 0:
    #     k_eq = 2 / ((1 / xi) + 0)
    # elif xi == 0 and xj == 0:
    #     k_eq = 0
    if xi == 0 or xj == 0:
        k_eq = 0
    else:
        k_eq = 2 / ((1 / xi) + (1 / xj))

    k_eq *= 9.869233e-16
    return k_eq


def find_indexs(dataframe, value):
    """
    Function to find index for column and row in the id matrix for the value passed
    :param dataframe:
    :param value:
    :return:
    """
    try:
        indexs = dataframe.where(dataframe == value).stack().dropna()
        return indexs.index[0]
    except BaseException as e:
        raise e


def create_pressurecoeficientes_flowboundaries2d(n_cells: int, map_permeability, rx, ry, beta, wellposition, ct, mi,
                                                 pho, dx, dy):
    """
        Function to create the coefficient matrix with mxm dimensions that will be used to run implicit simulation

        :param dx:
        :param dy:
        :param pho:
        :param mi:
        :param ct:
        :param wellposition:
        :param n_cells:
        :param map_permeability:
        :param rx:
        :param ry:
        :param beta:
        :return: The coefficient matrix for the problem, and the matrix of font terms.
    """

    index_for_map = np.linspace(1, n_cells, n_cells)
    index_for_map = [int(i) for i in index_for_map]
    map_permeability = map_permeability.set_index(pd.Index(index_for_map))
    map_permeability.columns = index_for_map

    matrix_id = np.arange(1, n_cells ** 2 + 1).reshape(n_cells, n_cells)
    matrix_id = Df(matrix_id, columns=index_for_map, index=index_for_map)

    index_for_coef = np.linspace(1, (n_cells ** 2), (n_cells ** 2))
    index_for_coef = [int(i) for i in index_for_coef]
    coefficient_matrix = Df(float(0), columns=index_for_coef, index=index_for_coef)

    cont = 1

    for line in coefficient_matrix:

        if line == 1:  # primeira célula da primeira linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (1,1) e (1,2)
            k_eq_rigth = calc_permeability(xi=map_permeability.loc[m, n],
                                           xj=map_permeability.loc[m, n + 1])
            # permeabilidade equivalente entre os blocos (1,1) e (2,1)
            k_eq_low = calc_permeability(xi=map_permeability.loc[m, n],
                                         xj=map_permeability.loc[m + 1, n])

            coefficient_matrix.loc[line, line] = (2 + (beta * ((ry * k_eq_low) + (rx * k_eq_rigth)))) / 2
            coefficient_matrix.loc[m, n + 1] = (- beta * rx * k_eq_rigth) / 2
            coefficient_matrix.loc[m, n + n_cells] = (- beta * ry * k_eq_low) / 2

        elif 1 < line < n_cells:  # células interiores da primeira linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (m,n) e (m+1,n)
            k_eq_low = calc_permeability(xi=map_permeability.loc[m, n],
                                         xj=map_permeability.loc[m + 1, n])

            coefficient_matrix.loc[line, line] = 1 + (beta * ry * k_eq_low)
            coefficient_matrix.loc[line, line + n_cells] = - beta * ry * k_eq_low

        elif line == n_cells:  # última célula da primeira linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (1,n) e (1,n-1)
            k_eq_left = calc_permeability(xi=map_permeability.loc[m, n],
                                          xj=map_permeability.loc[m, n - 1])
            # permeabilidade equivalente entre os blocos (1,n) e (2,n)
            k_eq_low = calc_permeability(xi=map_permeability.loc[m, n],
                                         xj=map_permeability.loc[m + 1, n])

            coefficient_matrix.loc[line, line] = (2 + (beta * ((ry * k_eq_low) + (rx * k_eq_left)))) / 2
            coefficient_matrix.loc[line, line - 1] = (- beta * rx * k_eq_left) / 2
            coefficient_matrix.loc[line, line + n_cells] = (- beta * ry * k_eq_low) / 2

        elif line == (n_cells ** 2 - (n_cells - 1)):  # primeira célula da última linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (m,1) e (m,2)
            k_eq_rigth = calc_permeability(xi=map_permeability.loc[m, n],
                                           xj=map_permeability.loc[m, n + 1])
            # permeabilidade equivalente entre os blocos (m,1) e (m-1,1)
            k_eq_righ = calc_permeability(xi=map_permeability.loc[m, n],
                                          xj=map_permeability.loc[m - 1, n])

            coefficient_matrix.loc[line, line] = (2 + (beta * ((ry * k_eq_righ) + (rx * k_eq_rigth)))) / 2
            coefficient_matrix.loc[line, line + 1] = (- beta * rx * k_eq_rigth) / 2
            coefficient_matrix.loc[line, line - n_cells] = (- beta * ry * k_eq_righ) / 2

        elif (n_cells ** 2 - (n_cells - 1)) < line < n_cells ** 2:  # células interiores da última linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (m,n) e (m-1,n)
            k_eq_righ = calc_permeability(xi=map_permeability.loc[m, n],
                                          xj=map_permeability.loc[m - 1, n])

            coefficient_matrix.loc[line, line] = 1 + (beta * ry * k_eq_righ)
            coefficient_matrix.loc[line, line - n_cells] = - beta * ry * k_eq_righ

        elif line == n_cells ** 2:  # última célula da última linha
            m, n = find_indexs(matrix_id, line)
            # permeabilidade equivalente entre os blocos (m,n) e (m,n-1)
            k_eq_left = calc_permeability(xi=map_permeability.loc[m, n],
                                          xj=map_permeability.loc[m, n - 1])
            # permeabilidade equivalente entre os blocos (m,n) e (m-1,n)
            k_eq_righ = calc_permeability(xi=map_permeability.loc[m, n],
                                          xj=map_permeability.loc[m - 1, n])

            coefficient_matrix.loc[line, line] = (2 + (beta * ((ry * k_eq_righ) + (rx * k_eq_left)))) / 2
            coefficient_matrix.loc[line, line - 1] = (- beta * rx * k_eq_left) / 2
            coefficient_matrix.loc[line, line - n_cells] = (- beta * ry * k_eq_righ) / 2

        else:
            if cont == 1:  # primeira célula das linhas interiores
                m, n = find_indexs(matrix_id, line)
                # permeabilidade equivalente entre os blocos (m,n) e (m,n+1)
                k_eq_mn1 = calc_permeability(xi=map_permeability.loc[m, n],
                                             xj=map_permeability.loc[m, n + 1])

                coefficient_matrix.loc[line, line] = 1 + (beta * rx * k_eq_mn1)
                coefficient_matrix.loc[line, line + 1] = - beta * rx * k_eq_mn1
                cont += 1

            elif 1 < cont < n_cells:  # células internas da malha
                m, n = find_indexs(matrix_id, line)
                # permeabilidade equivalente entre os blocos (m,n) e (m-1,n)
                k_eq_righ = calc_permeability(xi=map_permeability.loc[m, n],
                                              xj=map_permeability.loc[m - 1, n])
                # permeabilidade equivalente entre os blocos (m,n) e (m+1,n)
                k_eq_low = calc_permeability(xi=map_permeability.loc[m, n],
                                             xj=map_permeability.loc[m + 1, n])
                # permeabilidade equivalente entre os blocos (m,n) e (m,n-1)
                k_eq_left = calc_permeability(xi=map_permeability.loc[m, n],
                                              xj=map_permeability.loc[m, n - 1])
                # permeabilidade equivalente entre os blocos (m,n) e (m,n+1)
                k_eq_rigth = calc_permeability(xi=map_permeability.loc[m, n],
                                               xj=map_permeability.loc[m, n + 1])

                coefficient_matrix.loc[line, line] = (1 + (beta * rx * (k_eq_rigth + k_eq_left)) +
                                                      (beta * ry * (k_eq_low + k_eq_righ)))
                coefficient_matrix.loc[line, line + 1] = - beta * rx * k_eq_rigth
                coefficient_matrix.loc[line, line - 1] = - beta * rx * k_eq_left
                coefficient_matrix.loc[line, line + n_cells] = - beta * ry * k_eq_low
                coefficient_matrix.loc[line, line - n_cells] = - beta * ry * k_eq_righ
                cont += 1

            else:  # última célula das linhas interiores (cont = n_cells)
                m, n = find_indexs(matrix_id, line)
                # permeabilidade equivalente entre os blocos (m,n) e (m,n-1)
                k_eq_left = calc_permeability(xi=map_permeability.loc[m, n],
                                              xj=map_permeability.loc[m, n - 1])

                coefficient_matrix.loc[line, line] = 1 + (beta * rx * k_eq_left)
                coefficient_matrix.loc[line, line - 1] = - beta * rx * k_eq_left
                cont = 1

    font_term = np.zeros(n_cells ** 2)

    for well, item in wellposition.items():
        m = item.line
        n = item.column
        radius = item.radius
        item.permeability = map_permeability.loc[m, n] * 9.869233e-16
        pressure = item.pressure
        flow = item.flow
        type_well = item.type
        eta = item.permeability / (pho * mi * ct)
        line = matrix_id.loc[m, n]

        r_eq = np.sqrt((dx * dy) / np.pi)
        item.equivalent_radius = r_eq
        gama = (2 * eta * np.pi) / (dx * dy * np.log(r_eq / radius))

        if type_well == 'Production':
            coefficient_matrix.loc[line, line] += gama
            font_term[line - 1] = gama * pressure
        else:
            coefficient_matrix.loc[line, line] -= gama
            font_term[line - 1] = - gama * pressure

    coefficient_matrix = coefficient_matrix.to_numpy()

    return coefficient_matrix, font_term


def get_object_case(fluxtype: str, well_condiction=None, external_condiction=None,
                    top_condiction=None, base_condiction=None, left_condition=None,
                    right_condition=None):
    """
    :param left_condition: Left boundary condiction. Must be Pressure (P) or Flow (F).
    :param right_condition: Right boundary condiction. Must be Pressure (P) or Flow (F).
    :param well_condiction: Well boundary condiction. Must be Pressure (P) or Flow (F).
    :param external_condiction: External boundary condiction. Must be Pressure (P) or Flow (F).
    :param top_condiction: Top boundary condiction. Must be Pressure (P) or Flow (F).
    :param base_condiction: Base boundary condiction. Must be Pressure (P) or Flow (F).
    :param fluxtype: Flow dimension - 1D, 2D or 3D

    :return: The class corresponding to the simulation condictions.
    """
    list_condiction = ['P', 'F', None]
    flow_type = ['1D', '2D', '3D']

    if well_condiction not in list_condiction:
        print('Error: well boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if external_condiction not in list_condiction:
        print('Error: external boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if top_condiction not in list_condiction:
        print('Error: top boundary must be P (Pressure), F (Flow) or None if 1D flow.')
        sys.exit()

    if base_condiction not in list_condiction:
        print('Error: base boundary must be P (Pressure), F (Flow) or None if 1D flow.')
        sys.exit()

    if left_condition not in list_condiction:
        print('Error: Left boundary must be P (Pressure), F (Flow) or None if 1D flow.')
        sys.exit()

    if right_condition not in list_condiction:
        print('Error: Left boundary must be P (Pressure), F (Flow) or None if 1D flow.')
        sys.exit()

    if fluxtype not in flow_type:
        print('Error: the parameter fluxtype must be 1D, 2D or 3D.')
        sys.exit()

    if fluxtype == '1D':

        if well_condiction.upper() == 'P' and external_condiction.upper() == 'P':
            condiction = 'PressurePressure'

        elif well_condiction.upper() == 'F' and external_condiction.upper() == 'P':
            condiction = 'FlowPressure'

        else:
            condiction = 'FlowFlow'

        return Object_Case.OneDimensionalFlowCase(condiction=condiction)

    elif fluxtype == '2D':

        if (left_condition.upper() == 'P' and right_condition.upper() == 'P' and top_condiction.upper() == 'P'
                and base_condiction.upper() == 'P'):
            condiction = 'Pressure Boundaires '

        elif (left_condition.upper() == 'P' and right_condition.upper() == 'P' and top_condiction.upper() == 'F'
              and base_condiction.upper() == 'F'):
            condiction = 'Pressure LR and Flow TB '

        elif (left_condition.upper() == 'F' and right_condition.upper() == 'P' and top_condiction.upper() == 'P'
              and base_condiction.upper() == 'P'):
            condiction = 'Flow left and Pressure Boundaries '

        elif (left_condition.upper() == 'F' and right_condition.upper() == 'P' and top_condiction.upper() == 'F'
              and base_condiction.upper() == 'F'):
            condiction = 'Flow LTB with Pressure R '

        elif (left_condition.upper() == 'F' and right_condition.upper() == 'F' and top_condiction.upper() == 'P'
              and base_condiction.upper() == 'P'):
            condiction = 'Flow LR and Pressure TP '

        elif (left_condition.upper() == 'F' and right_condition.upper() == 'F' and top_condiction.upper() == 'F'
              and base_condiction.upper() == 'F'):
            condiction = 'Flow Boundaries '

        else:
            print('Error: You must set one possible condition combination. To see the possible combinations, '
                  'take a look in the readme file!')
            sys.exit()

        return Object_Case.TwoDimensionalFlowCase(condiction=condiction)


def get_object_mesh(flow_type: str, wellobject: object):
    """

    :param wellobject: Object containing all the case parameters.
    :param flow_type: Type of flux. Must be 1D, 2D or 3D.
    :return:
    """
    flux_type_dinam = ['1D', '2D', '3D']
    if flow_type not in flux_type_dinam:
        print("Error! The flow dinamics must be 1D, 2D or 3D.")
        sys.exit()

    if flow_type == '1D':
        return Object_Simulation.OneDimensionalFlowMesh(wellcase=wellobject)
    elif flow_type == '2D':
        return Object_Simulation.TwoDimensionalFlowMesh(wellcase=wellobject)
    else:
        return Object_Simulation.ThreeDimensionalFlowMesh


def set_object_simulation(flowtype: str, method=None):
    if flowtype == '1D':
        return (AnaliticalMethod.OneDimensionalAnaliticalMethod(), ExplicitMethod.OneDimensionalExplicitMethod(),
                ImplicitMethod.OneDimensionalImplicitMethod())
    elif flowtype == '2D':
        if method == 'explicit':
            return ExplicitMethod.TwoDimensionalExplicitMethod()
        elif method == 'implicit':
            return ImplicitMethod.TwoDimensionalImplicitMethod()
        else:
            print('Error: The method you choose was not implemented yet.')
            sys.exit()
    else:
        pass


def flowboundaries2d(grid: Df, n_cells: int):
    # Boundaries but corners -------------------------------------------------------------------------------------------
    grid.loc[1:n_cells, 0] = grid.loc[1:n_cells, 1]  # First column (x = 0) except for corners
    grid.loc[1:n_cells, n_cells + 1] = grid.loc[1:n_cells, n_cells]  # Last column (x = length in x) except for corners
    grid.loc[0, 1:n_cells] = grid.loc[1, 1:n_cells]  # First row (y = length in y) except for corners
    grid.loc[n_cells + 1, 1:n_cells] = grid.loc[n_cells, 1:n_cells]  # Last row (y = 0) except for corners

    # Corners ----------------------------------------------------------------------------------------------------------
    # top-left corner (x = 0, y = length in y)
    grid.loc[0, 0] = (grid.loc[0, 1] + grid.loc[1, 0]) / 2
    # base-left corner (x = 0, y = 0)
    grid.loc[n_cells + 1, 0] = (grid.loc[n_cells + 1, 1] + grid.loc[n_cells, 0]) / 2
    # top-rigth corner (x = length in x, y = length in y)
    grid.loc[0, n_cells + 1] = (grid.loc[0, n_cells] + grid.loc[1, n_cells + 1]) / 2
    # base-rigth corner (x = length in x, y = 0)
    grid.loc[n_cells + 1, n_cells + 1] = (grid.loc[n_cells, n_cells + 1] + grid.loc[n_cells + 1, n_cells]) / 2

    return grid


def plot_animation_map_2d(grid: dict, name: str, path: str):
    times_key = list(grid.keys())
    data_keys = list(grid.values())
    frames = [i for i in range(len(times_key))]

    # Converta os dados do DataFrame para tipo numérico e trate valores nulos
    for key in times_key:
        grid[key] = grid[key].apply(pd.to_numeric, errors='coerce').fillna(0)

    # ------------------------------------------------------------------------------------------------------------------
    def update(frame):
        if frame == 0:
            pass
        else:
            dataframe = grid[times_key[frame]]

            plt.cla()  # Limpa o eixo atual para atualizar o gráfico

            plt.imshow(dataframe, cmap='rainbow', interpolation='bicubic', origin='upper',
                       extent=(0, dataframe.index.max(), 0, dataframe.columns.max()),
                       norm=colors.Normalize(vmin=a, vmax=b)
                       )

            plt.xlabel('Comprimento x (m)')
            plt.ylabel('Comprimento y (m)')
            plt.title("Mapa de Pressão")
            plt.tight_layout()

    # Configuração do gráfico
    fig, ax = plt.subplots()
    df_map = grid[times_key[0]]
    a = grid[times_key[-1]].min().min()
    b = grid[times_key[0]].max().max()
    map_to_plot = plt.imshow(df_map, cmap='rainbow', interpolation='bicubic', origin='lower',
                             extent=(0, df_map.index.max(), 0, df_map.columns.max()),
                             norm=colors.Normalize(vmin=a, vmax=b))
    fig.colorbar(mappable=map_to_plot, ax=ax)
    plt.xlabel('Comprimento x (m)')
    plt.ylabel('Comprimento y (m)')
    plt.title("Mapa de Pressão")
    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=len(times_key) - 1, interval=500)  # Intervalo de 1000ms entre frames

    # Salvar a animação como GIF
    ani.save(f'{path}\\animacao_map.gif', writer='pillow', fps=60)  # 1 frame por segundo
    plt.close()


def plot_graphs_2d(welldata: dict, time_values: list, path: str):
    for well, data in welldata.items():
        if data.type == 'Production':
            # gridspec inside gridspec
            fig = plt.figure(layout='constrained', figsize=(10, 6))
            subfigs = fig.subfigures(1, 2, wspace=0.07)

            # axis
            leftaxis = subfigs[0].subplots(1, 1)  # For production curve
            rigthaxis = subfigs[1].subplots(1, 1)   # For wellflow curve

            # plots
            leftaxis.plot(time_values, data.production)
            leftaxis.set_ylabel('Production (m³)')
            leftaxis.set_xlabel('Time (s)')

            rigthaxis.plot(time_values[1:], data.flow[1:])
            rigthaxis.set_ylabel('Flow (m³/s)')
            rigthaxis.set_xlabel('Time (s)')

            # Titles
            subfigs[0].suptitle('Production', fontsize='x-large')
            subfigs[1].suptitle('Flow', fontsize='x-large')
            fig.suptitle('Accumulated Production Curves', fontsize='xx-large')

            fig.savefig(f'{path}\\CurveAnalysis _ {well}.png')
            plt.close('all')

    if len(welldata.keys()) > 1:
        prod_sum = np.zeros((len(welldata['well 1'].production)))
        flow_sum = np.zeros((len(welldata['well 1'].flow)))
        for values in welldata.values():
            for i in range(len(prod_sum)):
                prod_sum[i] += values.production[i]
                flow_sum[i] += values.flow[i]

        fig = plt.figure(layout='constrained', figsize=(10, 6))
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        # axis
        leftaxis = subfigs[0].subplots(1, 1)  # For production curve
        rigthaxis = subfigs[1].subplots(1, 1)  # For wellflow curve

        # plots
        leftaxis.plot(time_values, prod_sum)
        leftaxis.set_ylabel('Production (m³)')
        leftaxis.set_xlabel('Time (s)')

        rigthaxis.plot(time_values[1:], flow_sum[1:])
        rigthaxis.set_ylabel('Flow (m³/s)')
        rigthaxis.set_xlabel('Time (s)')

        # Titles
        subfigs[0].suptitle('Production', fontsize='x-large')
        subfigs[1].suptitle('Flow', fontsize='x-large')
        fig.suptitle('Accumulated Production Curves - Wells', fontsize='xx-large')

        fig.savefig(f'{path}\\CurveAnalysis _ AllWells.png')
        plt.close('all')
