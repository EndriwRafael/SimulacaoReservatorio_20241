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
            print(f'rx = {wellclass.r_x_explicit} // eta = {wellclass.eta}  // (rx * eta) = '
                  f'{wellclass.r_x_explicit * wellclass.eta}')
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

    deltax, deltay = wellclass.res_length / n_cells, wellclass.res_thickness / n_cells

    if deltax != deltay:
        print('Error! the discretization is not equal for x and y axis.')
        sys.exit()

    initial_point = deltax / 2

    final_point_x = wellclass.res_length - deltax / 2
    x_array = np.linspace(initial_point, final_point_x, n_cells)  # internal points of the grid
    x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
    x_array = np.append(x_array, int(wellclass.res_length))  # insert the final contour point
    x_array = [round(i, ndigits=3) for i in x_array]

    final_point_y = wellclass.res_thickness - deltax / 2
    y_array = np.linspace(initial_point, final_point_y, n_cells)  # internal points of the grid
    y_array = np.insert(y_array, 0, 0)  # insert the initial contour point
    y_array = np.append(y_array, int(wellclass.res_thickness))  # insert the final contour point
    y_array = [round(i, ndigits=3) for i in y_array]
    y_array.reverse()

    grid = {i: Df([], columns=x_array, index=y_array) for i in time_values}

    delta_t = time_values[1] - time_values[0]
    if method == 'Explicit':
        wellclass.rx_explicit = delta_t / (deltax ** 2)
        wellclass.time_explicit = time_values
        if wellclass.rx_explicit * wellclass.eta >= 0.25:
            print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "(rx * eta) > 0.25".')
            print(f'rx = {wellclass.r_x_explicit} // eta = {wellclass.eta}  // (rx * eta) = '
                  f'{wellclass.r_x_explicit * wellclass.eta}')
            sys.exit()
        else:
            pass
    else:
        wellclass.rx_implicit = delta_t / (deltax ** 2)
        wellclass.time_implicit = time_values

    return grid, deltax


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


def get_object_case(well_condiction: str, external_condiction: str, top_condiction: str or None,
                    base_condiction: str or None,
                    fluxtype: str):
    """
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
        print('Error!!! well boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if external_condiction not in list_condiction:
        print('Error!!! external boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if top_condiction not in list_condiction:
        print('Error!!! top boundary must be P (Pressure), F (Flow) or None (if 1D flow).')
        sys.exit()

    if base_condiction not in list_condiction:
        print('Error! base boundary must be P (Pressure), F (Flow) or None (if 1D flow).')
        sys.exit()

    if fluxtype not in flow_type:
        print('Error!!! the parameter fluxtype must be 1D, 2D or 3D.')

    if fluxtype == '1D':

        if well_condiction.upper() == 'P' and external_condiction.upper() == 'P':
            condiction = 'PressurePressure'

        elif well_condiction.upper() == 'F' and external_condiction.upper() == 'P':
            condiction = 'FlowPressure'

        else:
            condiction = 'FlowFlow'

        return Object_Case.OneDimensionalFlowCase(condiction=condiction)

    elif fluxtype == '2D':

        if (well_condiction.upper() == 'P' and external_condiction.upper() == 'P' and top_condiction.upper() == 'P'
                and base_condiction.upper() == 'P'):
            condiction = 'Pressure Boundaires Only '

        elif (well_condiction.upper() == 'P' and external_condiction.upper() == 'P' and top_condiction.upper() == 'F'
              and base_condiction.upper() == 'F'):
            condiction = 'Pressure Well_Edge and Flow Boundaries '

        elif (well_condiction.upper() == 'F' and external_condiction.upper() == 'P' and top_condiction.upper() == 'P'
              and base_condiction.upper() == 'P'):
            condiction = 'Flow Well and Pressure Boundaries '

        elif (well_condiction.upper() == 'F' and external_condiction.upper() == 'P' and top_condiction.upper() == 'F'
              and base_condiction.upper() == 'F'):
            condiction = 'Flow Well with Pressure Edge and Flow Boundaries '

        elif (well_condiction.upper() == 'F' and external_condiction.upper() == 'F' and top_condiction.upper() == 'P'
              and base_condiction.upper() == 'P'):
            condiction = 'Flow Well_Edge and Pressure Boundaries '

        else:
            condiction = 'Flow Boundaries Only '

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


def set_object_simulation(flowtype: str):
    if flowtype == '1D':
        return (AnaliticalMethod.OneDimensionalAnaliticalMethod(), ExplicitMethod.OneDimensionalExplicitMethod(),
                ImplicitMethod.OneDimensionalImplicitMethod())
    elif flowtype == '2D':
        pass
    else:
        pass
