import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from pandas import DataFrame as Df
import numpy as np
import random as rm
from matplotlib.lines import Line2D
import sys
import Object_Case
import Object_Simulation
from Simuladores import Analitical_OneDimensional, Explicit_OneDimensional, Implicit_OneDimensional


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
    plt.title("Comparação das Curvas (Analítico '-' , Explicito 'o' e Implicito '^'")
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
    columns = [coll for coll in df_imp.columns if coll != 0.0]

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
    ani = FuncAnimation(fig, update, frames=len(df_imp.columns) - 1, interval=1000)  # Intervalo de 1000ms entre frames

    # Salvar a animação como GIF
    ani.save(f'{root}\\animacao_dataframe.gif', writer='pillow', fps=1)  # 1 frame por segundo
    pass


def fo_erro(data_analitical: Df, data_method: Df, columns: list, n_cell: int):
    erro_list = []
    # for data in columns:
    #     suma = 0
    for col in data_method.columns:
        suma = 0
        if col in columns:
            for index in data_method.index:
                suma += abs(data_analitical.loc[index, col] -
                            data_method.loc[index, col]) / data_analitical.loc[index, col]
            erro_col = np.sqrt((1 / n_cell) * suma)
            erro_list.append(erro_col)
    return erro_list


def create_errordataframe_1d(explit_list: list, implicit_list: list, columns: list):
    data_erro = {'Explicit': explit_list, 'Implicit': implicit_list}
    data_columns = ['dx', 'dt', 'tempo', 'n']
    for i in columns:
        data_columns.append(f'Erro {i}')

    dataframe = Df([], columns=data_columns, index=['Explicit', 'Implicit'])
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


def get_object_case(well_condiction: str, external_condiction: str):
    """

    :param well_condiction: Well boundary condiction. Must be Pressure (P) or Flow (F).
    :param external_condiction: External boundary condiction. Must be Pressure (P) or Flow (F).
    :return: The class corresponding to the simulation condictions.
    """
    if well_condiction != 'P' and well_condiction != 'F':
        print('Error!!! well boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if external_condiction != 'P' and external_condiction != 'F':
        print('Error!!! external boundary must be P (Pressure) or F (Flow).')
        sys.exit()

    if well_condiction.upper() == 'P' and external_condiction.upper() == 'P':
        return Object_Case.PressureBoundaries()
    elif well_condiction.upper() == 'F' and external_condiction.upper() == 'P':
        return Object_Case.FlowPressureBoundaries()
    else:
        return Object_Case.FlowBoundaries()


def get_object_mesh(flow_type: str, wellobject: object):
    """

    :param wellobject:
    :param flow_type:
    :return:
    """
    if flow_type != '1D' and flow_type != '2D' and flow_type != '3D':
        print("Error! The flow dinamics must be 1D, 2D or 3D.")
        sys.exit()

    if flow_type == '1D':
        return Object_Simulation.OneDimensionalFlowMesh(wellcase=wellobject)
    elif flow_type == '2D':
        return Object_Simulation.TwoDimensionalFlowMesh
    else:
        return Object_Simulation.ThreeDimensionalFlowMesh


def set_object_simulation(boundaries: str):
    if boundaries == 'PP':
        return (Analitical_OneDimensional.PressureBoundaries(), Explicit_OneDimensional.PressureBoundaries(),
                Implicit_OneDimensional.PressureBoundaries())
    elif boundaries == 'FP':
        return (Analitical_OneDimensional.WellFlowAndPressureBoundaries(),
                Explicit_OneDimensional.WellFlowAndPressureBoundaries(),
                Implicit_OneDimensional.WellFlowAndPressureBoundaries())
    else:
        print('Error!!! The flow bondaries simulator was not implemented yet.')
        sys.exit()
