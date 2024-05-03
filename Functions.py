import matplotlib.pyplot as plt
from pandas import DataFrame as Df
import numpy as np
import random as rm
from matplotlib.lines import Line2D
import sys
import Objects_Cases as ObjC
import Object_Case
import Object_Simulation


def set_color(list_color: list):
    while True:
        a = rm.randint(0, 9)
        if f'C{a}' not in list_color:
            break
    return f'C{a}'


def plot_graphs_compare(root: str, dataclass: object, time: np.ndarray):
    """
    Function to plot graphs from both methods (analitical and numerical) to compare curves.

    :param root: Path that contains the results from methods to save the final graph. Must be string.
    :param dataclass: Class object that contains the simulation results.
    :param time: Array for times that will be plot on graph.
    :return: Plot and save the comparision graph.
    """

    analitical_solution = dataclass.dataframe_to_analitical
    explicit_solution = dataclass.dataframe_to_explicit
    implicit_solution = dataclass.dataframe_to_implicit

    # conjunto1, conjunto2 = set(explicit_solution.index.values), set(implicit_solution.index.values)

    color_list = []
    fig, ax = plt.subplots()
    for t in time:

        if len(color_list) == 10:
            color_list = []

        color_list.append(set_color(list_color=color_list))

        # color: color_list[-1]
        ax.plot(analitical_solution.index, analitical_solution[t], '-', color='black', label='_nolegend_', linewidth=1)
        ax.scatter(explicit_solution.index, explicit_solution[t], marker='o', s=5,
                   color='blue', label='_nolegend_')
        ax.scatter(implicit_solution.index, implicit_solution[t], marker='^', s=2,
                   color='red', label='_nolegend_')

    # Criação manual das entradas da legenda
    legend_elements = [Line2D([0], [0], linestyle='-', color='black', label='Analítico'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Explicit'),
                       Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Implicit')]

    plt.ticklabel_format(axis='y', style='plain')
    plt.xlabel('Comprimento (m)')
    plt.ylabel('Pressão (psia)')
    plt.title('Comparação das Curvas (Analítico e Numérico)')
    ax.legend(framealpha=1, handles=legend_elements)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{root}\\ComparacaoFinal.png')
    plt.close()


def plot_animation_compare(root: str, dataclass: object, time: np.ndarray):
    """
    Function to create and plot animation from simulation's dataframes to compare the results.

    :param root: Path that contains the results from methods to save the final graph. Must be string.
    :param dataclass: Class object that contains the simulation results.
    :param time: Array for times that will be plot on graph.
    :return: Plot and save the comparision animation.
    """

    args = {atributo: valor for atributo, valor in dataclass.__dict__.items() if Df == type(valor)}
    limx, limy = [min(dataclass.dataframe_to_explicit.index), max(dataclass.dataframe_to_explicit.index)], \
        [dataclass.well_pressure, dataclass.initial_pressure]

    ObjC.Animation(root=root, dict_frames=args, t_values=time, limits=[limx, limy])


def calc_erro(root: str, dataclass: object, time: list):
    args = {atributo: valor for atributo, valor in dataclass.__dict__.items() if Df == type(valor)}  # type(valor) == Df
    conjunto_analitical, conjunto_explicit, conjunto_implicit = set(args['dataframe_to_analitical'].index.values), \
        set(args['dataframe_to_explicit'].index.values), set(args['dataframe_to_implicit'].index.values)

    equal_index_explicit = conjunto_analitical & conjunto_explicit
    equal_index_implicit = conjunto_analitical & conjunto_implicit

    dict_erro_explicit = {t: .0 for t in time}
    for key, values in dict_erro_explicit.items():
        suma = 0
        for index in equal_index_explicit:
            suma += (abs(args['dataframe_to_analitical'].loc[index, key] -
                         args['dataframe_to_explicit'].loc[index, key]) /
                     args['dataframe_to_analitical'].loc[index, key])
        dict_erro_explicit[key] = np.sqrt((1/dataclass.n_cells_explicit) * suma)

    dict_erro_implicit = {t: .0 for t in time}
    for key, values in dict_erro_implicit.items():
        suma = 0
        for index in equal_index_implicit:
            suma += (abs(args['dataframe_to_analitical'].loc[index, key] -
                         args['dataframe_to_implicit'].loc[index, key]) /
                     args['dataframe_to_analitical'].loc[index, key])
        dict_erro_implicit[key] = np.sqrt((1 / dataclass.n_cells_implicit) * suma)

    return []


def create_mesh(well_class: object, time_values: np.ndarray, method: str, n_cells: int = 0, deltax: float or int = 0):
    """
    Function to generate the grid for simulating pressure field. You can pass the value of cells that you wish your
    grid to have or the distance between the points. If you set one of then, you must set the other value equal to
    zero.

    :param well_class: Class containing the initialized data for the simulation.
    :param n_cells: Number of cells that you wish to divide your grid. Must be an integer value. Set n_cells = 0 if
    you pass deltax!
    :param deltax: The distance between the points in the grid. It can be an integer or a float value. Set deltax =
    0 if you pass n_cells!
    :param time_values: The
    :param method:
    :return: The mesh dict of your problem with the internal points and the two contour points!
    """
    if type(n_cells) is not int:
        print(f'Error!!! The parameter "n_cells" must be set if an integer value. Type passed: {type(n_cells)}.')
        sys.exit()

    if n_cells != 0 and deltax != 0:
        print(f'Error!!! Both parameters were set non-zero. You must set at least one of then different from zero.'
              f'Or you can set just one of then.')

    if n_cells == 0 and deltax == 0:
        print(f'Error! Both parameters were set if value zero. '
              f'You must set at least one of then non-zero. Or you can set just one of then.')
        sys.exit()

    if method != 'Explicit' and method != 'Implicit' and method != 'Analitical':
        print(f'Error!!! The parameter "method" must be set as "Explicit", "Implicit" or "Analitical".')
        sys.exit()

    if deltax == 0:  # the creation of the grid depends on the number of cells
        deltax = well_class.res_length / n_cells
        initial_point = deltax / 2
        final_point = well_class.res_length - deltax / 2
        x_array = np.linspace(initial_point, final_point, n_cells)  # internal points of the grid
        x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
        x_array = np.append(x_array, int(well_class.res_length))  # insert the final contour point
        x_array = [round(i, ndigits=3) for i in x_array]
        grid = {i: x_array[i] for i in range(len(x_array))}
        # well_class.mesh = {i: x_array[i] for i in range(len(x_array))}
        ObjC.MeshCases(grid=grid, cells=n_cells, method=method, well_method=well_class)
    else:  # the creation of the grid depends on the value of deltax
        n_cells = int(well_class.res_length / deltax)
        initial_point = deltax / 2
        final_point = well_class.res_length - deltax / 2
        x_array = np.linspace(initial_point, final_point, n_cells)  # internal points of the grid
        x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
        x_array = np.append(x_array, int(well_class.res_length))  # insert the final contour point
        x_array = [round(i, ndigits=3) for i in x_array]
        grid = {i: x_array[i] for i in range(len(x_array))}
        # well_class.mesh = {i: x_array[i] for i in range(len(x_array))}
        ObjC.MeshCases(grid=grid, cells=n_cells, method=method, well_method=well_class)

    delta_t = time_values[1] - time_values[0]

    if method == 'Explicit':
        r_x_explicit = delta_t / (deltax ** 2)
        well_class.rx_explicit = r_x_explicit
        well_class.deltax_explicit = deltax
        if r_x_explicit * well_class.eta >= 0.25:
            print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "(rx * eta) > 0.25".')
            print(f'rx = {r_x_explicit} // eta = {well_class.eta}  // (rx * eta) = {r_x_explicit * well_class.eta}')
            sys.exit()
        else:
            pass
    else:
        well_class.rx_implicit = delta_t / (deltax ** 2)
        well_class.deltax_implicit = deltax


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
        # r_x_explicit = delta_t / (deltax ** 2)
        wellclass.rx_explicit = delta_t / (deltax ** 2)
        # well_class.deltax_explicit = deltax
        if wellclass.rx_explicit * wellclass.eta >= 0.25:
            print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "(rx * eta) > 0.25".')
            print(f'rx = {wellclass.r_x_explicit} // eta = {wellclass.eta}  // (rx * eta) = '
                  f'{wellclass.r_x_explicit * wellclass.eta}')
            sys.exit()
        else:
            pass
    else:
        wellclass.rx_implicit = delta_t / (deltax ** 2)
        # wellclass.deltax_implicit = deltax

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
            field_matrix[i, n_cells-2], field_matrix[i, n_cells-1] = param_values['b'], param_values['a']
        else:
            field_matrix[i, i-1], field_matrix[i, i], field_matrix[i, i+1] = param_values['c'], param_values['d'], \
                param_values['c']

    constant_matrix = np.zeros(n_cells)
    constant_matrix[0] = param_values['f1']
    constant_matrix[-1] = param_values['fn']

    return field_matrix,  constant_matrix


def create_pressurecoeficients_flowboundaries(n_cells: int, param_values: dict):

    field_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        if i == 0:
            field_matrix[i, 0], field_matrix[i, 1] = param_values['a'], param_values['b']
        elif i == n_cells - 1:
            field_matrix[i, n_cells-2], field_matrix[i, n_cells-1] = param_values['d'], param_values['e']
        else:
            field_matrix[i, i-1], field_matrix[i, i], field_matrix[i, i+1] = param_values['b'], param_values['c'], \
                param_values['b']

    constant_matrix = np.zeros(n_cells)
    constant_matrix[0] = param_values['f1']
    constant_matrix[-1] = param_values['fn']

    return field_matrix,  constant_matrix


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
