import matplotlib.pyplot as plt
from pandas import DataFrame as Df
import numpy as np
import random as rm
from matplotlib.lines import Line2D
import sys


def set_color(list_color: list):
    while True:
        a = rm.randint(0, 9)
        if f'C{a}' not in list_color:
            list_color.append(f'C{a}')
            break
    return f'C{a}'


def plot_graphs_compare(root: str, arq_ana: Df, arq_num: Df, time: np.ndarray):
    """
    Function to plot graphs from both methods (analitical and numerical) to compare curves.

    :param root: Path that contains the results from methods to save the final graph. Must be string.
    :param arq_ana: Dataframe that refers to the result of analitical solution.
    :param arq_num: Dataframe that refers to the result of numerical solution.
    :param time: Array for times that will be plot on graph.
    :return: Plot and save the comparision graph.
    """

    arq_ana_index = arq_ana.index.values
    arq_num_index = arq_num.index.values
    if arq_ana_index.all() != arq_num_index.all():
        print('Error! Os dataframes disponibilizados possuem discretização de malha diferentes.')
        sys.exit()

    color_list = []
    fig, ax = plt.subplots()
    for t in time:
        if t not in arq_ana.columns or t not in arq_num.columns:
            print(f'Error! O valor selecionado (t = {t}) não foi encontrado nos dataframes disponibilizados.')
            sys.exit()

        if len(color_list) == 10:
            color_list = []

        color = set_color(list_color=color_list)

        ax.plot(arq_ana.index, arq_ana[t], '.', color=color, label='_nolegend_')
        ax.plot(arq_num.index, arq_num[t], '-', color=color, label='_nolegend_')

        color_list.append(color)

    # Criação manual das entradas da legenda
    legend_elements = [Line2D([0], [0], linestyle='dotted', color='black', label='Analítico'),
                       Line2D([0], [0], linestyle='-', color='black', label='Numérico')]

    plt.ticklabel_format(axis='y', style='plain')
    plt.xlabel('Comprimento (m)')
    plt.ylabel('Pressão (psia)')
    plt.title('Comparação das Curvas (Analítico e Numérico)')
    ax.legend(framealpha=1, handles=legend_elements)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f'{root}\\ComparacaoFinal.png')
    plt.close()


def create_mesh(well_class: object, time_values: np.ndarray, n_cells: int = 0, deltax: float or int = 0):
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
        well_class.n_cells = n_cells
        well_class.deltax = well_class.res_length / well_class.n_cells
        initial_point = well_class.deltax / 2
        final_point = well_class.res_length - well_class.deltax / 2
        x_array = np.linspace(initial_point, final_point, well_class.n_cells)  # internal points of the grid
        x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
        x_array = np.append(x_array, int(well_class.res_length))  # insert the final contour point
        x_array = [round(i, ndigits=3) for i in x_array]
        well_class.mesh = {i: x_array[i] for i in range(len(x_array))}
    else:  # the creation of the grid depends on the value of deltax
        well_class.deltax = deltax
        well_class.n_cells = int(well_class.res_length / well_class.deltax)
        initial_point = well_class.deltax / 2
        final_point = well_class.res_length - well_class.deltax / 2
        x_array = np.linspace(initial_point, final_point, well_class.n_cells)  # internal points of the grid
        x_array = np.insert(x_array, 0, 0)  # insert the initial contour point
        x_array = np.append(x_array, int(well_class.res_length))  # insert the final contour point
        x_array = [round(i, ndigits=3) for i in x_array]
        well_class.mesh = {i: x_array[i] for i in range(len(x_array))}

    delta_t = time_values[1] - time_values[0]
    r_x = delta_t / (well_class.deltax ** 2)

    if r_x * well_class.eta >= 0.25:
        print(f'Error!!! O critério de convergência não foi atingido. Parâmetro "(rx * eta) > 0.25".')
        print(f'rx = {r_x} // eta = {well_class.eta}  // (rx * eta) = {r_x * well_class.eta}')
        sys.exit()
    else:
        well_class.rx = r_x
