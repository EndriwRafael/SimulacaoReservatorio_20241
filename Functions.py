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

    if arq_ana.index.all() != arq_num.index.all():
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
