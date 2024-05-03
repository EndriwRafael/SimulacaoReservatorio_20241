"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condições de Neumann nas duas Extremidades
"""
import Objects_Cases as Case
from Simuladores_1DFlow import Numerical_EXPLICIT as NsimExp
import Functions
import numpy as np
import pandas as pd

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000.
pressure_well = 9000000.
length_reser = 20.
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 30.
thickness = 10.
wellflow = 0.01
injectivityflow = 0.01

''' Inicializando simuladores Fluxo - Fluxo ------------------------------------------------------------------------ '''
case = Case.FlowBoundaries(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                                    res_thick=thickness, porosity=porosit, viscosity=viscosi,
                                    compressibility=compressibi, wellflow=wellflow, permeability=permeabi,
                                    res_len=length_reser, injectflow=injectivityflow)

''' Discretização da malha ----------------------------------------------------------------------------------------- '''
# Valores de discretização devem ser conferidos antes de rodar, por conta do critério de convergência. Caso os valores
# estejam incoerentes, o código retornar um erro avisando que o critério de convergência não foi respeitado!
t_explicit = np.linspace(start=0, stop=100, num=401)
t_implicit = np.linspace(start=0, stop=100, num=11)
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_explicit, deltax=0.5, method='Explicit')
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_implicit, deltax=0.1, method='Implicit')

''' Iniciando simulação para os métodos - Numérico ----------------------------------------------------------------- '''
NsimExp.FlowBoundaries(t=t_explicit, well_class=case)

''' Aferição dos resultados e comparação --------------------------------------------------------------------------- '''
root_results = r'results\Simulador_Fluxo-Fluxo'
data_for_numerical = pd.read_excel(f'{root_results}\\fluxo-fluxo_numerico_Explicit.xlsx').set_index('x')

# Plotagem de apenas algumas curvas para melhor visualização. O arquivo .xlsx completo contém a quantidade curvas
# inseridas na discretização da malha.
# time_values = np.linspace(t[0], t[-1], 11)
# Functions.plot_graphs_compare(root=root_results, arq_ana=data_for_analitical, arq_num=data_for_numerical,
#                               time=time_values)
