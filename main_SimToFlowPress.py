"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condição de Neumann no Poço e de Dirichlet na Extremidade da
Fronteira
"""
import Objects_Cases as Case
from Simuladores_LinearFlow import Analitical as Asim
from Simuladores_LinearFlow import Numerical_EXPLICIT as NsimExp
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

''' Inicializando simuladores Fluxo - Pressão -------------------------------------------------------------------- '''
case_to_sim = Case.FlowPressureBoundaries(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                                          res_thick=thickness, porosity=porosit, viscosity=viscosi,
                                          compressibility=compressibi, wellflow=wellflow, permeability=permeabi,
                                          res_len=length_reser)

''' Discretização da malha ----------------------------------------------------------------------------------------- '''
# Valores de discretização devem ser conferidos antes de rodar, por conta do critério de convergência. Caso os valores
# estejam incoerentes, o código retornar um erro avisando que o critério de convergência não foi respeitado!
t = np.linspace(start=0, stop=100, num=401)
Functions.create_mesh(well_class=case_to_sim, n_cells=0, time_values=t, deltax=0.5)

''' Iniciando simulação para ambos os métodos - Analítico e Numérico ----------------------------------------------- '''
Asim.WellFlowAndPressureBoundaries(t=t, well_class=case_to_sim)

NsimExp.WellFlowAndPressureBoundaries(t=t, well_class=case_to_sim)

''' Aferição dos resultados e comparação --------------------------------------------------------------------------- '''
root_results = r'results\Simulador_Fluxo-Pressao'
data_for_analitical = pd.read_excel(f'{root_results}\\fluxo-pressao_analitico_Explicit.xlsx').set_index('x')
data_for_numerical = pd.read_excel(f'{root_results}\\fluxo-pressao_numerico_Explicit.xlsx').set_index('x')

# Plotagem de apenas algumas curvas para melhor visualização. O arquivo .xlsx completo contém a quantidade curvas
# inseridas na discretização da malha.
time_values = np.linspace(t[0], t[-1], 11)
Functions.plot_graphs_compare(root=root_results, arq_ana=data_for_analitical, arq_num=data_for_numerical,
                              time=time_values)
