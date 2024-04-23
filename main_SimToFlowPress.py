"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condição de Neumann no Poço e de Dirichlet na Extremidade da
Fronteira
"""
import Objects_Cases as Case
from Simuladores_LinearFlow import Analitical as Asim
from Simuladores_LinearFlow import Numerical_EXPLICIT as NsimExp
from Simuladores_LinearFlow import Numerical_IMPLICIT as NsimEmp
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
case = Case.FlowPressureBoundaries(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                                   res_thick=thickness, porosity=porosit, viscosity=viscosi,
                                   compressibility=compressibi, wellflow=wellflow, permeability=permeabi,
                                   res_len=length_reser)

''' Discretização da malha ----------------------------------------------------------------------------------------- '''
# Valores de discretização devem ser conferidos antes de rodar, por conta do critério de convergência. Caso os valores
# estejam incoerentes, o código retornar um erro avisando que o critério de convergência não foi respeitado!
t_explicit = np.linspace(start=0, stop=100, num=401)
t_analitical, t_implicit = np.linspace(start=0, stop=100, num=11), np.linspace(start=0, stop=100, num=111)
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_analitical, deltax=0.1, method='Analitical')
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_explicit, deltax=0.5, method='Explicit')
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_implicit, deltax=0.1, method='Implicit')

''' Iniciando simulação para ambos os métodos - Analítico e Numérico ----------------------------------------------- '''
# Para a analítica e a numérica explicita, será usado a mesma discretização de malha!
Asim.WellFlowAndPressureBoundaries(t=t_analitical, well_class=case)  # Solução Analítica
NsimExp.WellFlowAndPressureBoundaries(t=t_explicit, well_class=case)  # Solução Numérica Explicita

# Para a numérica implicita, a discretização será diferente, pois não possui critério de convergência!
NsimEmp.WellFlowAndPressureBoundaries(t=t_implicit, well_class=case)  # Solução Numérica Implicita

''' Aferição dos resultados e comparação --------------------------------------------------------------------------- '''
root_results = r'results\Simulador_Fluxo-Pressao'

# Plotagem de apenas algumas curvas para melhor visualização. O arquivo .xlsx completo contém a quantidade curvas
# inseridas na discretização da malha.
time_values = np.linspace(10, 100, 4)
Functions.plot_graphs_compare(root=root_results, dataclass=case,
                              time=time_values)

# time_values = np.linspace(0, 100, 11)
# Functions.plot_animation_compare(root=root_results, dataclass=case, time=time_values)

time_values = [20., 50., 80.]
Functions.calc_erro(root=root_results, dataclass=case,
                    time=time_values)
