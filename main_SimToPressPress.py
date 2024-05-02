"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condições de Dirichlet nas duas Extremidades
"""
import Objects_Cases as Case
from Simuladores_LinearFlow import Analitical as Asim
from Simuladores_LinearFlow import Numerical_EXPLICIT as NsimExp
from Simuladores_LinearFlow import Numerical_IMPLICIT as NsimEmp
import Functions
import numpy as np
import pandas as pd

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000
pressure_well = 9000000
length_reser = 20
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 30
thickness = 10
wellflow = 0.01

''' Inicializando simuladores Pressão - Pressão -------------------------------------------------------------------- '''
case = Case.PressureBoundaries(initialpressure=pressure_initial, wellpressure=pressure_well,
                               reserlength=length_reser, permeability=permeabi, viscosity=viscosi,
                               porosity=porosit, compresibility=compressibi)

''' Discretização da malha ----------------------------------------------------------------------------------------- '''
# Valores de discretização devem ser conferidos antes de rodar, por conta do critério de convergência. Caso os valores
# estejam incoerentes, o código retornar um erro avisando que o critério de convergência não foi respeitado!
t_explicit = np.linspace(start=0, stop=100, num=401)
t_analitical, t_implicit = np.linspace(start=0, stop=100, num=11), np.linspace(start=0, stop=100, num=11)
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_analitical, deltax=0.01, method='Analitical')
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_explicit, deltax=0.5, method='Explicit')
Functions.create_mesh(well_class=case, n_cells=0, time_values=t_implicit, deltax=0.09, method='Implicit')

''' Iniciando simulação para ambos os métodos - Analítico e Numérico ----------------------------------------------- '''
# Para a analítica e a numérica explicita, será usado a mesma discretização de malha!
Asim.PressureBoundaries(t=t_analitical, well_class=case)  # Solução Analítica
NsimExp.PressureBoundaries(t=t_explicit, well_class=case)  # Solução Numérica Explicita

# Para a numérica implicita, a discretização será diferente, pois não possui critério de convergência!
NsimEmp.PressureBoundaries(t=t_implicit, well_class=case)  # Solução Numérica Implicita

''' Aferição dos resultados e comparação --------------------------------------------------------------------------- '''
root_results = r'results\Simulador_Pressao-Pressao'

# Plotagem de apenas algumas curvas para melhor visualização. O arquivo .xlsx completo contém a quantidade curvas
# inseridas na discretização da malha.
time_values = np.linspace(10, 100, 4)
Functions.plot_graphs_compare(root=root_results, dataclass=case,
                              time=time_values)
