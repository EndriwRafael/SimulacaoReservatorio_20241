"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condições de Dirichlet nas duas Extremidades
"""
import Sim_LinearFlow_CCFlowPress as Sim_fp
import Functions
import numpy as np
import pandas as pd

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000.
pressure_well = 9000000.
length_reser = 10.
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 30.
thickness = 10.
wellflow = 0.01

''' Inicializando simuladores Pressão - Pressão -------------------------------------------------------------------- '''
case_to_sim = Sim_fp.InitializeData(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                                    res_thick=thickness, porosity=porosit, viscosity=viscosi,
                                    compressibility=compressibi, wellflow=wellflow, permeability=permeabi,
                                    res_len=length_reser)

''' Discretização da malha ----------------------------------------------------------------------------------------- '''
# Valores de discretização devem ser conferidos antes de rodar, por conta do critério de convergência. Caso os valores
# estejam incoerentes, o código retornar um erro avisando que o critério de convergência não foi respeitado!
Functions.create_mesh(well_class=case_to_sim, n_cells=0, time_values=, deltax=0.5)
t = np.linspace(start=0, stop=100, num=401)

''' Iniciando simulação para ambos os métodos - Analítico e Numérico ----------------------------------------------- '''
Sim_fp.AnaliticalAnalysis(t=t, well_class=case_to_sim)

