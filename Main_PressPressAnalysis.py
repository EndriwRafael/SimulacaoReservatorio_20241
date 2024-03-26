"""
Main for Cases to Simulate Linear Flow Monofase 1D
"""
import Sim_Analiticos as Sa
import Sim_Numericos as Sn
import numpy as np
import sys

pressure_initial = 19000000
pressure_well = 9000000
length_reser = 10
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 30
thickness = 10
wellflow = 0.01

''' Inicializando simuladores Pressão - Pressão '''
data_case_PressPress_Ana = Sa.SimuladorPressPressLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                        reserlength=length_reser,
                                                        permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                        compresibility=compressibi)
data_case_PressPress_Num = Sn.SimuladorPressPressLinear(initial_pressure=pressure_initial, well_pressure=pressure_well,
                                                        res_length=length_reser, porosity=porosit,
                                                        compressibility=compressibi, viscosity=viscosi,
                                                        permeability=permeabi)

''' Discretização da malha '''
x = np.linspace(start=0, stop=data_case_PressPress_Ana.res_length, num=1001)
t = np.linspace(start=0, stop=100, num=11)

''' Iniciando simulação '''
data_case_PressPress_Ana.start_simulation(x=x, t=t)
# conv = data_case_PressPress_Num.calc_estab(n_cells=20, t=t)
# if conv > 0.25:
#     print(f'Error!!! Critério de convergência não foi atingido: conv = {conv} > 0.25')
#     sys.exit()
# print(f'Critério de convergência foi atingido: conv = {conv} < 0.25')
# data_case_PressPress_Num.start_simulation(100, t)
