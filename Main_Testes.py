"""
Main for Cases to Simulate Linear Flow Monofase 1D
"""
import Sim_LinearFlow_CCPresPres as Sim_pp
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
data_case_PressPress_Ana = Sim_pp.InitializeData(initialpressure=pressure_initial,
                                                 wellpressure=pressure_well, reserlength=length_reser,
                                                 permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                 compresibility=compressibi)

''' Discretização da malha '''
data_case_PressPress_Ana.create_mesh(n_cells=0, deltax=.01)
t = np.linspace(start=0, stop=100, num=11)

# Sim_pp.AnaliticalAnalysis(grid=data_case_PressPress_Ana.mesh, t=t, eta=data_case_PressPress_Ana.eta,
#                           res_length=data_case_PressPress_Ana.res_length,
#                           delta_press=data_case_PressPress_Ana.deltaPressure,
#                           well_pressure=data_case_PressPress_Ana.well_pressure,
#                           initial_pressure=data_case_PressPress_Ana.initial_pressure)

Sim_pp.NumericalAnalysis(grid=data_case_PressPress_Ana.mesh, t=t, eta=data_case_PressPress_Ana.eta,
                         well_pressure=data_case_PressPress_Ana.well_pressure,
                         initial_pressure=data_case_PressPress_Ana.initial_pressure)

