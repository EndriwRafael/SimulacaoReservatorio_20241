"""
Simulador Fluxo Linear Monofásico - 1D
"""
import Simuladores as sm
import numpy as np

pressure_initial = 20000
pressure_well = 18000
length_reser = 5000
permeabi = 200
porosit = 0.2
viscosi = .005
compressibi = 10e-5

data_case = sm.SimuladorPressPress(initialpressure=pressure_initial, wellpressure=pressure_well,
                                   reserlength=length_reser,
                                   permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                   compresibility=compressibi)

x = np.linspace(start=0, stop=data_case.res_length, num=100)
t = np.linspace(start=0, stop=10, num=11)
data_case.start_simulation(x=x, t=t, deltapressure=data_case.deltaPressure, wellpressure=data_case.well_pressure)
