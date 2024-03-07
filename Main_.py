"""
Simulador Fluxo Linear Monofásico - 1D
"""
import matplotlib.pyplot as plt
import numpy as np
from Simuladores import *


pressure_initial = 100
pressure_well = 50
length_reser = 500
permeabi = 200
porosit = 0.2
viscosi = .5
compressibi = 10e-6

SimuladorPressPress(initialpressure=pressure_initial, wellpressure=pressure_well, reserlength=length_reser,
                    permeability=permeabi, viscosity=viscosi, porosity=porosit, compresibility=compressibi)
