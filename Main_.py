"""
Simulador Fluxo Linear Monofásico - 1D
"""
import matplotlib.pyplot as plt
import numpy as np


class Simulador:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = reserlength
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.calc()

    def calc(self):
        delta_pressure = self.initial_pressure - self.well_pressure


pressure_initial = 100
pressure_well = 50
length_reser = 500
permeabi = 200
porosit = 0.2
viscosi = .5
compressibi = 10e-6

Simulador(initialpressure=pressure_initial, wellpressure=pressure_well, reserlength=length_reser,
          permeability=permeabi, viscosity=viscosi, porosity=porosit, compresibility=compressibi)