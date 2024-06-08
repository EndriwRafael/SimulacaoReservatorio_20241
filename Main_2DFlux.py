"""
Main script to simulate Bilinear (2D) flow Monofase - Heterogeneous and homogeneous environments
"""
import Functions
import numpy as np

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000.
pressure_well = 9000000.
length_reser = 20.
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 200.
thickness = 10.
wellflow = 0.01
injectivityflow = 0.01

''' Definindo condições de simulação ------------------------------------------------------------------------------- '''
well_boundary = 'F'
external_boundary = 'F'
top_boundary = 'F'
base_boundary = 'F'
flux = '2D'

''' Inicializando simulador ---------------------------------------------------------------------------------------- '''
case = Functions.get_object_case(well_condiction=well_boundary, external_condiction=external_boundary,
                                 top_condiction=top_boundary, base_condiction=base_boundary, fluxtype=flux)
