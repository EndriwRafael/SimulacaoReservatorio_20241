"""
Main script to simulate Bilinear (2D) flow Monofase - Heterogeneous and homogeneous environments
"""
import pandas as pd

import Functions
import numpy as np

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000.
pressure_well = 9000000.
length_reser_x = 30.
length_reser_y = 50
permeabi = 9.87e-14
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 200.
thickness = 20.
wellflow = 0.01
injectivityflow = 0.01
qleft = 0
qtop = 0
qbase = 0
qright = 0
permeability_map = 'PermeabilityMaps/Permeability_Map_10x10.txt'

''' Definindo condições de simulação ------------------------------------------------------------------------------- '''
# Para fluxo 1D:
well_boundary = 'P'
external_boundary = 'F'
# Para fluxo 2D:
left_boundary = 'F'
right_boundary = 'F'
top_boundary = 'F'
base_boundary = 'F'

flux = '2D'

well_positions = {
    'well 1': {'Position': (35, 60), 'Type': 'Production'}
}

# Dados para criação de malha - O número de células deve ser igual, para x e y.
time_implicit = np.linspace(start=0, stop=100, num=101)
n_cells_implicit = 10

''' Inicializando simulador ---------------------------------------------------------------------------------------- '''
case = Functions.get_object_case(top_condiction=top_boundary, base_condiction=base_boundary, fluxtype=flux,
                                 left_condition=left_boundary, right_condition=right_boundary)

case.set_case_parameters(initial_press=pressure_initial, well_press=pressure_well, res_area=area, res_thick=thickness,
                         porosity=porosit, viscosity=viscosi, compressibility=compressibi,
                         permeability=permeability_map, res_len=length_reser_x, res_width=length_reser_y,
                         baseflow=qbase, leftflow=qleft, rightflow=qright, topflow=qtop, well_position=well_positions)

''' Criando malhas ------------------------------------------------------------------------------------------------- '''
simulation = Functions.get_object_mesh(flow_type=flux, wellobject=case)
simulation.set_mesh_grid(time_explicit=None, time_implicit=time_implicit, n_cells_explicit=None,
                         n_cells_implicit=n_cells_implicit)

''' Iniciando simulação -------------------------------------------------------------------------------------------- '''
simulation.simulate()
