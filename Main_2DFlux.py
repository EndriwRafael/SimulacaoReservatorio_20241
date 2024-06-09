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
thickness = 20.
wellflow = 0.01
injectivityflow = 0.01

''' Definindo condições de simulação ------------------------------------------------------------------------------- '''
well_boundary = 'F'
external_boundary = 'F'
top_boundary = 'F'
base_boundary = 'F'
flux = '2D'
# Dados para criação de malha - O número de células deve ser igual, para x e y.
time_explicit = np.linspace(start=0, stop=100, num=401)
time_implicit = np.linspace(start=0, stop=100, num=101)
n_cells_explicit = 40
n_cells_implicit = 1000

''' Inicializando simulador ---------------------------------------------------------------------------------------- '''
case = Functions.get_object_case(well_condiction=well_boundary, external_condiction=external_boundary,
                                 top_condiction=top_boundary, base_condiction=base_boundary, fluxtype=flux)
# Nos casos de fluxo, os fluxos devem ser passados. No caso de apenas pressão, não é necessário infomar os fluxos.
case.set_case_parameters(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                         res_thick=thickness, porosity=porosit, viscosity=viscosi,
                         compressibility=compressibi, permeability=permeabi,
                         res_len=length_reser, wellflow=wellflow)

''' Criando malhas ------------------------------------------------------------------------------------------------- '''
simulation = Functions.get_object_mesh(flow_type=flux, wellobject=case)
simulation.set_mesh_grid(time_explicit=time_explicit, time_implicit=time_implicit, n_cells_explicit=n_cells_explicit,
                         n_cells_implicit=n_cells_implicit)
