"""
Main for Cases to Simulate Linear Flow Monofase 1D - EDH _ Condições de Dirichlet nas duas Extremidades
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
area = 30.
thickness = 10.
wellflow = 0.01
injectivityflow = 0.01

''' Definindo condições de simulação ------------------------------------------------------------------------------- '''
well_boundary = 'P'
external_boundary = 'P'
flux = '1D'
time_explicit = np.linspace(start=0, stop=100, num=401)
time_implicit = np.linspace(start=0, stop=100, num=11)
n_cells_explicit = 40
n_cells_implicit = 100

''' Inicializando simulador ---------------------------------------------------------------------------------------- '''
simulation = Functions.get_object_case(well_condiction=well_boundary, external_condiction=external_boundary)
simulation.set_case_parameters(initial_press=pressure_initial, well_press=pressure_well, res_area=area,
                               res_thick=thickness, porosity=porosit, viscosity=viscosi,
                               compressibility=compressibi, permeability=permeabi,
                               res_len=length_reser)

''' Criando malhas ------------------------------------------------------------------------------------------------- '''
mesh = Functions.get_object_mesh(flow_type=flux)
mesh.set_mesh_grid(time_explicit=time_explicit, time_implicit=time_implicit, n_cells_explicit=n_cells_explicit,
                   n_cells_implicit=n_cells_implicit, wellclass=simulation)
