"""
Main script to simulate Bilinear (2D) flow Monofase - Heterogeneous and homogeneous environments
"""
import Functions
import numpy as np

''' Dados de Entrada ----------------------------------------------------------------------------------------------- '''
pressure_initial = 19000000.
length_reser_x = 20.
length_reser_y = 20
porosit = 0.2
viscosi = .001
compressibi = 2.04e-9
area = 200.
thickness = 20.
qleft = 0
qtop = 0
qbase = 0
qright = 0
permeability_map = 'PermeabilityMaps/Permeability_Map_20x20_Hetero_Last.txt'
wells_configuration = {
    'well 1': {'Position': (7, 14), 'Radius': 0.1, 'Pressure': 9000000., 'Flow': None,
               'Type': 'Production'},
    'well 2': {'Position': (17, 6), 'Radius': 0.1, 'Pressure': 9000000., 'Flow': None,
               'Type': 'Production'}
    # 'well 1': {'Position': (3, 3), 'Radius': 0.1, 'Pressure': 9000000., 'Flow': None,
    #            'Type': 'Production'},
    # 'well 2': {'Position': (2, 3), 'Radius': 0.1, 'Pressure': 9000000., 'Flow': None,
    #            'Type': 'Production'}
}

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

# Dados para criação de malha - O número de células deve ser igual, para x e y.
time_implicit = np.linspace(start=0, stop=20000, num=301)
n_cells_implicit = 20

''' Inicializando simulador ---------------------------------------------------------------------------------------- '''
case = Functions.get_object_case(top_condiction=top_boundary, base_condiction=base_boundary, fluxtype=flux,
                                 left_condition=left_boundary, right_condition=right_boundary)

case.set_case_parameters(initial_press=pressure_initial, res_area=area, res_thick=thickness,
                         porosity=porosit, viscosity=viscosi, compressibility=compressibi,
                         permeability=permeability_map, res_len=length_reser_x, res_width=length_reser_y,
                         baseflow=qbase, leftflow=qleft, rightflow=qright, topflow=qtop,
                         well_position=wells_configuration)

''' Criando malhas ------------------------------------------------------------------------------------------------- '''
simulation = Functions.get_object_mesh(flow_type=flux, wellobject=case)
simulation.set_mesh_grid(time_explicit=None, time_implicit=time_implicit, n_cells_explicit=None,
                         n_cells_implicit=n_cells_implicit)

''' Iniciando simulação -------------------------------------------------------------------------------------------- '''
simulation.simulate()

''' Aferição dos resultados e comparação --------------------------------------------------------------------------- '''
case.compute()
