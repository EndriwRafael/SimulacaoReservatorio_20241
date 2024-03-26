"""
Simulador Fluxo Linear Monofásico - 1D
"""
import Sim_Analiticos as Sa
import Sim_Numericos as Sn
import numpy as np

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
data_case_PressPress = Sa.SimuladorPressPressLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                    reserlength=length_reser,
                                                    permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                    compresibility=compressibi)
data_case_PressPress_Num = Sn.SimuladorPressPressLinear(initial_pressure=pressure_initial, well_pressure=pressure_well,
                                                        res_length=length_reser, porosity=porosit,
                                                        compressibility=compressibi, viscosity=viscosi,
                                                        permeability=permeabi)

''' Inicializando simuladores Fluxo - Pressão '''
data_case_FlowPress = Sa.SimuladorFlowPressureLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                     reserlength=length_reser,
                                                     permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                     compresibility=compressibi, reserarea=area, reserthick=thickness,
                                                     wellflow=wellflow)
data_case_FlowPress_Num = Sn.SimuladorFlowPressLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                      reserlength=length_reser,
                                                      permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                      compresibility=compressibi, reserarea=area, reserthick=thickness,
                                                      wellflow=wellflow)

x = np.linspace(start=0, stop=data_case_PressPress.res_length, num=1001)
t = np.linspace(start=0, stop=100, num=10)

# data_case_PressPress.start_simulation(x=x, t=t)
# conv = data_case_PressPress_Num.calc_estab(n_cells=40, t=t)
# data_case_PressPress_Num.start_simulation(100, t)

conv = data_case_FlowPress_Num.calc_estab(n_cells=40, t=t)
data_case_FlowPress_Num.start_simulation(cell_number=100, t=t)
# data_case_FlowPress.start_simulation(x=x, t=t)

