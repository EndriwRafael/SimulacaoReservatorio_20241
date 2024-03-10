"""
Simulador Fluxo Linear Monofásico - 1D
"""
import Simuladores as Sm
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

data_case_PressPress = Sm.SimuladorPressPressLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                    reserlength=length_reser,
                                                    permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                    compresibility=compressibi)

data_case_FlowPress = Sm.SimuladorFlowPressureLinear(initialpressure=pressure_initial, wellpressure=pressure_well,
                                                     reserlength=length_reser,
                                                     permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                     compresibility=compressibi, reserarea=area, reserthick=thickness,
                                                     wellflow=wellflow)

data_case_FlowPress_Radial = Sm.SimuladorFlowPressureRadial(initialpressure=pressure_initial,
                                                            wellpressure=pressure_well,
                                                            reserlength=length_reser,
                                                            permeability=permeabi, viscosity=viscosi, porosity=porosit,
                                                            compresibility=compressibi,
                                                            reserthick=thickness,
                                                            wellflow=wellflow)

x = np.linspace(start=0, stop=data_case_PressPress.res_length, num=1001)
t = np.linspace(start=1, stop=100, num=10)

data_case_PressPress.start_simulation(x=x, t=t)
data_case_FlowPress.start_simulation(x=x, t=t)
data_case_FlowPress_Radial.start_simulation(x=x, t=t)
