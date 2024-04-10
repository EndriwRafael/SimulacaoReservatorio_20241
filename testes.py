import Functions

n = 10
r = 1
eta = 0.24


param = {
    'a': 1 + (4 * r * eta),
    'b': - (4/3) * r * eta,
    'c': - r * eta,
    'd': 1 + (2 * r * eta),
    'f1': (8/3) * r * eta * 90000,
    'fn': (8/3) * r * eta * 190000
}

pressure_field, const = Functions.create_fieldpressure(n_cells=n, param_values=param)
print(pressure_field)
