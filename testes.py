import numpy as np

# Defina o lado esquerdo das equações (coeficientes das variáveis)
left_side = np.array([[3, 2, -1],
                      [2, -2, 4],
                      [-1, 0.5, -1]])

# Defina o lado direito das equações (resultados das equações)
right_side = np.array([1, -2, 0])

# Resolva o sistema linear
solution = np.linalg.solve(left_side, right_side)

# Exiba os valores das variáveis
x, y, z = solution
print(f"Solução: x = {x}, y = {y}, z = {z}")
