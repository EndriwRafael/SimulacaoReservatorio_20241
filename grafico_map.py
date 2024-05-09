import numpy as np
import matplotlib.pyplot as plt

solido2 = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7]
])

# Criando o mapa de cores
plt.imshow(solido2, cmap='hot', interpolation='bicubic', vmin=1, vmax=10, origin='lower',
           extent=(0, 10, 0, 1))
plt.colorbar()
plt.show()
