import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Dados de exemplo
x = np.linspace(0, 10, 100)

# Inicialização do gráfico
fig, ax = plt.subplots()

# Laço iterativo para plotar as curvas
for i in range(5):
    y1 = np.sin(x) + i
    y2 = np.cos(x) + i

    # Plot da curva contínua
    ax.plot(x, y1, '-', color='C'+str(i), label='_nolegend_')  # Usando label='_nolegend_' para evitar a adição automática à legenda
    # Plot da curva tracejada
    ax.plot(x, y2, '--', color='C'+str(i), label='_nolegend_')

# Criação manual das entradas da legenda
legend_elements = [Line2D([0], [0], linestyle='-', color='black', label='Linear'),
                   Line2D([0], [0], linestyle='--', color='black', label='Não linear')]

# Adicionando a legenda
ax.legend(handles=legend_elements)

# Exibindo o gráfico
plt.show()
