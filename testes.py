import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Crie um DataFrame de exemplo
data = {
    'coluna1': [1, 2, 3, 4, 5],
    'coluna2': [2, 3, 4, 5, 6],
    'coluna3': [3, 4, 5, 6, 7]
}
df = pd.DataFrame(data)

# Crie a figura e o eixo
fig, ax = plt.subplots()

# Inicialize o gráfico vazio
lines = [ax.plot([], [], label=coluna)[0] for coluna in df.columns]

# Defina a função de inicialização
def init():
    for line in lines:
        line.set_data([], [])
    ax.set_xlim(0, len(df.index) - 1)
    ax.set_ylim(df.min().min(), df.max().max())
    ax.legend()
    return lines

# Defina a função de animação
def animate(i):
    for line, coluna in zip(lines, df.columns):
        line.set_data(df.index[:i+1], df[coluna][:i+1])
    return lines

# Crie a animação
ani = FuncAnimation(fig, animate, frames=len(df), init_func=init, blit=True)

# Salve a animação em um arquivo MP4 usando o escritor de filmes ffmpeg
ani.save('animacao.gif', fps=2, writer='ffmpeg')

plt.show()
