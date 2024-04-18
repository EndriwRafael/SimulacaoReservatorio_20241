import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Crie DataFrames de exemplo
data1 = {
    'coluna1': [1, 2, 3, 4, 5, 6],
    'coluna2': [2, 3, 4, 5, 6, 7],
    'coluna3': [3, 4, 5, 6, 7, 8]
}
data2 = {
    'coluna1': [5, 4, 3, 2, 1],
    'coluna2': [6, 5, 4, 3, 2],
    'coluna3': [7, 6, 5, 4, 3]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Crie a figura e o eixo
fig, ax = plt.subplots()

# Inicialize o gráfico vazio
lines1 = [ax.plot([], [], label=coluna, color='blue')[0] for coluna in df1.columns]
lines2 = [ax.plot([], [], label=coluna, color='red')[0] for coluna in df2.columns]

# Defina a função de inicialização
def init():
    for line in lines1 + lines2:
        line.set_data([], [])
    ax.set_xlim(0, len(df1.index) - 1)
    ax.set_ylim(min(df1.min().min(), df2.min().min()), max(df1.max().max(), df2.max().max()))
    ax.legend()
    return lines1 + lines2

# Defina a função de animação
def animate(i):
    for line, coluna in zip(lines1, df1.columns):
        line.set_data(df1.index[:i+1], df1[coluna][:i+1])
    for line, coluna in zip(lines2, df2.columns):
        line.set_data(df2.index[:i+1], df2[coluna][:i+1])
    return lines1 + lines2

# Crie a animação
ani = FuncAnimation(fig, animate, frames=len(df1), init_func=init, blit=True)

# Salve a animação em um arquivo MP4 usando o escritor de filmes ffmpeg
ani.save('animacao.gif', fps=2, writer='imagemagick')

plt.show()
