import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Suponha que você tenha um DataFrame com três colunas (x, y1, y2)
# Vamos criar um DataFrame de exemplo para este caso
df = pd.read_excel('results/OneDimensionalFlow/PressurePressure_Simulator/PressurePressure_Analitical.xlsx').set_index('x')
columns = [coll for coll in df.columns if coll != 0.0]


# Função para atualizar o gráfico a cada quadro da animação
def update(frame):
    print(frame)
    a = df.loc[:, columns[frame]]
    pressure = []
    for _, j in enumerate(a):
        pressure.append(j)
    plt.cla()  # Limpa o eixo atual para atualizar o gráfico
    plt.plot(df.index, pressure, label=columns[frame])  # Plotar cada coluna y
    plt.ticklabel_format(axis='y', style='plain')
    plt.xlabel('Comprimento (m)')
    plt.ylabel('Pressão (Pa)')
    plt.title("Comparação das Curvas (Analítico, Explicito e Implicito")
    ax.legend(framealpha=1)
    ax.grid()
    plt.tight_layout()


# Configuração do gráfico
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(df.columns)-1, interval=1000)  # Intervalo de 1000ms entre frames

# Salvar a animação como GIF
ani.save('animacao_dataframe.gif', writer='pillow', fps=3)  # 1 frame por segundo
