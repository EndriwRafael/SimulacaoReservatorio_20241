import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PressureBoundaries:
    def __init__(self, initialpressure, wellpressure, reserlength, permeability, viscosity, porosity, compresibility):
        self.initial_pressure = initialpressure
        self.well_pressure = wellpressure
        self.res_length = int(reserlength)
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compresibility
        self.eta, self.deltaPressure = self.calc()

    def calc(self) -> tuple[float, float]:
        delta_pressure = self.initial_pressure - self.well_pressure
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta, delta_pressure


class FlowBoundaries:
    def __init__(self, initial_press: float, well_press: float, res_len: float, permeability: float, viscosity: float,
                 porosity: float, compressibility: float, res_area: float, res_thick: float or int, wellflow: float,
                 injectflow: float):
        self.initial_pressure = initial_press
        self.well_pressure = well_press
        self.res_length = res_len
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compressibility
        self.res_area = res_area
        self.res_thickness = res_thick
        self.well_flow = wellflow
        self.injectflow = injectflow
        self.eta = self.calc_eta()

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta


class FlowPressureBoundaries:
    def __init__(self, initial_press: float, well_press: float, res_len: float, permeability: float, viscosity: float,
                 porosity: float, compressibility: float, res_area: float, res_thick: float or int, wellflow: float):
        self.initial_pressure = initial_press
        self.well_pressure = well_press
        self.res_length = res_len
        self.permeability = permeability
        self.viscosity = viscosity
        self.porosity = porosity
        self.compressibility = compressibility
        self.res_area = res_area
        self.res_thickness = res_thick
        self.well_flow = wellflow
        self.eta = self.calc_eta()

    def calc_eta(self) -> float:
        eta = self.permeability / (self.viscosity * self.compressibility * self.porosity)
        return eta


class MeshCases:
    def __init__(self, grid: dict, cells: int, method: str, well_method: object):
        self.mesh_to_plot = grid
        self.n_cells = cells
        self.method_name = method
        self.wellclass = well_method
        self.create_mesh()

    def create_mesh(self):
        if self.method_name == "Explicit":
            self.wellclass.explicit_mesh = self.mesh_to_plot
            self.wellclass.n_cells_explicit = self.n_cells
        elif self.method_name == "Implicit":
            self.wellclass.implicit_mesh = self.mesh_to_plot
            self.wellclass.n_cells_implicit = self.n_cells
        else:
            self.wellclass.analitical_mesh = self.mesh_to_plot
            self.wellclass.n_cells_analitical = self.n_cells
        pass


class Animation:
    def __init__(self, root: str, dict_frames: dict, t_values: np.ndarray, limits: list):
        self.root_results = root
        self.data = dict_frames
        self.t_values = t_values
        self.xlims, self.ylims = limits[0], limits[1]
        self.lines = []
        self.fig, self.ax = plt.subplots()
        self.start_animation()

    def init_anim(self):
        if len(self.lines) == 2:  ## Acho que só dois for já resolvia!!!
            for line in self.lines[0] + self.lines[1]:
                line.set_data([], [])
            self.ax.set_xlim(self.xlims)
            self.ax.set_ylim(self.ylims)
            plt.ticklabel_format(axis='y', style='plain')
            plt.xlabel('Comprimento (m)')
            plt.ylabel('Pressão (psia)')
            plt.title('Comparação das Curvas (Analítico e Numérico)')
            # self.ax.legend(framealpha=1, handles=legend_elements)
            self.ax.grid()
            plt.tight_layout()
            return self.lines[0] + self.lines[1]
        else:
            for line in self.lines[0] + self.lines[1] + self.lines[2]:
                line.set_data([], [])
            self.ax.set_xlim(self.xlims)
            self.ax.set_ylim(self.ylims)
            plt.ticklabel_format(axis='y', style='plain')
            plt.xlabel('Comprimento (m)')
            plt.ylabel('Pressão (psia)')
            plt.title('Comparação das Curvas (Analítico e Numérico)')
            # self.ax.legend(framealpha=1, handles=legend_elements)
            self.ax.grid()
            plt.tight_layout()
            return self.lines[0] + self.lines[1] + self.lines[2]

    def animate(self, i):
        # list_dict = list(self.data.keys())
        # for i in range(len(self.lines)):
        #     for line, column in zip(self.lines[i], self.data[list_dict[i]]):
        #         if column in self.t_values:
        #             line.set_data(self.data[list_dict[i]].index[:framedata+1], self.data[list_dict[i]][column][:framedata+1])
        # return self.lines
        if len(self.lines) == 2:
            for line, coluna in zip(self.lines[0], self.data['dataframe_to_explicit'].columns):
                if coluna in self.t_values:
                    line.set_data(self.data['dataframe_to_explicit'].index[:i + 1],
                                  self.data['dataframe_to_explicit'][coluna][:i + 1])

            for line, coluna in zip(self.lines[1], self.data['dataframe_to_implicit'].columns):
                if coluna in self.t_values:
                    line.set_data(self.data['dataframe_to_implicit'].index[:i + 1],
                                  self.data['dataframe_to_implicit'][coluna][:i + 1])
            return self.lines[0] + self.lines[1]
        else:
            for line, coluna in zip(self.lines[0], self.data['dataframe_to_analitical'].columns):
                if coluna in self.t_values:
                    line.set_data(self.data['dataframe_to_analitical'].index[:i + 1],
                                  self.data['dataframe_to_analitical'][coluna][:i + 1])

            for line, coluna in zip(self.lines[1], self.data['dataframe_to_explicit'].columns):
                if coluna in self.t_values:
                    line.set_data(self.data['dataframe_to_explicit'].index[:i + 1],
                                  self.data['dataframe_to_explicit'][coluna][:i + 1])

            for line, coluna in zip(self.lines[2], self.data['dataframe_to_implicit'].columns):
                if coluna in self.t_values:
                    line.set_data(self.data['dataframe_to_implicit'].index[:i + 1],
                                  self.data['dataframe_to_implicit'][coluna][:i + 1])

            return self.lines[0] + self.lines[1] + self.lines[2]

    def start_animation(self):

        try:
            self.lines.append([self.ax.plot([], [])[0]
                               for column in self.data['dataframe_to_analitical'].columns
                               if column in self.t_values])
            self.lines.append([self.ax.plot([], [], marker='o')[0]
                               for column in self.data['dataframe_to_explicit'].columns
                               if column in self.t_values])
            self.lines.append([self.ax.plot([], [], marker='^')[0]
                               for column in self.data['dataframe_to_implicit'].columns
                               if column in self.t_values])
        except KeyError:
            print('Para a simulação de fluxo nas duas extremidades, não existe resultado analítico!')
            self.lines.append([self.ax.plot([], [], marker='o')[0]
                               for column in self.data['dataframe_to_explicit'].columns
                               if column in self.t_values])
            self.lines.append([self.ax.plot([], [], marker='^')[0]
                               for column in self.data['dataframe_to_implicit'].columns
                               if column in self.t_values])

        animation = FuncAnimation(fig=self.fig, func=self.animate, frames=len(self.data['dataframe_to_explicit']),
                                  init_func=self.init_anim, blit=False)
        plt.show()
        animation.save(f'{self.root_results}\\Animation_results.gif', fps=2, writer='ffmpeg')
