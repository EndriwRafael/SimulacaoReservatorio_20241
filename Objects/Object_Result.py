class ResultsOneDimFlow:
    def __init__(self, data_analitical, data_explicit, data_implicit, data_ana_exp, data_ana_imp, n_cell_explicit,
                 n_cell_implicit, dx_explicit, dx_implicit, t_process_explicit, t_process_implicit, mesh_explicit,
                 mesh_implicit, time_mesh_explicit, time_mesh_implicit):
        self.analitical = data_analitical
        self.explicit = data_explicit
        self.implicit = data_implicit
        self.analitical_explicit = data_ana_exp
        self.analitical_implicit = data_ana_imp
        self.ncell_explicit = n_cell_explicit
        self.ncell_implicit = n_cell_implicit
        self.dx_explicit = dx_explicit
        self.dx_implicit = dx_implicit
        self.time_explicit = time_mesh_explicit
        self.time_implicit = time_mesh_implicit
        self.mesh_explicit = mesh_explicit
        self.mesh_implicit = mesh_implicit
        self.timeprocess_explicit = t_process_explicit
        self.timeprocess_implicit = t_process_implicit
