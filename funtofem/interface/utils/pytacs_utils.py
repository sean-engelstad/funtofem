__all__ = [
    "TacsOutputGenerator",
    "TacsPanelDimensions",
]

class TacsOutputGenerator:
    def __init__(self, prefix, name="tacs_output_file", f5=None):
        """Store information about how to write TACS output files"""
        self.count = 0
        self.prefix = prefix
        self.name = name
        self.f5 = f5

    def __call__(self):
        """Generate the output from TACS"""

        if self.f5 is not None:
            file = self.name + "%03d.f5" % (self.count)
            filename = os.path.join(self.prefix, file)
            self.f5.writeToFile(filename)
        self.count += 1
        return

    def _deallocate(self):
        """free up memory before delete"""
        self.f5.__dealloc__()


class TacsPanelDimensions:
    def __init__(self, comm, panel_length_dv_index: int, panel_width_dv_index: int):
        self.comm = comm
        self.panel_length_dv_index = panel_length_dv_index
        self.panel_width_dv_index = panel_width_dv_index

        self.panel_length_constr = None
        self.panel_width_constr = None

    def compute_panel_length(
        self, assembler, struct_vars, constr_base_name, var_base_name
    ):
        if self.panel_length_constr is not None:
            length_funcs = None
            if assembler is not None:

                # get the panel length from the TACS constraint object
                length_funcs = {}
                # clear up to date otherwise it might copy the old value
                self.panel_length_constr.externalClearUpToDate()
                self.panel_length_constr.evalConstraints(length_funcs)

            # assume rank 0 is a TACS proc (this is true as TACS uses rank 0 as root)
            length_funcs = self.comm.bcast(length_funcs, root=0)

            # update the panel length and width dimensions into the F2F variables
            # these will later be set into the TACS constitutive objects in the self.set_variables() call
            length_comp_ct = 0
            first_key = [_ for _ in length_funcs][0]
            # constraint values return actual - current (so use this to solve for actual panel length)
            for var in struct_vars:
                if var_base_name in var.name:
                    var.value += length_funcs[first_key][length_comp_ct]
                    length_comp_ct += 1
        return

    def compute_panel_width(
        self, assembler, struct_vars, constr_base_name, var_base_name
    ):
        if self.panel_width_constr is not None:
            width_funcs = None
            if assembler is not None:

                # get the panel width from the TACS constraint object
                width_funcs = {}
                self.panel_width_constr.externalClearUpToDate()
                self.panel_width_constr.evalConstraints(width_funcs)

            # assume rank 0 is a TACS proc (this is true as TACS uses rank 0 as root)
            width_funcs = self.comm.bcast(width_funcs, root=0)

            # update the panel length and width dimensions into the F2F variables
            # these will later be set into the TACS constitutive objects in the self.set_variables() call
            width_comp_ct = 0
            first_key = [_ for _ in width_funcs][0]
            # constraint values return actual - current (so use this to solve for actual panel length)
            for var in struct_vars:
                if var_base_name in var.name:
                    var.value += width_funcs[first_key][width_comp_ct]
                    width_comp_ct += 1
        return

    def setDesignVars(self, xvec):
        if self.panel_length_constr is not None:
            self.panel_length_constr.setDesignVars(xvec)
        if self.panel_width_constr is not None:
            self.panel_width_constr.setDesignVars(xvec)

    def _compute_panel_dimension_xpt_sens(self, scenario):
        """
        compute panel length and width coordinate derivatives
        for stiffened panel constitutive objects and write into the f2f variables
        """

        # TBD - haven't written this yet
        # need to multiply the panel length, width DV values by the xpt sens of the constraints
        # then add to the struct xpt sens for each function in F2F
        func_grad = self.scenario_data[scenario].func_grad

        for ifunc, func in enumerate(scenario.functions):
            for i, var in enumerate(self.struct_variables):
                pass

        # end of prototype

        grads_dict = None
        if self.assembler is not None:
            funcSens = {}
            if self.comm.rank == 0:
                grads_dict = {}
            ifunc = 0
            self.panel_length_constraint.evalConstraintsSens(funcSens)
            for func in self.model.composite_functions:
                if self.PANEL_LENGTH_CONSTR in func.name and self.comm.rank == 0:

                    grads_dict[func.full_name] = {}

                    # assume name of form f"{self.PANEL_LENGTH_CONSTR}-fnum"
                    for ivar, var in enumerate(self.struct_variables):
                        func.derivatives[var] = funcSens[self.panel_length_name][
                            "struct"
                        ].toarray()[ifunc, ivar]
                        grads_dict[func.full_name][var.full_name] = func.derivatives[
                            var
                        ]

                    ifunc += 1

        # broadcast the funcs dict to other processors
        grads_dict = self.comm.bcast(grads_dict, root=0)

        for func in self.model.composite_functions:
            if func.full_name in list(grads_dict.keys()):
                for ivar, var in enumerate(self.struct_variables):
                    func.derivatives[var] = grads_dict[func.full_name][var.full_name]

        # # compute the panel length values
        # # -----------------------------------------
        # if self.panel_length_dv_index:
        #     length_funcs = None
        #     if self.assembler is not None:

        #         # get the panel length from the TACS constraint object
        #         length_funcs = {}
        #         self.panel_length_constr.evalConstraints(length_funcs)

        #     # assume rank 0 is a TACS proc (this is true as TACS uses rank 0 as root)
        #     length_funcs = self.comm.bcast(length_funcs, root=0)

        #     # update the panel length and width dimensions into the F2F variables
        #     # these will later be set into the TACS constitutive objects in the self.set_variables() call
        #     length_comp_ct = 0
        #     for var in self.struct_variables:
        #         if self.LENGTH_VAR in var.name:
        #             var.value = length_funcs[self.LENGTH_CONSTR][length_comp_ct]
        #             length_comp_ct += 1

        # # compute the panel width values
        # # -----------------------------------------
        # if self.panel_length_dv_index:
        #     width_funcs = None
        #     if self.assembler is not None:

        #         # get the panel width from the TACS constraint object
        #         width_funcs = {}
        #         self.panel_width_constr.evalConstraints(width_funcs)

        #     # assume rank 0 is a TACS proc (this is true as TACS uses rank 0 as root)
        #     width_funcs = self.comm.bcast(width_funcs, root=0)

        #     # update the panel length and width dimensions into the F2F variables
        #     # these will later be set into the TACS constitutive objects in the self.set_variables() call
        #     width_comp_ct = 0
        #     for var in self.struct_variables:
        #         if self.WIDTH_VAR in var.name:
        #             var.value = width_funcs[self.WIDTH_CONSTR][width_comp_ct]
        #             width_comp_ct += 1

        #     return