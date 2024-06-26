from __future__ import print_function

__all__ = ["FuntofemComponent"]

import warnings

warnings.filterwarnings("ignore")

import os, matplotlib.pyplot as plt, numpy as np
from openmdao.api import ExplicitComponent


class FuntofemComponent(ExplicitComponent):
    """
    OpenMDAO component for funtofem design optimization
    """

    def register_to_model(self, openmdao_model, subsystem_name):
        """
        register the funtofem variables and functions
        to the openmdao problem (called in your run script)
        see examples/naca_wing/1_tacs_sizing_opt.py
        """
        driver = self.options["driver"]
        model = driver.model

        # add design variables to the model
        for var in model.get_variables():
            openmdao_model.add_design_var(
                f"{subsystem_name}.{var.full_name}",
                lower=var.lower,
                upper=var.upper,
                scaler=var.scale,
            )

        # add objectives & constraints to the model
        for func in model.get_functions(optim=True):
            if func._objective:
                openmdao_model.add_objective(
                    f"{subsystem_name}.{func.full_name}", scaler=func.scale
                )
            else:
                openmdao_model.add_constraint(
                    f"{subsystem_name}.{func.full_name}",
                    lower=func.lower,
                    upper=func.upper,
                    scaler=func.scale,
                )

    def initialize(self):
        self.options.declare("driver", types=object)
        self.options.declare("track_history", types=bool, default=True)
        self.options.declare(
            "write_dir", default=None
        )  # where to write design and opt plot files
        self.options.declare("design_out_file", types=str)

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)
        driver = self.options["driver"]
        track_history = self.options["track_history"]
        write_dir = self.options["write_dir"]
        model = driver.model

        # get vars and funcs, only funcs with optim=True
        # which are for optimization
        variables = model.get_variables()
        assert len(variables) > 0

        # add f2f variables to openmdao
        for var in variables:
            self.add_input(var.full_name, val=var.value)

        # add f2f functions to openmdao
        functions = model.get_functions(optim=True)
        for func in functions:
            self.add_output(func.full_name)

        # store the variable dictionary of values
        # to prevent repeat analyses
        self._x_dict = {var.full_name: var.value for var in model.get_variables()}
        comm = driver.comm
        if write_dir is None:
            write_dir = os.getcwd()
        self._write_path = write_dir
        self._plot_filename = f"f2f-{model.name}_optimization.png"

        self._first_analysis = True
        self._first_opt = True
        self._iteration = 0
        self._prev_forward_changed_design = None

        # store function optimization history
        if track_history:
            self._func_history = {
                func.plot_name: [] for func in functions if func._plot
            }

            if comm.rank == 0:
                self._design_hdl = open(
                    os.path.join(self._write_path, "design_hist.txt"), "w"
                )

        # add all variables (for off-scenario variables to derivatives dict for each function) to analysis functions
        for func in model.get_functions():
            for var in model.get_variables():
                func.derivatives[var] = 0.0

    def setup_partials(self):
        driver = self.options["driver"]
        model = driver.model

        # declare any partial derivatives for optimization functions
        for func in model.get_functions(optim=True):
            for var in model.get_variables():
                self.declare_partials(func.full_name, var.full_name)

    def update_design(self, inputs, analysis=True):
        driver = self.options["driver"]
        model = driver.model
        changed_design = False

        if analysis:  # forward analysis check new design
            for var in model.get_variables():
                if var.value != float(inputs[var.full_name]):
                    changed_design = True
                    var.value = float(inputs[var.full_name])

            if self._first_analysis:
                self._first_analysis = False
                changed_design = True

            if changed_design:
                self._iteration += 1
                self._design_report()
            self._prev_forward_changed_design = changed_design

        else:  # adjoint runs if forward previously ran
            changed_design = self._prev_forward_changed_design
        return changed_design

    def compute(self, inputs, outputs):
        driver = self.options["driver"]
        track_history = self.options["track_history"]
        model = driver.model
        design_out_file = self.options["design_out_file"]
        self.update_design(inputs, analysis=True)

        model.write_design_variables_file(driver.comm, design_out_file)
        driver.solve_forward()
        model.evaluate_composite_functions(compute_grad=False)

        if track_history:
            self._update_history()

        for func in model.get_functions(optim=True):
            outputs[func.full_name] = func.value.real
        return

    def compute_partials(self, inputs, partials):
        driver = self.options["driver"]
        model = driver.model
        self.update_design(inputs, analysis=False)

        driver.solve_adjoint()
        model.evaluate_composite_functions(compute_grad=True)

        for func in model.get_functions(optim=True):
            for var in model.get_variables():
                partials[func.full_name, var.full_name] = func.get_gradient_component(
                    var
                ).real
        return

    def cleanup(self):
        """close the design handle file and any other cleanup"""
        track_history = self.options["track_history"]
        if track_history:
            self._design_hdl.close()

    # helper methods for writing history, plotting history, etc.
    def _update_history(self):
        driver = self.options["driver"]
        model = driver.model
        for func in model.get_functions(optim=True):
            if func.plot_name in self._func_history:
                self._func_history[func.plot_name].append(func.value.real)

        if driver.comm.rank == 0:
            self._plot_history()
            self._function_report()

    def _function_report(self):
        driver = self.options["driver"]

        if driver.comm.rank == 0:
            self._design_hdl.write(f"Analysis #{self._iteration}:\n")
            for func_name in self._func_history:
                self._design_hdl.write(
                    f"\tfunc {func_name} = {self._func_history[func_name][-1]}\n"
                )
            self._design_hdl.write("\n")
            self._design_hdl.flush()

    def _plot_history(self):
        driver = self.options["driver"]
        model = driver.model

        if driver.comm.rank == 0:
            func_keys = list(self._func_history.keys())
            num_iterations = len(self._func_history[func_keys[0]])
            iterations = [_ for _ in range(num_iterations)]
            fig = plt.figure()
            ax = plt.subplot(111)
            nkeys = len(func_keys)
            ind = 0
            colors = plt.cm.jet(np.linspace(0, 1, nkeys))
            for func in model.get_functions(optim=True):
                if func.plot_name in func_keys:
                    yvec = np.array(self._func_history[func.plot_name])
                    if func._objective:
                        yvec *= func.scale
                    else:  # constraint
                        constr_bndry = 1.0
                        # take relative errors against constraint boundaries, lower upper
                        yfinal = yvec[-1]
                        err_lower = 1e5
                        err_upper = 1e5
                        if func.lower is not None:
                            # use abs error since could have div 0
                            err_lower = abs(yfinal - func.lower)
                        if func.upper is not None:
                            # use abs error since could have div 0
                            err_upper = abs(yfinal - func.upper)
                        if err_lower < err_upper:
                            constr_bndry = func.lower
                        else:
                            constr_bndry = func.upper
                        if constr_bndry == 0.0:
                            yvec = np.abs(yvec * func.scale)
                        else:
                            yvec = np.abs((yvec - constr_bndry) / constr_bndry)
                    # plot the function
                    ax.plot(
                        iterations,
                        yvec,
                        color=colors[ind],
                        linewidth=2,
                        label=func.plot_name,
                    )
                    ind += 1

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.xlabel("iterations")
            plt.ylabel("func values")
            plt.yscale("log")
            plot_filepath = os.path.join(self._write_path, self._plot_filename)
            plt.savefig(plot_filepath, dpi=300)
            plt.close("all")

    def _design_report(self):
        driver = self.options["driver"]
        model = driver.model

        if driver.comm.rank == 0:
            variables = model.get_variables()
            self._design_hdl.write(f"Design #{self._iteration}...\n")
            self._design_hdl.write(f"\tf2f vars = {[_.name for _ in variables]}\n")
            real_xarray = [var.value for var in variables]
            self._design_hdl.write(f"\tvalues = {real_xarray}\n")
            self._design_hdl.flush()
