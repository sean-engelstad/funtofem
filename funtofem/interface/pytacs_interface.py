
__all__ = ["PytacsInterface"]

from mpi4py import MPI
from tacs import pytacs, TACS, functions
from .utils import f2f_callback, addLoadsFromBDF
from ._solver_interface import SolverInterface
import os, numpy as np
from .tacs_interface_unsteady import TacsUnsteadyInterface
from .utils.general_utils import real_norm, imag_norm
from .utils.relaxation_utils import AitkenRelaxationTacs
from .utils.pytacs_utils import TacsOutputGenerator, TacsPanelDimensions


class PytacsInterface:
    LENGTH_VAR = "LENGTH"
    LENGTH_CONSTR = "PANEL-LENGTH"
    WIDTH_VAR = "WIDTH"
    WIDTH_CONSTR = "PANEL-WIDTH"

    def __init__(
        self,
        comm,
        model,
        static_problem, # use tacs_comm input for this
        output_dir="",
        thermal_index=0,
        const_load=None,
        nprocs=None,
        relaxation_scheme:AitkenRelaxationTacs = None,
        tacs_panel_dimensions=None,
    ):
        self.comm = comm
        self.model = model
        self.static_problem = static_problem
        self.output_dir = output_dir
        self.thermal_index = thermal_index
        self.const_load_vec = const_load_vec
        self.nprocs = nprocs
        self.relaxation_scheme = relaxation_scheme
        self.tacs_panel_dimensions = tacs_panel_dimensions

        self.num_analysis = 0
        self.funcs_sens_dict = None

        self.tacs_proc = False
        if self.static_problem.assembler is not None:
            self.tacs_proc = True

        self.assembler = self.static_problem.assembler

        self.const_force = None
        if self.tacs_proc and const_load is not None:
            self.const_force = const_load.getArray()

        self.variables = model.get_variables()
        # initialize tolerances in base class
        super().__init__()

        # Get the structural variables from the global list of variables.
        self.struct_variables = []
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        # allocate the functions
        for scenario in model.scenarios:
            self._allocate_functions(scenario)

    @property
    def tacs_comm(self):
        return self.static_problem.comm

    def _allocate_functions(self, scenario):
        """
        allocate all scenario functions into TACS static problem
        could be modified later to allow only certain component IDs potentially

        TODO : improve this functionality => maybe have funtofem functions be able to include tacs functions and set more data through that?
        or using kwargs into f2f functions?
        """
        if self.tacs_proc:
            SP = self.static_problem
            assembler = self.assembler
            
            for func in scenario.functions:
                if func.analysis_type != "structural":
                    continue

                elif func.name.lower() == "ksfailure":
                    ksweight = 50.0
                    if func.options is not None and "ksweight" in func.options:
                        ksweight = func.options["ksweight"]
                    safetyFactor = 1.0
                    if func.options is not None and "safetyFactor" in func.options:
                        safetyFactor = func.options["safetyFactor"]
                    my_func = functions.KSFailure(
                        assembler, ksWeight=ksweight, safetyFactor=safetyFactor
                    )

                elif func.name.lower() == "compliance":
                    my_func = functions.Compliance(assembler)

                elif func.name.lower() == "temperature":
                    # TODO : need way to set volume into here or user can make the functions first from TACS
                    my_func = functions.AverageTemperature(assembler, volume=1.0)

                elif func.name.lower() == "heatflux":
                    my_func = functions.HeatFlux(assembler)

                elif func.name.lower() == "xcom":
                    my_func = functions.CenterOfMass(assembler, direction=[1, 0, 0])

                elif func.name.lower() == "ycom":
                    my_func = functions.CenterOfMass(assembler, direction=[0, 1, 0])

                elif func.name.lower() == "zcom":
                    my_func = functions.CenterOfMass(assembler, direction=[0, 0, 1])

                elif func.name == "mass":
                    my_func = functions.StructuralMass(assembler)

                else:
                    print("WARNING: Unknown function being set into TACS set to mass")
                    my_func = functions.StructuralMass(assembler)

                # could be func.full_name here?
                SP.addFunction(func.name, my_func)

    def set_variables(self, scenario, bodies):
        """
        Set the design variable values into the structural solver.

        This takes the variables that are set in the list of :class:`~variable.Variable` objects
        and sets them into the TACSAssembler object.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        if self.tacs_panel_dimensions is not None:
            # compute panel length and width (checks if we need to inside the object)
            self.tacs_panel_dimensions.compute_panel_length(
                self.assembler,
                self.struct_variables,
                self.LENGTH_CONSTR,
                self.LENGTH_VAR,
            )
            self.tacs_panel_dimensions.compute_panel_width(
                self.assembler, self.struct_variables, self.WIDTH_CONSTR, self.WIDTH_VAR
            )

        # write F2F variable values into TACS
        if self.comm.rank == 0:
            des_vars = np.array([var.value for var in self.struct_variables])
            self.static_problem.setDesignVars(des_vars)

            # also copy to the constraints local vectors
            if self.tacs_panel_dimensions is not None:
                self.tacs_panel_dimensions.setDesignVars(self.static_problem.x)

        return

    def set_functions(self, scenario, bodies):
        """
        Set and initialize the types of functions that will be evaluated based
        on the list of functions stored in the scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the structural functions of interest.

        The functions are evaluated based on the values of the state variables set
        into the TACSAssembler object. These values are only available on the
        TACS processors, but these values are broadcast to all processors after the
        evaluation.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        funcs_dict = {}
        self.static_problem.evalFunctions(funcs_dict)
        
        for key in funcs_dict:
            for func in scenario.functions:
                if func.analysis_type == "structural" and key == func.name:
                    func.value = funcs_dict[key]

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Take the gradients that were computed in the post_adjoint() call and
        place them into the functions of interest. This function can only be called
        after solver.post_adjoint(). This call order is guaranteed.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        funcs_sens_dict = self.funcs_sens_dict
        
        for key in funcs_sens_dict:
            for func in scenario.functions:
                if key == func.name:
                    grad = funcs_sens_dict[key]['struct']
                    for i,var in enumerate(self.struct_variables):
                        func.add_gradient_component(var, grad[i])

        return

    def initialize(self, scenario, bodies):
        """
        Initialize the internal data here for solving the governing
        equations. Set the nodes in the structural mesh to be consistent
        with the nodes stored in the body classes.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        if self.tacs_proc:
            for body in bodies: # TODO : what to do for multi-body?
                self.static_problem.setNodes(body.get_struct_nodes())

        return 0

    def iterate(self, scenario, bodies, step):
        """
        This function performs an iteration of the structural solver

        The code performs an update of the governing equations

        S(u, fS, hS) = r(u) - fS - hS

        where fS are the structural loads and hS are the heat fluxes stored in the body
        classes.

        The code computes

        res = r(u) - fS - hS

        and then computes the update

        mat * update = -res

        applies the update

        u = u + update

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """
        fail = 0

        if self.tacs_proc:

            # Add the external forces into a TACS vector that will be added to
            # the residual
            ext_force_array = self.static_problem.externalForce.getArray()

            # Add the external load and heat fluxes on the structure
            ndof = self.assembler.getVarsPerNode()
            for body in bodies:
                struct_loads = body.get_struct_loads(scenario, time_index=step)
                if self._debug:
                    print(f"========================================")
                    print(f"Inside tacs_interface, step: {step}")
                    print(f"norm of real struct_loads: {real_norm(struct_loads)}")
                    print(f"norm of imaginary struct_loads: {imag_norm(struct_loads)}")
                    print(f"========================================\n", flush=True)
                if struct_loads is not None:
                    for i in range(3):
                        ext_force_array[i::ndof] += struct_loads[i::3].astype(
                            TACS.dtype
                        )

                struct_flux = body.get_struct_heat_flux(scenario, time_index=step)
                if struct_flux is not None:
                    ext_force_array[self.thermal_index :: ndof] += struct_flux[
                        :
                    ].astype(TACS.dtype)

            # add in optional constant load
            if self.has_const_load:
                ext_force_array[:] += self.const_force[:]
            # TODO : add aitken relaxation back in here..

            self.static_problem.solve(Fext=ext_force_array)

            # copy the solution into the f2f body
            ans_array = self.static_problem.u.getArray()
            for body in bodies:
                struct_disps = body.get_struct_disps(scenario, time_index=step)
                if struct_disps is not None:
                    for i in range(3):
                        struct_disps[i::3] = ans_array[i::ndof].astype(body.dtype)

                # Set the structural temperature
                struct_temps = body.get_struct_temps(scenario, time_index=step)
                if struct_temps is not None:
                    # absolute temperature in Kelvin of the structural surface
                    struct_temps[:] = (
                        ans_array[self.thermal_index :: ndof].astype(body.dtype)
                        + scenario.T_ref
                    )

        self.num_analysis += 1

        return fail

    def post(self, scenario, bodies):
        # write solution from tacs into f5 file
        self.static_problem.writeSolution(self.output_dir, number=self.num_analysis)

        return

    def initialize_adjoint(self, scenario, bodies):
        """
        Initialize the solver for adjoint computations.

        This code computes the transpose of the Jacobian matrix dS/du^{T}, and factorizes
        it. For structural problems, the Jacobian matrix is symmetric, however for coupled
        thermoelastic problems, the Jacobian matrix is non-symmetric.

        The code also computes the derivative of the structural functions of interest with
        respect to the state variables df/du. These derivatives are stored in the list
        func_list that stores svsens = -df/du. Note that the negative sign applied here.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        # TODO : add aitken back into here?

        # reset the funcs sens dict to empty
        self.funcs_sens_dict = {}

        # write adjoint rhs into the static problem dIduList
        for func in se

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        This function solves the structural adjoint equations.

        The governing equations for the structures takes the form

        S(u, fS, hS) = r(u) - fS - hS = 0

        The function takes the following adjoint-Jacobian products stored in the bodies

        struct_disps_ajp = psi_D^{T} * dD/duS + psi_L^{T} * dL/dus
        struct_temps_ajp = psi_T^{T} * dT/dtS

        and computes the outputs that are stored in the same set of bodies

        struct_loads_ajp = psi_S^{T} * dS/dfS
        struct_flux_ajp = psi_S^{T} * dS/dhS

        Based on the governing equations, the outputs are computed based on the structural adjoint
        variables as

        struct_loads_ajp = - psi_S^{T}
        struct_flux_ajp = - psi_S^{T}

        To obtain these values, the code must solve the structural adjoint equation

        dS/duS^{T} * psi_S = - df/duS^{T} - dD/duS^{T} * psi_D - dL/duS^{T} * psi_L^{T} - dT/dtS^{T} * psi_T

        In the code, the right-hand-side for the

        dS/duS^{T} * psi_S = struct_rhs_array

        This right-hand-side is stored in the array struct_rhs_array, and computed based on the array

        struct_rhs_array = svsens - struct_disps_ajp - struct_flux_ajp

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """
        fail = 0

        if self.tacs_proc:

            # solve the adjoint RHS inside this method
            self.static_problem.evalFunctionsSens(self.funcs_sens_dict)

            for ifunc in range(len(func_list)):
                # Check if the function requires an adjoint computation or not
                if func_tags[ifunc] == -1:
                    continue

                # Copy values into the right-hand-side
                # res = - df/duS^{T}
                self.res.copyValues(dfdu[ifunc])

                # Extract the array in-place
                array = self.res.getArray()

                ndof = self.assembler.getVarsPerNode()
                for body in bodies:
                    # Form new right-hand side of structural adjoint equation using state
                    # variable sensitivites and the transformed temperature transfer
                    # adjoint variables. Here we use the adjoint-Jacobian products from the
                    # structural displacements and structural temperatures.
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        for i in range(3):
                            array[i::ndof] -= struct_disps_ajp[i::3, ifunc].astype(
                                TACS.dtype
                            )

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        array[self.thermal_index :: ndof] -= struct_temps_ajp[
                            :, ifunc
                        ].astype(TACS.dtype)

                # Zero the adjoint right-hand-side conditions at DOF locations
                # where the boundary conditions are applied. This is consistent with
                # the forward analysis where the forces/fluxes contributions are
                # zeroed at Dirichlet DOF locations.
                self.assembler.applyBCs(self.res)

                # Solve structural adjoint equation
                # print(f"linear adjoint solve", flush=True)
                self.gmres.solve(self.res, psi[ifunc])
                # print(f"\tdone withlinear adjoint solve", flush=True)

                # Aitken adjoint step
                if self.use_aitken:
                    psi_temp[ifunc].copyValues(psi[ifunc])
                    theta_adj = self.theta_adj
                    prev_theta_adj = self.prev_theta_adj

                    if step >= 2:
                        # Calculate adjoint update value
                        update_adj[ifunc].copyValues(psi_temp[ifunc])
                        update_adj[ifunc].axpy(-1, prev_psi[ifunc])

                    if step >= 3:
                        # Perform Aitken relaxation
                        delta_update_adj[ifunc].copyValues(update_adj[ifunc])
                        delta_update_adj[ifunc].axpy(-1, prev_update_adj[ifunc])

                        num = delta_update_adj[ifunc].dot(update_adj[ifunc])
                        den = delta_update_adj[ifunc].norm() ** 2.0

                        if self.comm.rank == 0 and self.aitken_debug_more:
                            print(
                                f"prev_theta_adj[ifunc]: {prev_theta_adj[ifunc]}",
                                flush=True,
                            )
                            print(f"num: {num}", flush=True)
                            print(f"den: {den}", flush=True)

                        # Only update theta if vector has changed more than tolerance
                        if np.real(den) > self.aitken_tol:
                            theta_adj[ifunc] = prev_theta_adj[ifunc] * (1.0 - num / den)
                            if self.comm.rank == 0 and self.aitken_debug:
                                print(
                                    f"Theta adjoint unbounded, ifunc {ifunc}: {theta_adj[ifunc]}",
                                    flush=True,
                                )
                        else:
                            # If den is too small, then reset theta to 1.0 to turn off relaxation
                            theta_adj[ifunc] = 1.0

                        theta_adj[ifunc] = max(
                            aitken_min, min(aitken_max, np.real(theta_adj[ifunc]))
                        )

                        # Use psi_temp variable to store scaled update
                        psi_temp[ifunc].copyValues(update_adj[ifunc])
                        psi_temp[ifunc].scale(theta_adj[ifunc])

                        psi[ifunc].copyValues(prev_psi[ifunc])
                        psi[ifunc].axpy(1, psi_temp[ifunc])

                # Extract the structural adjoint array in-place
                psi_array = psi[ifunc].getArray()

                if self.use_aitken:
                    # Store psi and update_adj as previous for next iteration
                    prev_psi[ifunc].copyValues(psi[ifunc])
                    prev_update_adj[ifunc].copyValues(update_adj[ifunc])

                # Set the adjoint-Jacobian products for each body
                for body in bodies:
                    # Compute the structural loads adjoint-Jacobian product. Here
                    # S(u, fS, hS) = r(u) - fS - hS, so dS/dfS = -I and dS/dhS = -I
                    # struct_loads_ajp = psi_S^{T} * dS/dfS
                    struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                    if struct_loads_ajp is not None:
                        for i in range(3):
                            struct_loads_ajp[i::3, ifunc] = -psi_array[i::ndof].astype(
                                body.dtype
                            )

                    # struct_flux_ajp = psi_S^{T} * dS/dfS
                    struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)
                    if struct_flux_ajp is not None:
                        struct_flux_ajp[:, ifunc] = -psi_array[
                            self.thermal_index :: ndof
                        ].astype(body.dtype)

        # print(f"done with iterate adjoint", flush=True)

        return fail