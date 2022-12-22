from .tacs_interface import TacsSteadyInterface


class TacsSteadyAnalysisDriver:
    """
    Class to perform only a TACS analysis with aerodynamic loads and heat fluxes in the body still retained.
    Similar to FUNtoFEMDriver class and FuntoFEMnlbgsDriver.
    Assumed to be ran after one solve_forward from a regular coupled problem, represents uncoupled
    TACS analysis from aerodynamic loads.
    """

    def __init__(self, tacs_interface: TacsSteadyInterface, model):
        self.tacs_interface = tacs_interface
        self.model = model

        # reset struct mesh positions
        for body in self.model.bodies:
            body.update_transfer()

        # zero out previous run data from funtofem
        # self._zero_tacs_data()
        # self._zero_adjoint_data()

    def solve_forward(self):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # zero all data to start fresh problem, u = 0, res = 0
        self._zero_tacs_data()

        for scenario in self.model.scenarios:

            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # run the forward analysis via iterate
            self.tacs_interface.initialize(scenario, self.model.bodies)
            self.tacs_interface.iterate(scenario, self.model.bodies, step=0)
            self.tacs_interface.post(scenario, self.model.bodies)

            # get functions to store the function values into the model
            self.tacs_interface.get_functions(scenario, self.model.bodies)

        return 0

    def solve_adjoint(self):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        # zero adjoint data
        self._zero_adjoint_data()

        for scenario in self.model.scenarios:
            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # zero all coupled adjoint variables in the body
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            self.tacs_interface.initialize_adjoint(scenario, self.model.bodies)
            self.tacs_interface.iterate_adjoint(scenario, self.model.bodies, step=0)
            self.tacs_interface.post_adjoint(scenario, self.model.bodies)

            # call get function gradients to store  the gradients from tacs
            self.tacs_interface.get_function_gradients(scenario, self.model.bodies)

    def _zero_tacs_data(self):
        """
        zero any TACS solution / adjoint data before running pure TACS
        """

        if self.tacs_interface.tacs_proc:

            # zero temporary solution data
            # others are zeroed out in the tacs_interface by default
            self.tacs_interface.res.zeroEntries()
            self.tacs_interface.ext_force.zeroEntries()
            self.tacs_interface.update.zeroEntries()

            # zero any scenario data
            for scenario in self.model.scenarios:

                # zero state data
                u = self.tacs_interface.scenario_data[scenario].u
                u.zeroEntries()
                self.tacs_interface.assembler.setVariables(u)

    def _zero_adjoint_data(self):

        if self.tacs_interface.tacs_proc:
            # zero adjoint variable
            for scenario in self.model.scenarios:
                psi = self.tacs_interface.scenario_data[scenario].psi
                for vec in psi:
                    vec.zeroEntries()