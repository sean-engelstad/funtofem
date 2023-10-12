import os, unittest, numpy as np
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    make_test_directories,
    StackTester,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD
results_folder, output_folder = make_test_directories(comm, base_dir)

# user-defined settings
steps = (
    10  # purposely set very large time step since I manage adj rhs here, Sean Engelstad
)
elastic_scheme = "meld"
dt = 0.001


class UnsteadyMeldCoordStack(StackTester):
    def __init__(self):
        # build the model and driver
        self.model = FUNtoFEMmodel("wedge")
        self.plate = Body.aeroelastic("plate", boundary=1)
        self.plate.register_to(self.model)
        self.bodies = [self.plate]

        # build the scenario
        self.scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(self.scenario)
        Function.test_aero().register_to(self.scenario)
        Variable.shape("rotation").register_to(self.scenario)
        self.scenario.register_to(self.model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        self.solvers = SolverManager(comm)
        self.solvers.flow = TestAerodynamicSolver(comm, self.model)
        self.solvers.structural = TestStructuralSolver(comm, self.model)
        self.transfer_settings = TransferSettings(
            elastic_scheme=elastic_scheme, npts=10
        )

        # just build the driver but only used here to initialize the transfer schemes
        FUNtoFEMnlbgs(
            self.solvers, transfer_settings=self.transfer_settings, model=self.model
        )

        super(UnsteadyMeldCoordStack, self).__init__(comm=comm)

        # store the initial aero coordinates
        self._init_aero_X = self.plate.aero_X.copy()
        na = self.plate.get_num_aero_nodes()

        # choose a perturbation direction for the aero coordinates
        # could add option for struct vs aero coords later in this test
        self.p_aero_X = np.random.rand(3 * na)

    def reset(self):
        """reset the aero coordinates for the next test"""
        self.plate.aero_X = self._init_aero_X.copy()
        self.plate.update_transfer()
        self.plate.initialize_variables(self.scenario)
        self.plate.initialize_adjoint_variables(self.scenario)
        return

    def perturb_design(self, epsilon):
        self.plate.aero_X += self.p_aero_X * epsilon * 1j
        return

    # FORWARD STACK METHODS
    # ---------------------------------------------------------
    def forward_1(self):
        # u_S0 = 0 to u_A1 (disp transfer)
        self.plate.transfer_disps(self.scenario, time_index=0)
        aero_disps = self.plate.get_aero_disps(self.scenario, time_index=0)
        return np.sum(aero_disps)

    def forward_2(self):
        # u_A1 to f_A1 (aero analysis)
        self.solvers.flow.iterate(self.scenario, self.bodies, step=1)
        return np.sum(self.plate.get_aero_loads(self.scenario, time_index=1))

    def forward_3(self):
        # f_A1 to f_S1 (load transfer)
        self.plate.transfer_loads(self.scenario, time_index=1)
        return np.sum(self.plate.get_struct_loads(self.scenario, time_index=1))

    def forward_4(self):
        # f_S1 to u_S1 (struct analysis)
        self.solvers.structural.iterate(self.scenario, self.bodies, step=1)
        return np.sum(self.plate.get_struct_disps(self.scenario, time_index=1))

    def forward_5(self):
        # u_S1 to u_A2 (disp transfer)
        self.plate.transfer_disps(self.scenario, time_index=1)
        return np.sum(self.plate.get_aero_disps(self.scenario, time_index=2))

    def forward_6(self):
        # u_A2 to f_A2 (aero analysis step 2)
        self.solvers.flow.iterate(self.scenario, self.bodies, step=2)
        return np.sum(self.plate.get_aero_loads(self.scenario, time_index=1))

    # COMPLETE FORWARD STACK
    FORWARD_STACK = [forward_1, forward_2, forward_3, forward_4, forward_5, forward_6]

    # ADJOINT STACK METHODS
    # ---------------------------------------------------------
    def adjoint_6(self, start: bool):
        if start:
            self.plate.aero_loads_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
        self.solvers.flow.iterate_adjoint(self.scenario, self.bodies, step=2)
        return 0.0

    def adjoint_5(self, start: bool):
        if start:
            self.plate.aero_disps_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
        temp_xa = np.zeros(3 * self.plate.aero_nnodes, dtype=self.plate.dtype)
        psi_D = -self.plate.aero_disps_ajp[:, 0].copy()
        self.plate.transfer.applydDdxA0(psi_D, temp_xa)
        return np.sum(temp_xa)

    def adjoint_4(self, start: bool):
        if start:
            self.plate.struct_disps_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
        self.plate.transfer_disps(self.scenario, time_index=0)
        # purposely set the
        self.solvers.structural.iterate_adjoint(self.scenario, self.bodies, step=1)
        return 0.0

    def adjoint_3(self, start: bool):
        if start:
            self.plate.struct_loads_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
        self.plate.transfer_loads_adjoint(self.scenario, time_index=1)
        temp_xa = np.zeros(3 * self.plate.aero_nnodes, dtype=self.plate.dtype)
        psi_L = -self.plate.struct_loads_ajp[:, 0].copy()
        self.plate.transfer.applydLdxA0(psi_L, temp_xa)
        print(f"3 - {np.sum(temp_xa)}")
        return np.sum(temp_xa)

    def adjoint_2(self, start: bool):
        if start:
            self.plate.aero_loads_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
        self.solvers.flow.iterate_adjoint(self.scenario, self.bodies, step=1)
        return 0.0

    def adjoint_1(self, start: bool):
        if start:
            self.plate.aero_disps_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
        temp_xa = np.zeros(3 * self.plate.aero_nnodes, dtype=self.plate.dtype)
        psi_D = -self.plate.aero_disps_ajp[:, 0].copy()
        self.plate.transfer.applydDdxA0(psi_D, temp_xa)
        return np.sum(temp_xa)

    # COMPLETE ADJOINT STACK
    ADJOINT_STACK = [adjoint_1, adjoint_2, adjoint_3, adjoint_4, adjoint_5, adjoint_6]

    # FORWARD_STACK = FORWARD_STACK[:4]
    # ADJOINT_STACK = ADJOINT_STACK[:4]


@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFrameworkUnsteadyCoordStack(unittest.TestCase):
    FILENAME = "framework-unsteady-meld-coord-stack.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_unsteady_aero_aeroelastic(self):
        unsteady_test_stack = UnsteadyMeldCoordStack()
        fail = unsteady_test_stack.complex_step(
            test_name="framework-unsteady-meld",
            status_file=self.FILEPATH,
            epsilon=1e-30,
            rtol=1e-9,
        )
        assert not fail


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyCoordStack.FILEPATH, "w").close()
    unittest.main()
