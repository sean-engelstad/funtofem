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
    imag_norm,
    real_norm,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD
results_folder, output_folder = make_test_directories(comm, base_dir)

class UnsteadyMeldCoordStack(StackTester):
    def __init__(self):
        # build the model and driver
        self.model = FUNtoFEMmodel("wedge")
        self.plate = Body.aeroelastic("plate")
        self.plate.register_to(self.model)
        self.bodies = [self.plate]

        # build the scenario
        self.scenario = Scenario.unsteady("test", steps=10)
        Function.test_struct().register_to(self.scenario)
        Function.test_aero().register_to(self.scenario)
        Variable.shape("rotation").register_to(self.scenario)
        self.scenario.register_to(self.model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        self.solvers = SolverManager(comm)
        self.solvers.flow = TestAerodynamicSolver(comm, self.model, npts=10)
        self.solvers.structural = TestStructuralSolver(comm, self.model, npts=25)
        self.transfer_settings = TransferSettings(
            elastic_scheme="meld", npts=10, beta=0.5, isym=-1
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

        # try changing u_S0 to nonzero..
        self.plate.struct_disps[self.scenario.id][0] = np.zeros(3 * self.plate.struct_nnodes).astype(self.plate.dtype)
        #self.plate.struct_disps[self.scenario.id][0] = np.random.rand(3 * self.plate.struct_nnodes).astype(self.plate.dtype)
        return

    def perturb_design(self, epsilon):
        self.plate.aero_X += self.p_aero_X * epsilon * 1j
        self.plate.update_transfer()
        return

    # FORWARD STACK METHODS
    # ---------------------------------------------------------
    def forward_1(self):
        # u_S0 = 0 to u_A1 (disp transfer)
        self.plate.transfer_disps(self.scenario, time_index=1)
        aero_disps = self.plate.get_aero_disps(self.scenario, time_index=1)
        return np.sum(aero_disps)

    def forward_2(self):
        # u_A1 to f_A1 (aero analysis)
        self.solvers.flow.iterate(self.scenario, self.bodies, step=1)
        return np.sum(self.plate.get_aero_loads(self.scenario, time_index=1))

    def forward_3(self):
        # f_A1 to f_S1 (load transfer)
        self.plate.transfer_loads(self.scenario, time_index=1)
        fS1 = self.plate.get_struct_loads(self.scenario, time_index=1)
        return np.sum(fS1)

    def forward_4(self):
        # f_S1 to u_S1 (struct analysis)
        self.solvers.structural.iterate(self.scenario, self.bodies, step=1)
        return np.sum(self.plate.get_struct_disps(self.scenario, time_index=1))

    def forward_5(self):
        # u_S1 to u_A2 (disp transfer)
        self.plate.transfer_disps(self.scenario, time_index=2)
        return np.sum(self.plate.get_aero_disps(self.scenario, time_index=2))

    def forward_6(self):
        # u_A2 to f_A2 (aero analysis step 2)
        self.solvers.flow.iterate(self.scenario, self.bodies, step=2)
        return np.sum(self.plate.get_aero_loads(self.scenario, time_index=2))

    def forward_7(self):
        # f_A2 to f_S2 (load transfer)
        self.plate.transfer_loads(self.scenario, time_index=2)
        fS2 = self.plate.get_struct_loads(self.scenario, time_index=2)
        return np.sum(fS2)

    def forward_8(self):
        # f_S2 to u_S2 (struct analysis step 2)
        self.solvers.structural.iterate(self.scenario, self.bodies, step=2)
        return np.sum(self.plate.get_struct_disps(self.scenario, time_index=2))

    # COMPLETE FORWARD STACK
    FORWARD_STACK = [
        forward_1,
        forward_2,
        forward_3,
        forward_4,
        forward_5,
        forward_6,
        forward_7,
        forward_8,
    ]

    # ADJOINT STACK METHODS
    # ---------------------------------------------------------
    def adjoint_8(self, start: bool):
        if start:
            self.plate.struct_disps_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
            self.plate.transfer_disps(self.scenario, time_index=2)
        # purposely set the
        self.solvers.structural.iterate_adjoint(self.scenario, self.bodies, step=2)
        return 0.0

    def adjoint_7(self, start: bool):
        if start:
            self.plate.struct_loads_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
            self.plate.transfer_disps(self.scenario, time_index=2)
        #self.plate.transfer_loads_adjoint(self.scenario, time_index=2)
        temp_xa = np.zeros(3 * self.plate.aero_nnodes, dtype=self.plate.dtype)
        psi_L = -self.plate.struct_loads_ajp[:, 0].copy()
        self.plate.transfer.applydLdxA0(psi_L, temp_xa)
        print(f"adj term 7 - {np.dot(temp_xa, self.p_aero_X)}")
        # transfer backwards to previous step
        self.plate.transfer_loads_adjoint(self.scenario, time_index=2)
        return np.dot(temp_xa, self.p_aero_X)

    def adjoint_6(self, start: bool):
        if start:
            self.plate.aero_loads_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
            self.plate.transfer_disps(self.scenario, time_index=2)
        self.solvers.flow.iterate_adjoint(self.scenario, self.bodies, step=2)
        return 0.0

    def adjoint_5(self, start: bool):
        if start:
            self.plate.aero_disps_ajp[:, 0] = np.ones((3 * self.plate.aero_nnodes,))
            self.plate.transfer_disps(self.scenario, time_index=2)
        temp_xa = np.zeros(3 * self.plate.aero_nnodes, dtype=self.plate.dtype)
        psi_D = -self.plate.aero_disps_ajp[:, 0].copy()
        self.plate.transfer.applydDdxA0(psi_D, temp_xa)
        print(f"adj term 5 = {np.dot(temp_xa, self.p_aero_X)}")
        # transfer backwards to previous step
        self.plate.transfer_disps_adjoint(self.scenario, time_index=2)
        return np.dot(temp_xa, self.p_aero_X)

    def adjoint_4(self, start: bool):
        if start:
            self.plate.struct_disps_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
        self.plate.transfer_disps(self.scenario, time_index=1)
        # self.plate.transfer_disps(self.scenario, time_index=0)
        # purposely set the
        self.solvers.structural.iterate_adjoint(self.scenario, self.bodies, step=1)
        return 0.0

    def adjoint_3(self, start: bool):
        if start:
            self.plate.struct_loads_ajp[:, 0] = np.ones((3 * self.plate.struct_nnodes,))
        self.plate.transfer_loads_adjoint(self.scenario, time_index=1)
        temp_xa = np.zeros((3 * self.plate.aero_nnodes,), dtype=self.plate.dtype)
        psi_L = -self.plate.struct_loads_ajp[:, 0].copy()
        self.plate.transfer.applydLdxA0(psi_L, temp_xa)
        print(f"adj term 3 - {np.dot(temp_xa, self.p_aero_X)}")
        return np.dot(temp_xa, self.p_aero_X)

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
        print(f"adj term 1 - {np.dot(temp_xa, self.p_aero_X)}")
        return np.dot(temp_xa, self.p_aero_X)

    # COMPLETE ADJOINT STACK
    ADJOINT_STACK = [
        adjoint_1,
        adjoint_2,
        adjoint_3,
        adjoint_4,
        adjoint_5,
        adjoint_6,
        adjoint_7,
        adjoint_8,
    ]

    # temporarily only do first 4 since 4 fails
    stop_early = False
    if stop_early:
        FORWARD_STACK = FORWARD_STACK[:4]
        ADJOINT_STACK = ADJOINT_STACK[:4]


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

    def test_unsteady(self):
        # hard-code the same derivatives of step 4 for comparison
        stack = UnsteadyMeldCoordStack()
        plate = stack.plate
        scenario = stack.scenario
        struct_solver = stack.solvers.structural
        flow_solver = stack.solvers.flow
        ns = plate.struct_nnodes
        na = plate.aero_nnodes
        transfer = plate.transfer
        Ja = flow_solver.scenario_data[scenario.id].Jac1
        Js = struct_solver.scenario_data[scenario.id].Jac1
        aero_X = plate.aero_X.copy()
        struct_X = plate.struct_X.copy()
        test_vec_a = stack.p_aero_X.copy()
        test_vec_s = np.ones((3*ns,),dtype=plate.dtype)

        # constant vectors
        ca = np.dot(flow_solver.scenario_data[scenario.id].b1, aero_X) + flow_solver.scenario_data[scenario.id].omega1 * 0.01
        cs = np.dot(struct_solver.scenario_data[scenario.id].b1, struct_X) + struct_solver.scenario_data[scenario.id].omega1 * 0.01

        # real forward analysis
        uS0 = np.zeros((3*ns,), dtype=plate.dtype)
        uA1 = np.zeros((3*na,),dtype=plate.dtype)
        fA1 = np.zeros((3*na,),dtype=plate.dtype)
        fS1 = np.zeros((3*ns,), dtype=plate.dtype)
        uS1 = np.zeros((3*ns,), dtype=plate.dtype)
        transfer.transferDisps(uS0, uA1)
        fA1[:] = np.dot(Ja, uA1) + ca
        transfer.transferLoads(fA1, fS1)
        uS1[:] = np.dot(Js, fS1) + cs

        # real adjoint analysis
        plate.initialize_adjoint_variables(scenario)
        uS1_bar = test_vec_s.copy()
        fS1_bar = np.dot(Js.T, uS1_bar)
        xA0_bar = np.zeros((3*na,), dtype=plate.dtype)
        plate.transfer.applydLdxA0(fS1_bar, xA0_bar)
        deriv_adj = np.dot(xA0_bar, test_vec_a)
        print(f"deriv adj = {deriv_adj}")

        # complex forward analysis
        h = 1e-30
        plate.aero_X += test_vec_a * h * 1j
        plate.update_transfer()
        uS0 = np.zeros((3*ns,), dtype=plate.dtype)
        uA1 = np.zeros((3*na,),dtype=plate.dtype)
        fA1 = np.zeros((3*na,),dtype=plate.dtype)
        fS1 = np.zeros((3*ns,), dtype=plate.dtype)
        uS1 = np.zeros((3*ns,), dtype=plate.dtype)
        transfer.transferDisps(uS0, uA1)
        fA1[:] = np.dot(Ja, uA1) + ca
        transfer.transferLoads(fA1, fS1)
        uS1[:] = np.dot(Js, fS1) + cs
        deriv_cmplx = np.imag(np.dot(uS1, test_vec_s)) / h
        print(f"deriv cmplx = {deriv_cmplx}")




if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyCoordStack.FILEPATH, "w").close()
    unittest.main()
