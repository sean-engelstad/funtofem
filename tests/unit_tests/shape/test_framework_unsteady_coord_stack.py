import os, unittest, numpy as np
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    CoordinateDerivativeTester,
    make_test_directories,
    StackTester
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD
results_folder, output_folder = make_test_directories(comm, base_dir)

# user-defined settings
steps = 2
elastic_scheme = "meld"
dt = 0.001

class UnsteadyMeldCoordStack(StackTester):
    def __init__(self):
        # build the model and driver
        self.model = FUNtoFEMmodel("wedge")
        self.plate = Body.aeroelastic("plate", boundary=1)
        self.plate.register_to(self.model)

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
        self.transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10)

        # just build the driver but only used here to initialize the transfer schemes
        FUNtoFEMnlbgs(
            self.solvers, transfer_settings=self.transfer_settings, model=self.model
        )

        super(UnsteadyMeldCoordStack,self).__init__(comm=comm)

        # store the initial aero coordinates
        self._init_aero_X = self.plate.aero_X.copy()
        na = self.plate.get_num_aero_nodes()

        # choose a perturbation direction for the aero coordinates
        # could add option for struct vs aero coords later in this test
        self.p_aero_X = np.random.rand(3*na)

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
        self.plate.transfer_disps(self.scenario, time_index=0)
        aero_disps = self.plate.get_aero_disps(self.scenario,time_index=0)
        return np.sum(aero_disps)

    # COMPLETE FORWARD STACK
    FORWARD_STACK = [forward_1]

    # ADJOINT STACK METHODS 
    # ---------------------------------------------------------
    def adjoint_1(self):
        # NOTE : ADD SOME CHECK FOR WHETHER IT STARTED HERE?? in inputs?
        self.plate.transfer_disps_adjoint(self.scenario, time_index=0)
        struct_disps_ajp = self.plate.get_struct_disps_ajp(self.scenario)
        return np.sum(struct_disps_ajp)

    # COMPLETE ADJOINT STACK
    ADJOINT_STACK = [adjoint_1]

@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFrameworkUnsteadyCoordStack(unittest.TestCase):
    FILENAME = "framework-unsteady-meld-coord-stack.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_unsteady_aero_aeroelastic(self):
        unsteady_test_stack = UnsteadyMeldCoordStack()
        max_rel_error = unsteady_test_stack.complex_step(test_name="framework-unsteady-meld", status_file=self.FILEPATH, epsilon=1e-30)
        assert max_rel_error < 1e-9
        return

if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyCoordStack.FILEPATH, "w").close()
    unittest.main()
