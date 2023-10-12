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
    TestResult
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

@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFrameworkUnsteadyCoordStack(unittest.TestCase):
    FILENAME = "framework-unsteady-meld-coord-stack.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_meld_for_unsteady1(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(scenario)
        Function.test_aero().register_to(scenario)
        Variable.shape("rotation").register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # test the load transfer ajps
        h = 1e-30
        p = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        uS = np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
        fA = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        uA = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        q = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        xA0_bar = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        Ja = np.random.rand(3 * plate.aero_nnodes,3 * plate.aero_nnodes).astype(plate.dtype)
        Js = np.random.rand(3 * plate.struct_nnodes, 3 * plate.struct_nnodes).astype(plate.dtype)

        # real mode forward analysis
        plate.transfer.transferDisps(uS, uA) # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)
        
        # real mode adjoint analysis
        q[:] = 1.0
        plate.transfer.applydLdxA0(q, xA0_bar)
        dgdp_adj = -np.dot(xA0_bar, p).real

        # complex mode forward analysis
        plate.aero_X += 1j * p * h
        plate.update_transfer()
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        plate.transfer.transferDisps(uS, uA) # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)
        dgdp_cmplx = np.imag(np.sum(fS)) / h

        rel_error = (dgdp_adj - dgdp_cmplx) / dgdp_cmplx
        print(f"test 1 simple AJP scalar test")
        print(f"dgdp adj = {dgdp_adj}")
        print(f"dgdp cmplx = {dgdp_cmplx}")
        print(f"dgdp rel error = {rel_error}")

        assert abs(rel_error) < 1e-9

    def test_meld_for_unsteady2(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(scenario)
        Function.test_aero().register_to(scenario)
        Variable.shape("rotation").register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # test the load transfer ajps
        h = 1e-30
        p = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        uS = np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
        fA = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        uA = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        q = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        xA0_bar = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        Ja = np.random.rand(3 * plate.aero_nnodes,3 * plate.aero_nnodes).astype(plate.dtype)
        Js = np.random.rand(3 * plate.struct_nnodes, 3 * plate.struct_nnodes).astype(plate.dtype)

        # real mode forward analysis
        plate.transfer.transferDisps(uS, uA) # init disp transfer in order
        fA = Ja @ uA + 0.1*np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        plate.transfer.transferLoads(fA, fS)
        uS = Js @ fS + 0.1*np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
        
        # real mode adjoint analysis
        q[:] = 1.0
        fS_bar = Js.T @ q
        plate.transfer.applydLdxA0(fS_bar, xA0_bar)
        dgdp_adj = -np.dot(xA0_bar, p).real

        # complex mode forward analysis
        plate.aero_X += 1j * p * h
        plate.update_transfer()
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        plate.transfer.transferDisps(uS, uA) # init disp transfer in order
        fA = Ja @ uA + 0.1*np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        plate.transfer.transferLoads(fA, fS)
        uS = Js @ fS + 0.1*np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
        dgdp_cmplx = np.imag(np.sum(uS)) / h

        rel_error = TestResult.relative_error(dgdp_cmplx, dgdp_adj)
        print(f"test 2 coupling AJP test")
        print(f"dgdp adj = {dgdp_adj}")
        print(f"dgdp cmplx = {dgdp_cmplx}")
        print(f"dgdp rel error = {rel_error}")

        assert abs(rel_error) < 1e-9


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyCoordStack.FILEPATH, "w").close()
    unittest.main()
