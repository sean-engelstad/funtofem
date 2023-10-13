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
    TestResult,
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
zero_disps = True


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

        # use same meshes as in test_transfer_schemes.py
        aero_X = np.array([0.23702916849535 +0.j, 0.007648373861731+0.j, 0.019830308342374+0.j], dtype=plate.dtype)
        struct_X = np.array([0.313092618649513+0.j, 0.099454664148885+0.j, 0.195174292110793+0.j,
                    0.207298021984729+0.j, 0.164931191211255+0.j, 0.711878958601386+0.j,
                    0.032066673180243+0.j, 0.197369618546919+0.j, 0.964556955391805+0.j], dtype=plate.dtype)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model, aero_X=aero_X)
        solvers.structural = TestStructuralSolver(comm, model, struct_X=struct_X)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10, isym=-1, beta=0.5)
        FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=model)

        # test the load transfer ajps
        h = 1e-30
        test_vec_a = np.array([8.401877171547095e-01, 7.830992237586059e-01, 9.116473579367843e-01], dtype=plate.dtype)
        test_vec_s = np.array([7.682295948119040e-01, 4.773970518621602e-01, 5.134009101956155e-01, 
                               6.357117279599009e-01, 6.069688762570586e-01, 1.372315767860187e-01, 
                               4.009443942461835e-01, 9.989245180035590e-01, 8.391122346926072e-01], dtype=plate.dtype)
        uS = np.zeros((3 * plate.struct_nnodes,)).astype(plate.dtype)
        #fA = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        fA = np.array([0.573894578454824+0.j, 0.699227657374234+0.j, 0.974641421401732+0.j], dtype=plate.dtype)
        
        # temp variables
        uA = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        xA0_bar = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)

        # real mode forward analysis
        plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)

        # real mode adjoint analysis
        plate.transfer.applydLdxA0(test_vec_s, xA0_bar)
        dgdp_adj = -np.dot(xA0_bar, test_vec_a).real

        # complex mode forward analysis
        plate.aero_X += 1j * test_vec_a * h
        plate.update_transfer()
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)
        dgdp_cmplx = np.imag(np.dot(fS, test_vec_s)) / h

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

        aero_X = np.array([0.23702916849535 +0.j, 0.007648373861731+0.j, 0.019830308342374+0.j], dtype=plate.dtype)
        struct_X = np.array([0.313092618649513+0.j, 0.099454664148885+0.j, 0.195174292110793+0.j,
                    0.207298021984729+0.j, 0.164931191211255+0.j, 0.711878958601386+0.j,
                    0.032066673180243+0.j, 0.197369618546919+0.j, 0.964556955391805+0.j], dtype=plate.dtype)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model, aero_X=aero_X)
        solvers.structural = TestStructuralSolver(comm, model, struct_X=struct_X)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10)
        FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=model)

        # test the load transfer ajps
        h = 1e-30
        #test_vec_a = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        #test_vec_s = np.ones((3 * plate.struct_nnodes,)).astype(plate.dtype)

        test_vec_a = np.array([8.401877171547095e-01, 7.830992237586059e-01, 9.116473579367843e-01], dtype=plate.dtype)
        #test_vec_s = np.array([7.682295948119040e-01, 4.773970518621602e-01, 5.134009101956155e-01, 
        #                        6.357117279599009e-01, 6.069688762570586e-01, 1.372315767860187e-01, 
        #                        4.009443942461835e-01, 9.989245180035590e-01, 8.391122346926072e-01], dtype=plate.dtype)
        test_vec_s = np.random.rand(3 * plate.struct_nnodes,).astype(plate.dtype)
        #test_vec_s = np.ones((3 * plate.struct_nnodes,)).astype(plate.dtype)
        print(f"test vec s = {test_vec_s} shape = {test_vec_s.shape}")
        fA = np.array([0.573894578454824+0.j, 0.699227657374234+0.j, 0.974641421401732+0.j], dtype=plate.dtype)

        uS = np.zeros((3 * plate.struct_nnodes,)).astype(plate.dtype)
        #fA = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
        uA = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        xA0_bar = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)

        # real mode forward analysis
        plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)

        # real mode adjoint analysis
        
        plate.transfer.applydLdxA0(test_vec_s, xA0_bar)
        dgdp_adj = -np.dot(xA0_bar, test_vec_a).real

        # complex mode forward analysis
        plate.aero_X += 1j * test_vec_a * h
        plate.update_transfer()
        fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
        plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
        plate.transfer.transferLoads(fA, fS)
        dgdp_cmplx = np.imag(np.dot(fS, test_vec_s)) / h

        rel_error = (dgdp_adj - dgdp_cmplx) / dgdp_cmplx
        print(f"test 1 simple AJP scalar test")
        print(f"dgdp adj = {dgdp_adj}")
        print(f"dgdp cmplx = {dgdp_cmplx}")
        print(f"dgdp rel error = {rel_error}")

        assert abs(rel_error) < 1e-9

    # def test_meld_for_unsteady2(self):
    #     # build the model and driver
    #     model = FUNtoFEMmodel("wedge")
    #     plate = Body.aeroelastic("plate", boundary=1)
    #     plate.register_to(model)

    #     # build the scenario
    #     scenario = Scenario.unsteady("test", steps=steps)
    #     Function.test_struct().register_to(scenario)
    #     Function.test_aero().register_to(scenario)
    #     Variable.shape("rotation").register_to(scenario)
    #     scenario.register_to(model)

    #     # build the tacs interface, coupled driver, and oneway driver
    #     comm = MPI.COMM_WORLD
    #     solvers = SolverManager(comm)
    #     solvers.flow = TestAerodynamicSolver(comm, model)
    #     solvers.structural = TestStructuralSolver(comm, model)
    #     transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=5)
    #     FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=model)

    #     # test the load transfer ajps
    #     h = 1e-30
    #     p = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
    #     if zero_disps:
    #         uS = np.zeros((3 * plate.struct_nnodes,)).astype(plate.dtype)
    #     else:
    #         uS = np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
    #     fA = np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
    #     uA = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
    #     fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
    #     q = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
    #     xA0_bar = np.zeros((3 * plate.aero_nnodes)).astype(plate.dtype)
    #     Ja = np.random.rand(3 * plate.aero_nnodes, 3 * plate.aero_nnodes).astype(
    #         plate.dtype
    #     )
    #     Js = np.random.rand(3 * plate.struct_nnodes, 3 * plate.struct_nnodes).astype(
    #         plate.dtype
    #     )

    #     # real mode forward analysis
    #     plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
    #     fA = Ja @ uA + 0.1 * np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
    #     plate.transfer.transferLoads(fA, fS)
    #     uS = Js @ fS + 0.1 * np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)

    #     # real mode adjoint analysis
    #     q[:] = 1.0
    #     fS_bar = Js.T @ q
    #     plate.transfer.applydLdxA0(fS_bar, xA0_bar)
    #     dgdp_adj = -np.dot(xA0_bar, p).real

    #     # complex mode forward analysis
    #     plate.aero_X += 1j * p * h
    #     plate.update_transfer()
    #     fS = np.zeros((3 * plate.struct_nnodes)).astype(plate.dtype)
    #     plate.transfer.transferDisps(uS, uA)  # init disp transfer in order
    #     fA = Ja @ uA + 0.1 * np.random.rand(3 * plate.aero_nnodes).astype(plate.dtype)
    #     plate.transfer.transferLoads(fA, fS)
    #     uS = Js @ fS + 0.1 * np.random.rand(3 * plate.struct_nnodes).astype(plate.dtype)
    #     dgdp_cmplx = np.imag(np.sum(uS)) / h

    #     rel_error = TestResult.relative_error(dgdp_cmplx, dgdp_adj)
    #     print(f"test 2 coupling AJP test")
    #     print(f"dgdp adj = {dgdp_adj}")
    #     print(f"dgdp cmplx = {dgdp_cmplx}")
    #     print(f"dgdp rel error = {rel_error}")

    #     assert abs(rel_error) < 1e-9


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyCoordStack.FILEPATH, "w").close()
    unittest.main()
