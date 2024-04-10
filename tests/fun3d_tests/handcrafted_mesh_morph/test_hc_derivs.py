import os, unittest, importlib
from funtofem.interface import Fun3dModel, Fun3dBC, HandcraftedMeshMorph
from funtofem.driver import TransferSettings
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import niceplots

# Imports from FUNtoFEM
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import (
    TacsSteadyInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

from funtofem import TransferScheme

base_dir = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD
meshes_dir = os.path.join(base_dir, "meshes")
csm_path = os.path.join(meshes_dir, "flow_wing.csm")

fun3d_loader = importlib.util.find_spec("fun3d")
caps_loader = importlib.util.find_spec("pyCAPS")

has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface


# first just test the fun3d and aflr aim features
@unittest.skipIf(
    fun3d_loader is None and caps_loader is None,
    "need CAPS to run this job, FUN3D not technically required but skipping anyways.",
)
class TestFun3dAimHandcraftedMeshDerivatives(unittest.TestCase):
    def test_adjoint_process(self):
        """do a finite difference test of the adjoint process with central difference"""

        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build(
            csm_file=csm_path,
            comm=comm,
            project_name="wing_test",
            root=0,
            mesh_morph=True,
        )
        mesh_aim = fun3d_model.mesh_aim
        fun3d_aim = fun3d_model.fun3d_aim

        mesh_aim.surface_aim.set_surface_mesh(
            ff_growth=1.3, min_scale=0.01, max_scale=5.0
        )
        mesh_aim.volume_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        twist = (
            Variable.shape(name="twist")
            .set_bounds(lower=-1.0, value=0.0, upper=1.0)
            .register_to(wing)
        )
        # sweep = (
        #     Variable.shape(name="sweep")
        #     .set_bounds(lower=-5.0, value=0.0, upper=10.0)
        #     .register_to(wing)
        # )
        wing.register_to(model)
        test_scenario = (
            Scenario.steady("euler", steps=5000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 4000

        aero_func = Function.test_aero().register_to(test_scenario)
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        # solvers = SolverManager(comm)
        # solvers.flow = Fun3d14Interface(
        #     comm, model, fun3d_dir="meshes"
        # )

        # create handcrafted mesh morph object after FUN3D is built
        handcrafted_mesh_morph = HandcraftedMeshMorph(
            comm=comm,
            model=model,
            transfer_settings=TransferSettings(npts=200, beta=0.5),
        )
        fun3d_model.handcrafted_mesh_morph = handcrafted_mesh_morph

        # copy the coordinates from the fun3d14interface to the handcrafted mesh object
        # read in the initial dat file and store in the
        _hc_morph_file = os.path.join(meshes_dir, "hc_mesh.dat")
        handcrafted_mesh_morph.read_surface_file(_hc_morph_file, is_caps_mesh=False)

        # compute an HC mesh test vector
        test_vec_hc = np.random.rand(3 * wing.aero_nnodes).astype(TransferScheme.dtype)

        # start of FD test
        # ------------------------------------
        h = 1e-5
        fd_deriv = None
        adj_deriv = None

        # first compute f(x) and df/dx

        # change the shape
        fun3d_aim.pre_analysis()

        wing.initialize_variables(test_scenario)
        wing.initialize_adjoint_variables(test_scenario)

        # compute the aero test functional
        hc_obj = handcrafted_mesh_morph
        aero_func.value = np.dot(test_vec_hc, hc_obj.u_hc)

        hc_aero_shape_term = wing.aero_shape_term[test_scenario.id]
        hc_aero_shape_term[:, 0] = test_vec_hc

        hc_obj.compute_caps_coord_derivatives()
        # overwrite the previous sens file
        hc_obj.write_sensitivity_file(
            comm=comm,
            filename=fun3d_aim.sens_file_path,
            discipline="aerodynamic",
            root=fun3d_aim.root,
            write_dvs=False,
        )
        fun3d_aim.post_analysis(None)
        fun3d_aim.unlink()

        # compute the adjoint derivative
        derivative = None
        if comm.rank == 0:
            direct_flow_aim = fun3d_aim.aim
            derivative = direct_flow_aim.dynout[aero_func.full_name].deriv(twist)
        derivative = comm.bcast(derivative, root=0)
        adj_deriv = np.real(derivative)

        # compute f(x+h)
        twist.value += h
        fun3d_aim.pre_analysis()

        # compute the aero test functional
        aero_func.value = np.dot(test_vec_hc, hc_obj.u_hc)

        hc_obj.compute_caps_coord_derivatives()
        # overwrite the previous sens file
        hc_obj.write_sensitivity_file(
            comm=comm,
            filename=fun3d_aim.sens_file_path,
            discipline="aerodynamic",
            root=fun3d_aim.root,
            write_dvs=False,
        )
        fun3d_aim.post_analysis(None)

        fd_deriv += aero_func.value / 2.0 / h

        # compute f(x-h)
        twist.value -= 2 * h
        fun3d_aim.pre_analysis()

        # compute the aero test functional
        aero_func.value = np.dot(test_vec_hc, hc_obj.u_hc)

        hc_obj.compute_caps_coord_derivatives()
        # overwrite the previous sens file
        hc_obj.write_sensitivity_file(
            comm=comm,
            filename=fun3d_aim.sens_file_path,
            discipline="aerodynamic",
            root=fun3d_aim.root,
            write_dvs=False,
        )
        fun3d_aim.post_analysis(None)

        fd_deriv -= aero_func.value / 2.0 / h
        fd_deriv = np.real(fd_deriv)

        # compare the adjoint and finite difference derivative
        rel_err = (adj_deriv - fd_deriv) / fd_deriv

        if comm.rank == 0:
            print(f"fd deriv  = {fd_deriv}")
            print(f"adj deriv = {adj_deriv}")
            print(f"rel err = {rel_err}")

        assert abs(rel_err) < 1e-4


if __name__ == "__main__":
    unittest.main()
