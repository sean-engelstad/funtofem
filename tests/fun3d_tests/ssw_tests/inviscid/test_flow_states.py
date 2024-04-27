"""
Test the vector function q(uA) and its associated adjoints, namely the flow states
from the aero disps on a converged flow of the SSW inviscid mesh.
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from ._base_test import *

# ---------------------------------------------------->
cruise.name = "cruise_inviscid_flow"

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14AeroelasticTestInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
)

max_rel_error = Fun3d14AeroelasticTestInterface.finite_diff_test_flow_states(
    solvers.flow, 
    epsilon=1e-4,
    filename="results/flow_states.txt"
)
if comm.rank == 0:
    print(f"max rel error = {max_rel_error}")