export MY_MPI_EXEC=mpiexec_mpt
export MY_NPROCS=48
$MY_MPI_EXEC -n $MY_NPROCS python test_aero_loads_cmplx.py 2>&1 > aero_loads.txt
$MY_MPI_EXEC -n $MY_NPROCS python test_flow_states_cmplx.py 2>&1 > flow_states.txt
$MY_MPI_EXEC -n 1 python test_grid_deformation_cmplx.py 2>&1 > grid_deformation.txt
$MY_MPI_EXEC -n 1 python test_internal_adjoints_cmplx.py 2>&1 > internal_adjoints.txt
$MY_MPI_EXEC -n $MY_NPROCS python 1_test_DV_derivs_cmplx.py 2>&1 > DV_derivs.txt
