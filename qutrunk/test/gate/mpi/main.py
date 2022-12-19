import os  
main = "mpiexec -n 2 python qutrunk/test/gate/mpi/test_ch_gate.py"
os.system(main) 

main = "mpiexec -n 2 python qutrunk/test/gate/mpi/test_cp_gate.py"
os.system(main)