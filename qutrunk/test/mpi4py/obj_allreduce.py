from mpi4py import MPI
from mpi4py.MPI import Op

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = (rank + 1) ** 2

data = comm.allreduce(data, MPI.SUM)
print(data)
