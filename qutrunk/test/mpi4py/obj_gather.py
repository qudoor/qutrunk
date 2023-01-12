from mpi4py import MPI

# from mpi4py.util import dtlib

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = (rank + 1) ** 2
data = comm.gather(data, root=0)
if rank == 0:
    print(type(data), data)
else:
    assert data is not None
