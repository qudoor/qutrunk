from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = [(rank + 1) ** 2 for rank in range(size)]
else:
    data = None

data = comm.scatter(data, root=0)
print(data)

assert data == (rank + 1) ** 2
