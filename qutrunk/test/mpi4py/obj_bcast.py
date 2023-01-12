from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = rank

data = comm.bcast(data, root=0)
print(data)
