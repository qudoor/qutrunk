from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = numpy.arange(100, dtype='i')
else:
    data = numpy.empty(100, dtype='i')
comm.Bcast(data, root=0)
for i in range(100):
    print(rank, data[i])
