import numpy
from mpi4py import MPI
#from mpi4py.util import dtlib

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = numpy.arange(10, dtype='i')
    print(data)
    comm.Send(data, dest=1, tag=77)
    comm.Barrier()
else:
    comm.Barrier()
    data = numpy.empty(10, dtype='i')
    comm.Recv(data, source=0, tag=77)
    print(data)
