from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

arr_size = 5

data = [rank] * arr_size
pair_data = [10 - rank] * arr_size

pair_data = comm.sendrecv(data, size - rank - 1)
print(rank, data)
print(rank, pair_data)
