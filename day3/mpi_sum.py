from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = np.zeros(1)
rank[0] = comm.Get_rank()

sum = np.zeros(1)
comm.Reduce(rank, sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f'Sum of all ranks: {sum[0]}')