"""Implementation of quantum compute simulator for distributed running mode."""
import math
from mpi4py import MPI

class Env:
    """Represents run environment."""

    def __init__(self):
        self.rank = None
        self.num_ranks = None
        self.seeds = None
        self.num_seeds = None

class StateVec:
    """Represents state vector."""

    def __init__(self):
        self.real = None
        self.imag = None

class Reg:
    """Represents a system of qubits.."""

    def __init__(self):
        self.num_amps_per_chunk = None
        self.num_amps_total = None
        self.chunk_id = None
        self.num_chunks = None
        self.num_qubits_in_state_vec = None
        self.state_vec = StateVec()
        self.pair_state_vec = StateVec()


class SimDistribute:
    """Simulator-distribute implement."""

    def __init__(self):
        self.env = None
        self.reg = None
        self.comm = MPI.COMM_WORLD
        self.__create_env()

    def __create_env(self):
        """Create run environment."""
        self.env = Env()
        
        # init MPI environment
        if not MPI.Is_initialized():
            MPI.Init()
        
        self.env.rank = self.comm.Get_rank()
        self.env.num_ranks = self.comm.Get_size()

    def create_qureg(self, num_qubits):
        """Allocate resource.

        Args:
            num_qubits: number of qubits
        """
        self.reg = Reg()

        total_num_amps = 2**num_qubits
        num_amps_per_rank = total_num_amps/self.env.num_ranks
        self.reg.state_vec.real = [0] * total_num_amps
        self.reg.state_vec.imag = [0] * total_num_amps

        if self.env.num_ranks > 1:
            self.reg.pair_state_vec.real = [0] * total_num_amps
            self.reg.pair_state_vec.imag = [0] * total_num_amps

        self.reg.num_amps_total = total_num_amps
        self.reg.num_amps_per_chunk = num_amps_per_rank
        self.reg.chunk_id = self.env.rank
        self.reg.num_chunks = self.env.num_ranks

    def init_zero_state(self):
        """Init zero state"""
        self.__init_blank_state()
        if self.reg.chunk_id == 0:
            # zero state |0000..0000> has probability 1
            self.reg.state_vec.real[0] = 1.0
            self.reg.state_vec.imag[0] = 0.0

    def __init_blank_state(self):
        """Init blank state"""
        for i in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec.real[i] = 0.0
            self.reg.state_vec.imag[i] = 0.0

    def hadamard(self, target_qubit):
        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_qubit)

        # rank's chunk is in upper half of block 
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.__hadamard_local(target_qubit)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_isupper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
            pair_rank = self.__get_chunk_pair_id(rank_isupper, self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
            # printf("%d rank has pair rank: %d\n", qureg.rank, pairRank);
            # get corresponding values from my pair
            self.__exchange_state_vectors(pair_rank)
            # this rank's values are either in the upper of lower half of the block. send values to hadamardDistributed
            # in the correct order
            if rank_isupper:
                self.__hadamard_distributed(self.reg.state_vec, #upper
                        self.reg.pair_state_vec, #lower
                        self.reg.state_vec, rank_isupper); #output
            else:
                self.__hadamard_distributed(self.reg.pair_state_vec, #upper
                        self.reg.state_vec, #lower
                        self.reg.state_vec, rank_isupper); #output

    def __half_matrix_block_fits_in_chunk(self, chunkSize, target_qubit):
        size_half_block = 1 << (target_qubit)
        return 1 if chunkSize > size_half_block else 0

    def __hadamard_local(self, target_qubit):
        size_half_block = 2**target_qubit
        size_block = size_half_block * 2
        num_task = self.reg.num_amps_total // 2

        rec_root = 1.0 / math.sqrt(2)
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            state_real_up = self.reg.state_vec.real[index_up]
            state_imag_up = self.reg.state_vec.imag[index_up]

            state_real_lo = self.reg.state_vec.real[index_lo]
            state_imag_lo = self.reg.state_vec.imag[index_lo]

            self.reg.state_vec.real[index_up] = rec_root * (state_real_up + state_real_lo)
            self.reg.state_vec.imag[index_up] = rec_root * (state_imag_up + state_imag_lo)

            self.reg.state_vec.real[index_lo] = rec_root * (state_real_up - state_real_lo)
            self.reg.state_vec.imag[index_lo] = rec_root * (state_imag_up - state_imag_lo)

    def __chunk_isupper(self, chunk_id, chunk_size, target_ubit):
        size_half_block = 1 << target_ubit
        size_block = size_half_block * 2
        pos_in_block = (chunk_id * chunk_size) % size_block
        return pos_in_block < size_half_block

    def __get_chunk_pair_id(self, chunk_isupper, chunk_id, chunk_size, target_qubit):
        size_half_block = 1 << target_qubit
        chunks_per_half_block = size_half_block/chunk_size
        if chunk_isupper:
            return chunk_id + chunks_per_half_block
        else:
            return chunk_id - chunks_per_half_block

    def __exchange_state_vectors(self, pair_rank):
        # MPI send/receive vars
        tag = 100
        status = MPI.Status()
        # send my state vector to pairRank's qureg.pairStateVec
        # receive pairRank's state vector into qureg.pairStateVec
        for i in range(self.reg.num_amps_per_chunk):
            self.comm.Sendrecv(self.reg.state_vec.real[i], pair_rank, tag,
                    self.reg.pair_state_vec.real[i], pair_rank, tag, status)
            self.comm.Sendrecv(self.reg.state_vec.imag[i], pair_rank, tag,
                    self.reg.pair_state_vec.imag[i], pair_rank, tag, status)

    def __hadamard_distributed(self, state_vec_up, state_vec_lo, state_vec_out, update_upper):
        sign = 1 if update_upper else -1
        rec_root2 = 1.0/math.sqrt(2)

        for this_task in range(self.reg.num_amps_per_chunk):
            # store current state vector values in temp variables
            state_real_up = state_vec_up.real[this_task]
            state_imag_up = state_vec_up.imag[this_task]

            state_real_lo = state_vec_lo.real[this_task]
            state_imag_lo = state_vec_lo.imag[this_task]

            state_vec_out.real[this_task] = rec_root2*(state_real_up + sign * state_real_lo)
            state_vec_out.imag[this_task] = rec_root2*(state_imag_up + sign * state_imag_lo)