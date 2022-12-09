"""Implementation of quantum compute simulator for distributed running mode."""
from mpi4py import MPI

class DistEnv:
    """Run Environment."""

    def __init__(self):
        self.rank = None
        self.num_ranks = None
        self.seeds = None
        self.num_seeds = None


class DistReg:
    """Represents a system of qubits.."""
    
    def __init__(self):
        self.num_amps_per_chunk = None
        self.num_amps_total = None
        self.chunk_id = None
        self.num_chunks = None
        self.num_qubits_in_state_vec = None
        self.state_vec_real = None
        self.state_vec_imag = None
        self.pair_state_vec_real = None
        self.pair_state_vec_imag = None


class SimDistribute:
    """Simulator-distribute implement."""

    def __init__(self):
        self.env = None
        self.reg = None
        self.comm = MPI.COMM_WORLD
        self.create_env()

    def create_env(self):
        """Create run environment."""
        self.env = DistEnv()
        
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
        self.reg = DistReg()

        total_num_amps = 2**num_qubits
        num_amps_per_rank = total_num_amps/self.env.num_ranks
        self.reg.state_vec_real = [0] * total_num_amps
        self.reg.state_vec_imag = [0] * total_num_amps

        if self.env.num_ranks > 1:
            self.reg.pair_state_vec_real = [0] * total_num_amps
            self.reg.pair_state_vec_imag = [0] * total_num_amps

        self.reg.num_amps_total = total_num_amps
        self.reg.num_amps_per_chunk = num_amps_per_rank
        self.reg.chunk_id = self.env.rank
        self.reg.num_chunks = self.env.num_ranks

    def init_zero_state(self):
        """Init zero state"""
        self.init_blank_state()
        if self.reg.chunk_id == 0:
            # zero state |0000..0000> has probability 1
            self.reg.state_vec_real[0] = 1.0
            self.reg.state_vec_imag[0] = 0.0

    def init_blank_state(self):
        """Init blank state"""
        for i in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec_real[i] = 0.0
            self.reg.state_vec_imag[i] = 0.0
