"""Implementation of quantum compute simulator for gpu running mode."""
import math
from typing import Union

import numpy
from numba import cuda, float32
from .sim_local import SimLocal, PauliOpType


class GpuLocal:
    """Simulator-gpu implement."""

    def __init__(self):
        self.sim_cpu = None
    
    def create_qureg(self, num_qubits):  
        self.sim_cpu.create_qureg(num_qubits)
        num_amps = 2**num_qubits
        self.real = cuda.shared.array(num_amps, dtype=float32)
        self.imag = cuda.shared.array(num_amps, dtype=float32)
        
    @cuda.jit
    def init_classical_state_kernel(self, state_ind):
        """Init classical state kernel"""
        index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        if index < self.sim_cpu.num_amps_per_rank:
            self.real[index] = 0.0
            self.imag[index] = 0.0
        if index == state_ind:
            self.real[state_ind] = 1.0
            self.imag[state_ind] = 0.0

    def init_classical_state(self):
        """Init classical state"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        self.init_classical_state_kernel[blocks_per_grid, threads_per_block]()
   
    @cuda.jit
    def init_plus_state_kernel(self):
        """Init plus state kernel"""
        index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        norm_factor = 1.0/math.sqrt(self.sim_cpu.num_amps_per_rank)
        if index < self.sim_cpu.num_amps_per_rank:
            self.real[index] = norm_factor
            self.imag[index] = 0.0
            
    def init_plus_state(self):
        """Init plus state"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        self.init_plus_state_kernel[blocks_per_grid, threads_per_block]()