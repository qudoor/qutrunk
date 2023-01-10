"""Implementation of quantum compute simulator for gpu running mode."""
import math
from typing import Union

import numpy
from numba import cuda, float32
from .sim_local import SimLocal, PauliOpType

def extract_bit(ctrl, index):
    return (index & (2**ctrl)) // (2**ctrl)
    
@cuda.jit
def init_classical_state_kernel(num_amps_per_rank, real, imag, state_ind):
    """Init classical state kernel"""
    index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if index < num_amps_per_rank:
        real[index] = 0.0
        imag[index] = 0.0
    if index == state_ind:
        real[state_ind] = 1.0
        imag[state_ind] = 0.0
    cuda.syncthreads()
    
@cuda.jit
def init_plus_state_kernel(num_amps_per_rank, real, imag):
    """Init plus state kernel"""
    index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    norm_factor = 1.0/math.sqrt(num_amps_per_rank)
    if index < num_amps_per_rank:
        real[index] = norm_factor
        imag[index] = 0.0
    cuda.syncthreads()
    
@cuda.jit  
def init_zero_state_kernel(num_amps_per_rank, real, imag):
    """Init zero state kernel"""
    index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if index < num_amps_per_rank:
        real[index] = 0.0
        imag[index] = 0.0
    if index == 0:
        real[0] = 1.0
        imag[0] = 0.0
    cuda.syncthreads()

@cuda.jit  
def amp_kernel(num_amps_per_rank, real, imag, orgreal, orgimag, startindex):
    """Init zero state kernel"""
    index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if index < num_amps_per_rank:
        endindex = startindex+index
        real[endindex] = orgreal[index]
        imag[endindex] = orgimag[index]
    cuda.syncthreads()

@cuda.jit
def hadamard_kernel(num_amps_per_rank, real, imag, target_qubit):
    size_half_block = 2**target_qubit
    size_block = size_half_block * 2
    num_task = num_amps_per_rank // 2
    rec_root = 1.0 / math.sqrt(2)
        
    this_task = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if this_task>=num_task:
        return

    this_block = this_task // size_half_block
    index_up = this_block*size_block + this_task%size_half_block
    index_lo = index_up + size_half_block

    state_real_up = real[index_up]
    state_imag_up = imag[index_up]
    state_real_lo = real[index_lo]
    state_imag_lo = imag[index_lo]

    real[index_up] = rec_root*(state_real_up + state_real_lo)
    imag[index_up] = rec_root*(state_imag_up + state_imag_lo)
    real[index_lo] = rec_root*(state_real_up - state_real_lo)
    imag[index_lo] = rec_root*(state_imag_up - state_imag_lo)

@cuda.jit
def controlled_not_kernel(num_amps_per_rank, real, imag, control_qubit, target_qubit):
    size_half_block = 2**target_qubit
    size_block = size_half_block * 2
    num_task = num_amps_per_rank // 2
    
    this_task = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if this_task>=num_task:
        return
    
    this_block = this_task // size_half_block
    index_up = this_block*size_block + this_task%size_half_block
    index_lo = index_up + size_half_block
    
    control_bit = extract_bit(control_qubit, index_up)
    if control_bit:
        state_real_up = real[index_up]
        state_imag_up = imag[index_up]
        real[index_up] = real[index_lo]
        imag[index_up] = imag[index_lo]
        real[index_lo] = state_real_up
        imag[index_lo] = state_imag_up
    
class GpuLocal:
    """Simulator-gpu implement."""

    def __init__(self):
        self.sim_cpu = None
    
    def create_qureg(self, num_qubits):  
        self.sim_cpu = SimLocal()
        self.sim_cpu.create_qureg(num_qubits)
        num_amps = 2**num_qubits
        self.real = cuda.device_array(num_amps, numpy.float_)
        self.imag = cuda.device_array(num_amps, numpy.float_)

    def init_classical_state(self):
        """Init classical state"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        init_classical_state_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, 0)
            
    def init_plus_state(self):
        """Init plus state"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        init_plus_state_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag)
    
    def init_zero_state(self):
        """Init zero state"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        init_zero_state_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag)

    def amp(self, reals, imags, startindex):
        """Init amplitudes state"""
        orgreal = cuda.to_device(reals, stream=0)
        orgimag = cuda.to_device(imags, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        amp_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, orgreal, orgimag, startindex)
        
    def hadamard(self, target_qubit):
        """Apply hadamard gate.

        Args:
            target: target qubit.
        """
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        hadamard_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit)
