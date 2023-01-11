"""Implementation of quantum compute simulator for gpu running mode."""
import math
from typing import Union

import numpy
from numba import cuda, float32
from qutrunk.backends.local.sim_local import SimLocal, PauliOpType

@cuda.jit
def extract_bit(ctrl, index):
    return (index & (2**ctrl)) // (2**ctrl)

@cuda.jit
def flip_bit(number, bit_ind):
    return (number ^ (2**bit_ind))

@cuda.jit
def insert_zero_bit(number, index):
    left = (number >> index) << index
    right = number - left
    return (left << 1) ^ right

@cuda.jit
def insert_two_zero_bits(number, bit1, bit2):
    small = bit1 if bit1 < bit2 else bit2
    big = bit2 if bit1 < bit2 else bit1
    return insert_zero_bit(insert_zero_bit(number, small), big)
    
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
def phase_shift_kernel(num_amps_per_rank, real, imag, target, cos_angle, sin_angle):
    num_tasks = num_amps_per_rank >> 1

    size_half_block = 1 << target
    size_block      = 2 * size_half_block

    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return

    this_block = this_task // size_half_block
    index_up   = this_block * size_block + this_task % size_half_block
    index_lo   = index_up + size_half_block

    state_real_lo = real[index_lo]
    state_imag_lo = imag[index_lo]

    real[index_lo] = cos_angle*state_real_lo - sin_angle*state_imag_lo
    imag[index_lo] = sin_angle*state_real_lo + cos_angle*state_imag_lo

@cuda.jit
def controlled_phase_shift_kernel(num_amps_per_rank, real, imag, control, target, cos_angle, sin_angle):
    index = cuda.grid(1)
    if (index >= num_amps_per_rank):
        return

    bit1 = extract_bit(control, index)
    bit2 = extract_bit (target, index)
    if (bit1 and bit2):
        state_real_lo = real[index]
        state_imag_lo = imag[index]
        
        real[index] = cos_angle*state_real_lo - sin_angle*state_imag_lo
        imag[index] = sin_angle*state_real_lo + cos_angle*state_imag_lo

@cuda.jit
def rotate_around_axis_kernel(num_amps_per_rank, real, imag, target, alpha_real, alpha_imag, beta_real, beta_imag):
    num_tasks = num_amps_per_rank >> 1

    size_half_block = 1 << target
    size_block      = 2 * size_half_block

    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return

    this_block = this_task // size_half_block
    index_up   = this_block * size_block + this_task % size_half_block
    index_lo    = index_up + size_half_block

    # store current state vector values in temp variables
    state_real_up = real[index_up]
    state_imag_up = imag[index_up]

    state_real_lo = real[index_lo]
    state_imag_lo = imag[index_lo]

    real[index_up] = alpha_real * state_real_up - alpha_imag * state_imag_up - beta_real * state_real_lo - beta_imag * state_imag_lo
    imag[index_up] = alpha_real * state_imag_up + alpha_imag * state_real_up - beta_real * state_imag_lo + beta_imag * state_real_lo

    real[index_lo] = beta_real * state_real_up - beta_imag * state_imag_up + alpha_real * state_real_lo + alpha_imag * state_imag_lo
    imag[index_lo] = beta_real * state_imag_up + beta_imag * state_real_up + alpha_real * state_imag_lo - alpha_imag * state_real_lo

@cuda.jit
def rotate_around_axis_kernel(num_amps_per_rank, real, imag, target, alpha_real, alpha_imag, beta_real, beta_imag):
    num_tasks = num_amps_per_rank >> 1

    size_half_block = 1 << target
    size_block      = 2 * size_half_block

    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return

    this_block = this_task // size_half_block
    index_up   = this_block * size_block + this_task % size_half_block
    index_lo    = index_up + size_half_block

    # store current state vector values in temp variables
    state_real_up = real[index_up]
    state_imag_up = imag[index_up]

    state_real_lo = real[index_lo]
    state_imag_lo = imag[index_lo]

    real[index_up] = alpha_real * state_real_up - alpha_imag * state_imag_up - beta_real * state_real_lo - beta_imag * state_imag_lo
    imag[index_up] = alpha_real * state_imag_up + alpha_imag * state_real_up - beta_real * state_imag_lo + beta_imag * state_real_lo

    real[index_lo] = beta_real * state_real_up - beta_imag * state_imag_up + alpha_real * state_real_lo + alpha_imag * state_imag_lo
    imag[index_lo] = beta_real * state_imag_up + beta_imag * state_real_up + alpha_real * state_imag_lo - alpha_imag * state_real_lo

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

@cuda.jit
def multi_controlled_multi_qubit_not_kernel(num_amps_per_rank, real, imag, ctrl_mask, targ_mask):
    # althouugh each thread swaps/updates two amplitudes, we still invoke one thread per amp
    amp_ind = cuda.grid(1)
    if (amp_ind >= num_amps_per_rank):
        return

    # modify amplitudes only if control qubits are 1 for this state
    if (ctrl_mask and ((ctrl_mask & amp_ind) != ctrl_mask)):
        return
    
    mate_ind = amp_ind ^ targ_mask
    
    # if the mate is lower index, another thread is handling it
    if (mate_ind < amp_ind):
        return
        
    # /* it may seem wasteful to spawn more threads than are needed, and abort 
    #  * half of them due to the amp pairing above (and potentially abort
    #  * an exponential number due to ctrlMask). however, since we are moving 
    #  * global memory directly in a potentially non-contiguous fashoin, this 
    #  * method is likely to be memory bandwidth bottlenecked anyway 
    #  */
    
    mate_re = real[mate_ind]
    mate_im = imag[mate_ind]
    
    # swap amp with mate
    real[mate_ind] = real[amp_ind]
    imag[mate_ind] = imag[amp_ind]
    real[amp_ind] = mate_re
    imag[amp_ind] = mate_im

@cuda.jit
def cy_kernel(num_amps_per_rank, real, imag, control, target, conj_factor):                                 
    size_half_block = 1 << target
    sizeBlock       = 2 * size_half_block

    index = cuda.grid(1)
    if (index >= (num_amps_per_rank >> 1)):
        return

    this_block   = index / size_half_block
    index_up     = this_block * sizeBlock + index % size_half_block
    index_lo     = index_up + size_half_block

    control_bit = extract_bit(control, index_up)
    if (control_bit):
        state_real_up = real[index_up]
        state_imag_up = imag[index_up]

        # update under +-{{0, -i}, {i, 0}}
        real[index_up] = conj_factor * imag[index_lo]
        imag[index_up] = conj_factor * -real[index_lo]
        real[index_lo] = conj_factor * -state_imag_up
        imag[index_lo] = conj_factor * state_real_up

@cuda.jit
def cz_kernel(num_amps_per_rank, real, imag, mask):
    index = cuda.grid(1)
    if (index >= num_amps_per_rank):
        return

    if (mask == (mask & index)):
        real [index] = - real[index]
        imag [index] = - imag[index]

@cuda.jit
def unitary_kernel(num_amps_per_rank, real, imag, target_qubit, ureal, uimag):
    size_half_block = 2**target_qubit
    size_block = size_half_block * 2
    num_task = num_amps_per_rank // 2
    
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

    real[index_up] = ureal[0][0]*state_real_up - uimag[0][0]*state_imag_up + ureal[0][1]*state_real_lo - uimag[0][1]*state_imag_lo
    imag[index_up] = ureal[0][0]*state_imag_up + uimag[0][0]*state_real_up + ureal[0][1]*state_imag_lo + uimag[0][1]*state_real_lo
    real[index_lo] = ureal[1][0]*state_real_up - uimag[1][0]*state_imag_up + ureal[1][1]*state_real_lo - uimag[1][1]*state_imag_lo
    imag[index_lo] = ureal[1][0]*state_imag_up + uimag[1][0]*state_real_up + ureal[1][1]*state_imag_lo + uimag[1][1]*state_real_lo

@cuda.jit
def pauli_x_kernel(num_amps_per_rank, real, imag, target):
    num_tasks = num_amps_per_rank >>1

    size_half_block = 1 << target
    size_block     = 2 * size_half_block

    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return

    this_block   = this_task // size_half_block
    index_up     = this_block * size_block + this_task % size_half_block
    index_lo     = index_up + size_half_block

    # store current state vector values in temp variables
    state_real_up = real[index_up]
    state_imag_up = imag[index_up]

    real[index_up] = real[index_lo]
    imag[index_up] = imag[index_lo]

    real[index_lo] = state_real_up
    imag[index_lo] = state_imag_up

@cuda.jit
def pauli_y_kernel(num_amps_per_rank, real, imag, target, conjFac):
    size_half_block = 1 << target
    size_block      = 2 * size_half_block
    num_tasks       = num_amps_per_rank >> 1
    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return
    
    this_block = this_task // size_half_block
    index_up   = this_block * size_block + this_task % size_half_block
    index_lo   = index_up + size_half_block

    state_real_up = real[index_up]
    state_imag_up = imag[index_up]

    # update under +-{{0, -i}, {i, 0}}
    real[index_up] = conjFac * imag[index_lo]
    imag[index_up] = conjFac * -real[index_lo]
    real[index_lo] = conjFac * -state_imag_up
    imag[index_lo] = conjFac * state_real_up

@cuda.jit
def swap_kernel(num_amps_per_rank, real, imag, target_0, target_1):
    num_tasks = num_amps_per_rank >> 2
    this_task = cuda.grid(1)
    if (this_task >= num_tasks):
        return
    
    ind00 = insert_two_zero_bits(this_task, target_0, target_1)
    ind01 = flip_bit(ind00, target_0)
    ind10 = flip_bit(ind00, target_1)

    # extract statevec amplitudes 
    re01 = real[ind01]; im01 = imag[ind01]
    re10 = real[ind10]; im10 = imag[ind10]

    # swap 01 and 10 amps
    real[ind01] = re10; real[ind10] = re01
    imag[ind01] = im10; imag[ind10] = im01
    
@cuda.jit
def controlled_compact_unitary_kernel(num_amps_per_rank, real, imag, control_qubit, target_qubit, reals, imags):
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
        state_real_lo = real[index_lo]
        state_imag_lo = imag[index_lo]

        real[index_up] = reals[0]*state_real_up - imags[0]*state_imag_up - reals[1]*state_real_lo - imags[1]*state_imag_lo
        imag[index_up] = reals[0]*state_imag_up + imags[0]*state_real_up - reals[1]*state_imag_lo + imags[1]*state_real_lo
        real[index_lo] = reals[1]*state_real_up - imags[1]*state_imag_up + reals[0]*state_real_lo + imags[0]*state_imag_lo
        imag[index_lo] = reals[1]*state_imag_up + imags[1]*state_real_up + reals[0]*state_imag_lo - imags[0]*state_real_lo
    cuda.syncthreads()
    
@cuda.jit
def controlled_unitary_kernel(num_amps_per_rank, real, imag, control_qubit, target_qubit, ureal, uimag):
    size_half_block = 2**target_qubit
    size_block = size_half_block * 2
    num_task = num_amps_per_rank // 2
    
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
    
    control_bit = extract_bit(control_qubit, index_up)
    if control_bit:
        real[index_up] = ureal[0][0]*state_real_up - uimag[0][0]*state_imag_up + ureal[0][1]*state_real_lo - uimag[0][1]*state_imag_lo
        imag[index_up] = ureal[0][0]*state_imag_up + uimag[0][0]*state_real_up + ureal[0][1]*state_imag_lo + uimag[0][1]*state_real_lo
        real[index_lo] = ureal[1][0]*state_real_up - uimag[1][0]*state_imag_up + ureal[1][1]*state_real_lo - uimag[1][1]*state_imag_lo
        imag[index_lo] = ureal[1][0]*state_imag_up + uimag[1][0]*state_real_up + ureal[1][1]*state_imag_lo + uimag[1][1]*state_real_lo
    cuda.syncthreads()
    
@cuda.jit
def multi_controlled_two_qubit_unitary_kernel(num_amps_per_rank, real, imag, ctrl_mask, q1, q2, ureal, uimag):
    num_task = num_amps_per_rank // 2
    
    this_task = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if this_task>=num_task:
        return
  
    ind00 = insert_two_zero_bits(this_task, q1, q2)
    if (ctrl_mask and (ctrl_mask&ind00) != ctrl_mask):
        return
    
    ind01 = flip_bit(ind00, q1)
    ind10 = flip_bit(ind00, q2)
    ind11 = flip_bit(ind01, q2)
    
    re00 = real[ind00]
    im00 = imag[ind00]
    re01 = real[ind01] 
    im01 = imag[ind01]
    re10 = real[ind10]
    im10 = imag[ind10]
    re11 = real[ind11] 
    im11 = imag[ind11]
    
    real[ind00] = ureal[0][0]*re00 - uimag[0][0]*im00 + ureal[0][1]*re01 - uimag[0][1]*im01 + ureal[0][2]*re10 - uimag[0][2]*im10 + ureal[0][3]*re11 - uimag[0][3]*im11
    imag[ind00] = uimag[0][0]*re00 + ureal[0][0]*im00 + uimag[0][1]*re01 + ureal[0][1]*im01 + uimag[0][2]*re10 + ureal[0][2]*im10 + uimag[0][3]*re11 + ureal[0][3]*im11
        
    real[ind01] = ureal[1][0]*re00 - uimag[1][0]*im00 + ureal[1][1]*re01 - uimag[1][1]*im01 + ureal[1][2]*re10 - uimag[1][2]*im10 + ureal[1][3]*re11 - uimag[1][3]*im11
    imag[ind01] = uimag[1][0]*re00 + ureal[1][0]*im00 + uimag[1][1]*re01 + ureal[1][1]*im01 + uimag[1][2]*re10 + ureal[1][2]*im10 + uimag[1][3]*re11 + ureal[1][3]*im11
        
    real[ind10] = ureal[2][0]*re00 - uimag[2][0]*im00 + ureal[2][1]*re01 - uimag[2][1]*im01 + ureal[2][2]*re10 - uimag[2][2]*im10 + ureal[2][3]*re11 - uimag[2][3]*im11
    imag[ind10] = uimag[2][0]*re00 + ureal[2][0]*im00 + uimag[2][1]*re01 + ureal[2][1]*im01 + uimag[2][2]*re10 + ureal[2][2]*im10 + uimag[2][3]*re11 + ureal[2][3]*im11   
        
    real[ind11] = ureal[3][0]*re00 - uimag[3][0]*im00 + ureal[3][1]*re01 - uimag[3][1]*im01 + ureal[3][2]*re10 - uimag[3][2]*im10 + ureal[3][3]*re11 - uimag[3][3]*im11
    imag[ind11] = uimag[3][0]*re00 + ureal[3][0]*im00 + uimag[3][1]*re01 + ureal[3][1]*im01 + uimag[3][2]*re10 + ureal[3][2]*im10 + uimag[3][3]*re11 + ureal[3][3]*im11  
    cuda.syncthreads()

@cuda.jit
def find_prob_of_zero_kernel(num_amps_per_rank, real, imag, target, total_prob):
    num_task = num_amps_per_rank // 2
    size_half_block = 2**target
    size_block = size_half_block * 2

    this_task = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if this_task>=num_task:
        return
    
    this_block = this_task // size_half_block
    index = this_block*size_block + this_task%size_half_block
    
    total_prob[this_task] = real[index] * real[index] + imag[index] * imag[index]
    cuda.syncthreads()
    
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

    def amp(self, reals, imags, startindex, numamps):
        """Init amplitudes state"""
        orgreal = cuda.to_device(reals, stream=0)
        orgimag = cuda.to_device(imags, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(numamps / threads_per_block)
        amp_kernel[blocks_per_grid, threads_per_block](numamps, self.real, self.imag, orgreal, orgimag, startindex)
        
    def hadamard(self, target_qubit):
        """Apply hadamard gate.

        Args:
            target: target qubit.
        """
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        hadamard_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit)
    
    def phase_shift(self, target, angle):
        """Shift the phase between |0> and |1> of a single qubit by a given angle.

        Args:
            target: qubit to undergo a phase shift.
            angle:  amount by which to shift the phase in radians.
        """
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, cos_angle, sin_angle)

    def controlled_phase_shift(self, control, target, angle):
        """
        Controlled-Phase gate.

        Args:
            control: control qubit
            target: target qubit
            angle: amount by which to shift the phase in radians.
        """
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        controlled_phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control, target, cos_angle, sin_angle)

    def rotate(self, target, ureal, uimag):
        """Rotate gate."""
        self.apply_matrix2(target, ureal, uimag)
    
    def rotate_x(self, target, angle):
        """rx gate."""
        unit_axis = [1, 0, 0]
        self.rotate_around_axis(target, angle, unit_axis)

    def rotate_y(self, target, angle):
        """ry gate."""
        unit_axis = [0, 1, 0]
        self.rotate_around_axis(target, angle, unit_axis)

    def rotate_z(self, target, angle):
        """rz gate."""
        unit_axis = [0, 0, 1]
        self.rotate_around_axis(target, angle, unit_axis)
    
    def rotate_around_axis(self, target, angle, unit_axis):
        mag = math.sqrt(
            unit_axis[0] * unit_axis[0]
            + unit_axis[1] * unit_axis[1]
            + unit_axis[2] * unit_axis[2]
        )
        unit_vec = [unit_axis[0] / mag, unit_axis[1] / mag, unit_axis[2] / mag]

        alpha_real = math.cos(angle / 2.0)
        alpha_imag = -math.sin(angle / 2.0) * unit_vec[2]
        beta_real = math.sin(angle / 2.0) * unit_vec[1]
        beta_imag = -math.sin(angle / 2.0) * unit_vec[0]

        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        rotate_around_axis_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, alpha_real, alpha_imag, beta_real, beta_imag)

    def pauli_x(self, target):
        """The single-qubit Pauli-X gate."""
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        pauli_x_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target)

    def pauli_y(self, target):
        """The single-qubit Pauli-Y gate."""
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        pauli_y_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, 1)

    def pauli_z(self, target):
        """The single-qubit Pauli-Z gate."""
        cos_angle = -1
        sin_angle = 0
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, cos_angle, sin_angle)

    def s_gate(self, target):
        """The single-qubit S gate."""
        cos_angle = 0
        sin_angle = 1
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, cos_angle, sin_angle)

    def t_gate(self, target):
        """The single-qubit T gate."""
        cos_angle = 1 / math.sqrt(2)
        sin_angle = 1 / math.sqrt(2)
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, cos_angle, sin_angle)

    def sdg(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def tdg(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def sqrtx(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def sqrtswap(self, target_0, target_1, ureal, uimag):
        self.apply_matrix4(target_0, target_1, ureal, uimag)

    def swap(self, target_0, target_1):
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 2) / threads_per_block)
        swap_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_0, target_1)

    def control_not(self, control_qubit, target_qubit):
        """Control not gate"""
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        controlled_not_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control_qubit, target_qubit)

    def multi_controlled_multi_qubit_not(self, control_bits, num_control_bits, target_bits, num_target_bits):
        ctrl_mask = self.get_qubit_bit_mask(control_bits, num_control_bits)
        targ_mask = self.get_qubit_bit_mask(target_bits, num_target_bits)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        multi_controlled_multi_qubit_not_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, ctrl_mask, targ_mask)

    def get_qubit_bit_mask(self, qubits, numqubit):
        mask = 0
        for index in range(numqubit):
            mask = mask | (1 << qubits[index])
        return mask

    def cy(self, control, target):
        conj_factor = 1
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        cy_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control, target, conj_factor)

    def cz(self, control_bits, num_control_bits):
        mask = self.get_qubit_bit_mask(control_bits, num_control_bits)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        cz_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, mask)

    def u3(self, target_qubit, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit, orgreal, orgimag)

    def u2(self, target_qubit, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit, orgreal, orgimag)

    def u1(self, target_qubit, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit, orgreal, orgimag)
      
    def get_complex_pair_from_rotation(self, angle, axis):
        mag = math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])  
        unit_axis = [0] * 3
        unit_axis[0] = axis[0]/mag
        unit_axis[1] = axis[1]/mag
        unit_axis[2] = axis[2]/mag
        reals = [0] * 2
        imags = [0] * 2
        reals[0] =   math.cos(angle/2.0)
        imags[0] = - math.sin(angle/2.0)*unit_axis[2]
        reals[1] =   math.sin(angle/2.0)*unit_axis[1]
        imags[1] = - math.sin(angle/2.0)*unit_axis[0]
        return reals, imags

    def controlled_compact_unitary(self, control_qubit, target_qubit, angle, unit_axis):
        reals, imags = self.get_complex_pair_from_rotation(angle, unit_axis)
        orgreal = cuda.to_device(reals, stream=0)
        orgimag = cuda.to_device(imags, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        controlled_compact_unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control_qubit, target_qubit, orgreal, orgimag)

    def crx(self, control_qubit, target_qubit, angle):
        unit_axis = [1, 0, 0]
        self.controlled_compact_unitary(control_qubit, target_qubit, angle, unit_axis)
    
    def cry(self, control_qubit, target_qubit, angle):
        unit_axis = [0, 1, 0]
        self.controlled_compact_unitary(control_qubit, target_qubit, angle, unit_axis)

    def crz(self, control_qubit, target_qubit, angle):
        unit_axis = [0, 0, 1]
        self.controlled_compact_unitary(control_qubit, target_qubit, angle, unit_axis)

    def apply_matrix2(self, target_qubit, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target_qubit, orgreal, orgimag)
        
    def x1(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)

    def y1(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)

    def z1(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)

    def sqrtx(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)

    def sqrtxdg(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)
       
    def csqrtx(self, control_qubit, target_qubit, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        controlled_unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control_qubit, target_qubit, orgreal, orgimag)

    def apply_matrix4(self, target_qubit0, target_qubit1, ureal, uimag):
        ctrl_mask = 0
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 2) / threads_per_block)
        multi_controlled_two_qubit_unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, ctrl_mask, target_qubit0, target_qubit1, orgreal, orgimag)
        
    def cu1(self, target_qubit0, target_qubit1, ureal, uimag):
        self.apply_matrix4(target_qubit0, target_qubit1, ureal, uimag)

    def cu3(self, target_qubit0, target_qubit1, ureal, uimag):
        self.apply_matrix4(target_qubit0, target_qubit1, ureal, uimag)

    def cu(self, target_qubit0, target_qubit1, ureal, uimag):
        self.apply_matrix4(target_qubit0, target_qubit1, ureal, uimag)

    def cr(self, target_qubit0, target_qubit1, ureal, uimag):
        self.apply_matrix4(target_qubit0, target_qubit1, ureal, uimag)

    def iswap(self, target_qubit0, target_qubit1, ureal, uimag):
        self.apply_matrix4(target_qubit0, target_qubit1, ureal, uimag)

    def reset(self, target_qubit, ureal, uimag):
        self.apply_matrix2(target_qubit, ureal, uimag)
    
    def calc_prob_of_outcome(self, target, outcome):
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        outcome_prob_list = cuda.device_array(self.sim_cpu.num_amps_per_rank, numpy.float_)
        find_prob_of_zero_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, outcome_prob_list)
        outcome_prob = 0.0
        for temp in outcome_prob_list: 
            outcome_prob += temp
        if outcome == 1:
            outcome_prob = 1.0 - outcome_prob
        return outcome_prob
             
    def get_statevector(self):
        """
        Get the current state vector of probability amplitudes for a set of qubits
        """
        # todo better in float or ndarray
        real = self.real.copy_to_host()
        imag = self.imag.copy_to_host()
        state_list = []
        for i in range(self.sim_cpu.num_amps_per_rank):
            real = real[i]
            imag = imag[i]
            # TODO: need to improve.
            if real[i] > -1e-15 and real[i] < 1e-15:
                real = 0
            if imag[i] > -1e-15 and imag[i] < 1e-15:
                imag = 0
            state = str(real) + ", " + str(imag)
            state_list.append(state)
        return state_list
    

