"""Implementation of quantum compute simulator for gpu running mode."""
import math
import random
from typing import Union

import numpy
from numba import cuda, float32
from qutrunk.backends.local.sim_local import SimLocal, PauliOpType

REAL_EPS = 1e-13

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

@cuda.jit
def insert_zero_bits(number, targets, targets_num):
    cur_min = targets[0]
    prev_min = -1
    for n in range(targets_num):
        for t in range(targets_num):
            if targets[t]>prev_min and targets[t]<cur_min:
                cur_min = targets[t]
    
        number = insert_zero_bit(number, cur_min)
        
        prev_min = cur_min
        for t in range(targets_num):
            if targets[t] > cur_min:
                cur_min = targets[t]
                break
        
    return number
     
@cuda.jit
def multi_controlled_multi_qubit_unitary_kernel(num_amps_per_rank, real, imag, ctrl_mask, targets, d_ureal, d_uimag, d_amp_inds, d_real_amps, d_imag_amps, unum_rows):
    targets_num = len(targets)
    this_task = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    num_task = num_amps_per_rank >> targets_num
    if this_task>=num_task:
        return

    ind00 = insert_zero_bits(this_task, targets, targets_num)
    if ctrl_mask and (ctrl_mask&ind00) != ctrl_mask:
        return
    
    stride = cuda.gridDim.x*cuda.blockDim.x
    offset = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    
    ind = 0
    for i in range(unum_rows):
        ind = ind00
        for t in range(targets_num):
            if extract_bit(t, i):
                ind = flip_bit(ind, targets[t])
        
        d_amp_inds[i*stride+offset] = ind
        d_real_amps [i*stride+offset] = real[ind]
        d_imag_amps [i*stride+offset] = imag[ind]
    
    for r in range(unum_rows):
        ind = d_amp_inds[r*stride+offset]
        real[ind] = 0
        imag[ind] = 0
        for c in range(unum_rows):
            u_real_elem = d_ureal[c + r*unum_rows]
            u_imag_elem = d_uimag[c + r*unum_rows]
            real[ind] += d_real_amps[c*stride+offset]*u_real_elem - d_imag_amps[c*stride+offset]*u_imag_elem
            imag[ind] += d_real_amps[c*stride+offset]*u_imag_elem + d_imag_amps[c*stride+offset]*u_real_elem

@cuda.jit
def collapse_to_know_prob_outcome_kernel(num_amps_per_rank, real, imag, target, outcome, outcome_prob):
    num_task = num_amps_per_rank >> 1
    size_half_block = 2**target
    size_block = size_half_block * 2

    renorm=1/math.sqrt(outcome_prob)
   
    this_task = cuda.grid(1)
    if (this_task >= num_task):
        return

    this_block = this_task / size_half_block
    index      = this_block * size_block + this_task % size_half_block

    if (outcome == 0):
        real[index] = real[index] * renorm
        imag[index] = imag[index] * renorm

        real[index+size_half_block] = 0
        imag[index+size_half_block] = 0
    elif (outcome == 1):
        real[index] = 0
        imag[index] = 0

        real[index+size_half_block] = real[index+size_half_block] * renorm
        imag[index+size_half_block] = imag[index+size_half_block] * renorm
  
@cuda.jit
def calc_inner_product_local_kernel(self, num_amps_per_rank, bra_real, bra_imag, ket_real, ket_imag, prod_reals, prod_imags):
    index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if index>=num_amps_per_rank:
        return
    
    bra_re = bra_real[index]
    bra_im = bra_imag[index]
    ket_re = ket_real[index]
    ket_im = ket_imag[index]
    prod_reals[index] += bra_re * ket_re + bra_im * ket_im
    prod_imags[index] += bra_re * ket_im - bra_im * ket_re
      
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
    
    def ch(self, control, target, ureal, uimag):
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        controlled_unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, control, target, orgreal, orgimag)
            
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

    def cswap(self, control_qubit, target_qubit0, target_qubit1, ureal, uimag):
        ctrl_mask = 1 << control_qubit
        orgreal = cuda.to_device(ureal, stream=0)
        orgimag = cuda.to_device(uimag, stream=0)
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 2) / threads_per_block)
        multi_controlled_two_qubit_unitary_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, ctrl_mask, target_qubit0, target_qubit1, orgreal, orgimag)
        
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
            _real = real[i]
            _imag = imag[i]
            # TODO: need to improve.
            if _real > -1e-15 and _real < 1e-15:
                _real = 0
            if _imag > -1e-15 and _imag < 1e-15:
                _imag = 0
            state = str(_real) + ", " + str(_imag)
            state_list.append(state)
        return state_list

    def multi_controlled_multi_qubit_unitary(self, ctrl_mask, targets, ureal, uimag):        
        targets_num = len(targets)
        threads_per_cuda_block = 128
        cuda_blocks = math.ceil((self.sim_cpu.num_amps_per_rank>>targets_num)/threads_per_cuda_block)
        
        unum_rows = 1 << targets_num
        d_ureal = cuda.device_array(unum_rows*unum_rows, numpy.float_)
        d_uimag = cuda.device_array(unum_rows*unum_rows, numpy.float_)
        i = 0
        for r in range(unum_rows):
            for c in range(unum_rows):
                d_ureal[i] = ureal[r][c]
                d_uimag[i] = uimag[r][c]
                i += 1
        
        grid_size = threads_per_cuda_block * cuda_blocks
        d_amp_inds = cuda.device_array(unum_rows*grid_size, numpy.int64)
        d_real_amps = cuda.device_array(unum_rows*grid_size, numpy.float_)
        d_imag_amps = cuda.device_array(unum_rows*grid_size, numpy.float_)
        d_targets = cuda.to_device(targets, stream=0)
        multi_controlled_multi_qubit_unitary_kernel[cuda_blocks, threads_per_cuda_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, ctrl_mask, d_targets, d_ureal, d_uimag, d_amp_inds, d_real_amps, d_imag_amps, unum_rows)

    def apply_multi_controlled_matrix_n(self, controls, targets, ureal, uimag):
        controls_num = len(controls)
        ctrl_mask = self.get_qubit_bit_mask(controls, controls_num)
        self.multi_controlled_multi_qubit_unitary(ctrl_mask, targets, ureal, uimag)
    
    def apply_matrix_n(self, targets, ureal, uimag):
        ctrl_mask = 0
        self.multi_controlled_multi_qubit_unitary(ctrl_mask, targets, ureal, uimag)
    
    def matrix(self, controls, targets, ureal, uimag):
        self.apply_multi_controlled_matrix_n(controls, targets, ureal, uimag)

    def measure(self, target):
        zero_prob = self.calc_prob_of_outcome(target, 0)
        outcome, outcome_prob = self.generate_measure_outcome(zero_prob)
        self.collapse_to_know_prob_outcome(target, outcome, outcome_prob)
        return outcome

    def generate_measure_outcome(self, zero_prob):
        outcome = 0
        if zero_prob < REAL_EPS:
            outcome = 1
        elif (1 - zero_prob) < REAL_EPS:
            outcome = 0
        else:
            outcome = 1 if (random.random() > zero_prob) else 0

        outcome_prob = zero_prob if (outcome == 0) else (1 - zero_prob)
        return outcome, outcome_prob

    def collapse_to_know_prob_outcome(self, target, outcome, outcome_prob):
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        collapse_to_know_prob_outcome_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, self.real, self.imag, target, outcome, outcome_prob)
    def paulix_local(self, reals, imags, target):
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        pauli_x_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, reals, imags, target)

    def pauliy_local(self, reals, imags, target):
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        pauli_y_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, reals, imags, target, 1)

    def pauliz_local(self, reals, imags, target):
        cos_angle = -1
        sin_angle = 0
        threads_per_block = 128
        blocks_per_grid = math.ceil((self.sim_cpu.num_amps_per_rank >> 1) / threads_per_block)
        phase_shift_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, reals, imags, target, cos_angle, sin_angle)

    def calc_inner_product_local(self, bra_real, bra_imag, ket_real, ket_imag):
        prod_reals = cuda.device_array(self.sim_cpu.num_amps_per_rank, numpy.float_)
        prod_imags = cuda.device_array(self.sim_cpu.num_amps_per_rank, numpy.float_)
        threads_per_block = 128
        blocks_per_grid = math.ceil(self.sim_cpu.num_amps_per_rank / threads_per_block)
        calc_inner_product_local_kernel[blocks_per_grid, threads_per_block](self.sim_cpu.num_amps_per_rank, bra_real, bra_imag, ket_real, ket_imag, prod_reals, prod_imags)
        inner_prod_real = 0
        inner_prod_imag = 0
        for index in range(self.sim_cpu.num_amps_per_rank):
            inner_prod_real += prod_reals[index]
            inner_prod_imag += prod_imags[index]
        return (inner_prod_real, inner_prod_imag)
    
    def get_expec_pauli_prod(self, pauli_prod_list):
        """
        Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: a list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            the expected value of a product of Pauli operators.
        """
        work_real = cuda.to_device(self.real, stream=0)
        work_imag = cuda.to_device(self.imag, stream=0)
        for pauli_op in pauli_prod_list:
            op_type = pauli_op["oper_type"]
            if op_type == PauliOpType.PAULI_X.value:
                self.paulix_local(work_real, work_imag, pauli_op["target"])
            elif op_type == PauliOpType.PAULI_Y.value:
                self.pauliy_local(work_real, work_imag, pauli_op["target"])
            elif op_type == PauliOpType.PAULI_Z.value:
                self.pauliz_local(work_real, work_imag, pauli_op["target"])

        real, imag = self.calc_inner_product_local(
            work_real, work_imag, self.real, self.imag
        )
        return real

    def get_expec_pauli_sum(self, oper_type_list, term_coeff_list):
        """
        Computes the expected value of a sum of products of Pauli operators.

        Args:
            oper_type_list: a list of the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) \
                of all Paulis involved in the products of terms. A Pauli must be specified \
                for each qubit in the register, in every term of the sum.
            term_coeff_list: the coefficients of each term in the sum of Pauli products.

        Returns:
            the expected value of a sum of products of Pauli operators. 
        """
        """Computes the expected value of a sum of products of Pauli operators."""
        num_qb = self.sim_cpu.qubits
        targs = []
        for q in range(num_qb):
            targs.append(q)

        value = 0
        idx = 0
        num_sum_terms = len(term_coeff_list)
        for t in range(num_sum_terms):
            pauli_prod_list = []
            for i in range(num_qb):
                temp = {}
                temp["oper_type"] = oper_type_list[idx]
                idx += 1
                temp["target"] = targs[i]
                pauli_prod_list.append(temp)
            value += term_coeff_list[t] * self.get_expec_pauli_prod(pauli_prod_list)

        return value
