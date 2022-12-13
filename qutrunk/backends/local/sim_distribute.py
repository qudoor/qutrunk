"""Implementation of quantum compute simulator for distributed running mode."""
import math
import numpy

from mpi4py import MPI
from .sim_cpu import SimCpu
from typing import List, Union

REAL_EPS = 1e-13

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
        self.sim_cpu = SimCpu()

        total_num_amps = 2**num_qubits
        num_amps_per_rank = total_num_amps // self.env.num_ranks
        self.reg.state_vec.real = [0] * num_amps_per_rank
        self.reg.state_vec.imag = [0] * num_amps_per_rank

        if self.env.num_ranks > 1:
            self.reg.pair_state_vec.real = [0] * num_amps_per_rank
            self.reg.pair_state_vec.imag = [0] * num_amps_per_rank

        self.reg.num_amps_total = total_num_amps
        self.reg.num_amps_per_chunk = num_amps_per_rank
        self.reg.chunk_id = self.env.rank
        self.reg.num_chunks = self.env.num_ranks
        self.reg.num_qubits_in_state_vec = num_qubits

        self.sim_cpu.real = self.reg.state_vec.real
        self.sim_cpu.imag = self.reg.state_vec.imag
        self.sim_cpu.qubits = self.reg.num_qubits_in_state_vec
        self.sim_cpu.num_amps_per_rank = self.reg.num_amps_per_chunk
    
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
    
    def __half_matrix_block_fits_in_chunk(self, chunkSize, target_qubit):
        size_half_block = (1 << target_qubit)
        return 1 if chunkSize > size_half_block else 0

    def __chunk_isupper(self, chunk_id, chunk_size, target_qubit):
        size_half_block = 1 << target_qubit
        size_block = size_half_block * 2
        pos_in_block = (chunk_id * chunk_size) % size_block
        return pos_in_block < size_half_block

    def __get_chunk_pair_id(self, chunk_isupper, chunk_id, chunk_size, target_qubit):
        size_half_block = 1 << target_qubit
        chunks_per_half_block = size_half_block // chunk_size
        if chunk_isupper:
            return chunk_id + chunks_per_half_block
        else:
            return chunk_id - chunks_per_half_block

    def __exchange_state(self, target_qubit):
        # need to get corresponding chunk of state vector from other rank
        rank_isupper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
        pair_rank = self.__get_chunk_pair_id(rank_isupper, self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
        # get corresponding values from my pair
        self.__exchange_state_vectors(pair_rank)
        return rank_isupper

    def __exchange_state_vectors(self, pair_rank):
        send_buffer = numpy.zeros(1, dtype='float')
        recv_buffer = numpy.zeros(1, dtype='float')
        # send my state vector to pairRank's qureg.pairStateVec
        # receive pairRank's state vector into qureg.pairStateVec
        for i in range(self.reg.num_amps_per_chunk):
            send_buffer[0] = self.reg.state_vec.real[i]
            self.comm.Sendrecv(send_buffer, pair_rank, 0,
                    recv_buffer, pair_rank, 0)
            self.reg.pair_state_vec.real[i] = recv_buffer[0]

            send_buffer[0] = self.reg.state_vec.imag[i]
            self.comm.Sendrecv(send_buffer, pair_rank, 0,
                    recv_buffer, pair_rank, 0)
            self.reg.pair_state_vec.imag[i] = recv_buffer[0]

    def __is_chunk_to_skip_in_find_prob_zero(self, chunk_id, chunk_size, measure_qubit):
        size_half_block = 1 << measure_qubit
        num_chunks_to_skip = size_half_block // chunk_size
        # calculate probability by summing over numChunksToSkip, then skipping numChunksToSkip, etc
        bit_to_check = chunk_id and num_chunks_to_skip
        return bit_to_check

    def __find_Prob_of_zero_distributed(self):
        total_prob = 0.0
        for this_task in range(self.reg.num_amps_per_chunk):
            total_prob = total_prob + self.reg.state_vec.real[this_task] * self.reg.state_vec.real[this_task] + self.reg.state_vec.imag[this_task] * self.reg.state_vec.imag[this_task]

        return total_prob

    def __collapse_to_known_prob_outcome_distributed_renorm(self, measure_qubit, total_probability):
        renorm=1/math.sqrt(total_probability)
        for this_task in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec.real[this_task] = self.reg.state_vec.real[this_task] * renorm
            self.reg.state_vec.imag[this_task] = self.reg.state_vec.imag[this_task] * renorm

    def __collapse_to_outcome_distributed_set_zero(self):
        for this_task in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec.real[this_task] = 0
            self.reg.state_vec.real[this_task] = 0

    def __calc_prob_of_outcome(self, measure_qubit, outcome):
        skip_values_within_rank = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, measure_qubit)
        if skip_values_within_rank:
            state_prob = self.sim_cpu.find_prob_of_zero(measure_qubit)
        else:
            if not self.__is_chunk_to_skip_in_find_prob_zero(self.reg.chunk_id, self.reg.num_amps_per_chunk, measure_qubit):
                state_prob = self.__find_Prob_of_zero_distributed()
            else:
                state_prob = 0

        state_prob_buffer = numpy.zeros(1, dtype='float') + state_prob
        total_state_prob_buffer = numpy.zeros(1, dtype='float')
        self.comm.Allreduce(state_prob_buffer, total_state_prob_buffer)
        total_state_prob = total_state_prob_buffer[0]
        if outcome == 1:
            total_state_prob = 1.0 - total_state_prob
        return total_state_prob

    def __collapse_to_know_prob_outcome(self, measure_qubit, outcome, total_state_prob):
        skip_values_within_rank = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, measure_qubit)
        if skip_values_within_rank:
            self.sim_cpu.collapse_to_know_prob_outcome(measure_qubit, outcome, total_state_prob)
        else:
            if not self.__is_chunk_to_skip_in_find_prob_zero(self.reg.chunk_id, self.reg.num_amps_per_chunk, measure_qubit):
                # chunk has amps for q=0
                if outcome == 0:
                    self.__collapse_to_known_prob_outcome_distributed_renorm(measure_qubit, total_state_prob)
                else:
                    self.__collapse_to_outcome_distributed_set_zero()
            else:
                # chunk has amps for q=1
                if outcome == 1:
                    self.__collapse_to_known_prob_outcome_distributed_renorm(measure_qubit, total_state_prob)
                else:
                    self.__collapse_to_outcome_distributed_set_zero()

    def measure(self, target):
        zero_prob = self.__calc_prob_of_outcome(target, 0)
        outcome, outcome_prob = self.sim_cpu.generate_measure_outcome(zero_prob)
        self.__collapse_to_know_prob_outcome(target, outcome, outcome_prob)
        return outcome

    def hadamard(self, target_qubit):
        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_qubit)

        # rank's chunk is in upper half of block 
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.hadamard(target_qubit)
        else:
            # exchange state vectors between ranks
            rank_isupper = self.__exchange_state(target_qubit)
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

    def control_not(self, control_qubit, target_qubit):
        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_qubit)
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.control_not(control_qubit, target_qubit)
        else:
            # exchange state vectors between ranks
            self.__exchange_state(target_qubit)
            self.__controlled_not_distributed(control_qubit, self.reg.pair_state_vec, self.reg.state_vec)

    def __controlled_not_distributed(self, control_qubit, state_vec_in, state_vec_out):
        for this_task in range(self.reg.num_amps_per_chunk):
            control_bit = self.sim_cpu.extract_bit(control_qubit, this_task + self.reg.chunk_id * self.reg.num_amps_per_chunk)
            if control_bit:
                state_vec_out.real[this_task] = state_vec_in.real[this_task]
                state_vec_out.imag[this_task] = state_vec_in.imag[this_task]

    def phase_shift(self, target, angle):
        """Shift the phase between |0> and |1> of a single qubit by a given angle.

        Args:
            target: qubit to undergo a phase shift.
            angle:  amount by which to shift the phase in radians.
        """
        self.sim_cpu.phase_shift(target, angle)

    def controlled_phase_shift(self, ctrl, target, angle):
        """
        Controlled-Phase gate.

        Args:
            ctrl: control qubit
            target: target qubit
            angle: amount by which to shift the phase in radians.
        """
        self.sim_cpu.controlled_phase_shift(ctrl, target, angle)

    def rotate(self, target, ureal, uimag):
        """Rotate gate."""
        self.__apply_matrix2(target, ureal, uimag)

    def __apply_matrix2(self, target_qubit, ureal, uimag):
        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_qubit)
        rot1 = complex()
        rot2 = complex()
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.apply_matrix2(target_qubit, ureal, uimag)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_isupper = self.__exchange_state()
            self.__get_rot_angle_from_unitary_matrix(rank_isupper, rot1, rot2, ureal, uimag)

            # this rank's values are either in the upper of lower half of the block. 
            # send values to compactUnitaryDistributed in the correct order
            if rank_isupper:
                self.__apply_matrix2_distributed(rot1, rot2, self.reg.state_vec, self.reg.pair_state_vec, self.reg.state_vec)
            else:
                self.__apply_matrix2_distributed(rot1, rot2, self.reg.pair_state_vec, self.reg.state_vec, self.reg.state_vec)

    def __get_rot_angle_from_unitary_matrix(self, rank_isupper, rot1, rot2, ureal, uimag):
        if rank_isupper:
            rot1.real = ureal[0][0]
            rot1.imag = uimag[0][0]
            rot2.real = ureal[0][1]
            rot2.imag = uimag[0][1]
        else:
            rot1.real = ureal[1][0]
            rot1.imag = uimag[1][0]
            rot2.real = ureal[1][1]
            rot2.imag = uimag[1][1]

    def __apply_matrix2_distributed(self, rot1, rot2, state_vec_up, state_vec_lo, state_vec_out):
        for this_task in range(self.reg.num_amps_per_chunk):
            # store current state vector values in temp variables
            state_real_up = state_vec_up.real[this_task]
            state_imag_up = state_vec_up.imag[this_task]

            state_real_lo = state_vec_lo.real[this_task]
            state_imag_lo = state_vec_lo.imag[this_task]

            state_vec_out[this_task] = rot1.real * state_real_up - rot1.imag * state_imag_up \
                + rot2.real * state_real_lo - rot2.imag * state_imag_lo
            state_vec_out[this_task] = rot1.real * state_imag_up + rot1.imag * state_real_up \
                + rot2.real * state_imag_lo + rot2.imag * state_real_lo

    def __validate_target(self, target_qubit: Union[int, list]):
        if isinstance(target_qubit, int): 
            if target_qubit < 0 or target_qubit >= self.reg.num_qubits_in_state_vec:
                raise ValueError("Invalid target qubit. Must be >=0 and <numQubits.")
        elif isinstance(target_qubit, list):
            for target in target_qubit:
                self.__validate_target(target)
        else:
            raise TypeError("qubits parameter should be type of int or list.")
        
    def __validate_multi_qubit_matrix_fits_in_node(self, num_targets: int):
        if self.reg.num_amps_per_chunk < (1 << num_targets):
            raise ValueError("The specified matrix targets too many qubits; the batches of amplitudes to modify cannot all fit in a single distributed node's memory allocation.")

    def __validate_matrix(self, ureal, uimag, row_num: int, column_num: int):
        if len(ureal) != row_num or len(ureal[0]) != column_num or len(uimag) != row_num or len(uimag[0]) != column_num:
            raise ValueError("expected arrary length of ureal or uimag")

    def __mask_contains_bit(self, mask: int, bit_ind: int):
        return mask & (1 << bit_ind)
   
    def __flip_bit(self, number: int, bit_ind: int):
        return self.sim_cpu.flip_bit(number, bit_ind)

    def __insert_zero_bit(self, number: int, index: int):
        return self.sim_cpu.insert_zero_bit(number, index)

    def __insert_two_zero_bits(self, number: int, bit1: int, bit2: int):
        return self.sim_cpu.insert_two_zero_bits(number, bit1, bit2)

    def __extract_bit(self, ctrl: int, index: int):
        return self.sim_cpu.__extract_bit(ctrl, index)

    def __is_odd_parity(number: int, qb1: int, qb2: int):
        return self.__extract_bit(qb1, number) != self.__extract_bit(qb2, number)
    
    def __swap_qubit_amps_local(self, qb1: int, qb2: int):
        num_tasks = self.reg.num_amps_per_chunk >> 2
        for this_task in range(num_tasks):
            ind00 = self.__insert_two_zero_bits(this_task, qb1, qb2)
            ind01 = self.__flip_bit(ind00, qb1)
            ind10 = self.__flip_bit(ind00, qb2)

            # extract statevec amplitudes 
            re01 = self.reg.state_vec.real[ind01]
            im01 = self.reg.state_vec.imag[ind01]
            re10 = self.reg.state_vec.real[ind10]
            im10 = self.reg.state_vec.imag[ind10]

            # swap 01 and 10 amps
            self.reg.state_vec.real[ind01] = re10
            self.reg.state_vec.real[ind10] = re01
            self.reg.state_vec.imag[ind01] = im10
            self.reg.state_vec.imag[ind10] = im01

    def __get_global_ind_of_odd_parity_in_chunk(self, qb1: int, qb2: int):
        chunk_start_ind = self.reg.num_amps_per_chunk * self.reg.chunk_id
        chunk_end_ind = chunk_start_ind + self.reg.num_amps_per_chunk
        
        if (self.__extract_bit(qb1, chunk_start_ind) != self.__extract_bit(qb2, chunk_start_ind)):
            return chunk_start_ind
        
        odd_parity_ind = self.__flip_bit(chunk_start_ind, qb1)
        if (odd_parity_ind >= chunk_start_ind and odd_parity_ind < chunk_end_ind):
            return odd_parity_ind
            
        odd_parity_ind = self.__flip_bit(chunk_start_ind, qb2)
        if (odd_parity_ind >= chunk_start_ind and odd_parity_ind < chunk_end_ind):
            return odd_parity_ind
        
        return -1
    
    def __swap_qubit_amps_distributed(self, pair_rank, qb1: int, qb2: int):
        global_start_ind = self.reg.chunk_id * self.reg.num_amps_per_chunk
        pair_global_start_ind = pair_rank * self.reg.num_amps_per_chunk
        for local_ind in range(self.reg.num_amps_per_chunk):
            global_ind = global_start_ind + local_ind
            if self.__is_odd_parity(global_ind, qb1, qb2):
                
                pair_global_ind = self.__flip_bit(self.__flip_bit(global_ind, qb1), qb2)
                pair_local_ind = pair_global_ind - pair_global_start_ind
                
                self.reg.state_vec.real[local_ind] = self.reg.pair_state_vec.real[pair_local_ind]
                self.reg.state_vec.imag[local_ind] = self.reg.pair_state_vec.imag [pair_local_ind]
    
    def __swap_qubit_amps(self, qb1: int, qb2: int):
        # perform locally if possible 
        qb_big = qb1 if qb1 > qb2 else qb2
        if (self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, qb_big)):
            return self.__swap_qubit_amps_local(qb1, qb2)
        
        # do nothing if this node contains no amplitudes to swap
        odd_parity_global_ind = self.__get_global_ind_of_odd_parity_in_chunk(qb1, qb2)
        if odd_parity_global_ind == -1:
            return
        
        # determine and swap amps with pair node
        pair_rank = self.__flip_bit(self.__flip_bit(odd_parity_global_ind, qb1), qb2) // self.reg.num_amps_per_chunk
        self.__exchange_state_vectors(pair_rank)
        self.__swap_qubit_amps_distributed(pair_rank, qb1, qb2)
        
    def __multi_controlled_two_qubit_unitary_local(self, ctrl_mask: int, q1: int, q2: int, ureal, uimag):
        global_ind_start = self.reg.chunk_id * self.reg.num_amps_per_chunk
        num_tasks = self.reg.num_amps_per_chunk >> 2
        for this_task in range(num_tasks):
            # determine ind00 of |..0..0..>
            ind00 = self.__insert_two_zero_bits(this_task, q1, q2)
            
            # skip amplitude if controls aren't in 1 state (overloaded for speed)
            this_global_ind00 = ind00 + global_ind_start
            if ctrl_mask and ((ctrl_mask & this_global_ind00) != ctrl_mask):
                continue
            
            # inds of |..0..1..>, |..1..0..> and |..1..1..>
            ind01 = self.__flip_bit(ind00, q1)
            ind10 = self.__flip_bit(ind00, q2)
            ind11 = self.__flip_bit(ind01, q2)

            # extract statevec amplitudes 
            re00 = self.reg.state_vec.real[ind00]
            im00 = self.reg.state_vec.imag[ind00]
            re01 = self.reg.state_vec.real[ind01]
            im01 = self.reg.state_vec.imag[ind01]
            re10 = self.reg.state_vec.real[ind10]
            im10 = self.reg.state_vec.imag[ind10]
            re11 = self.reg.state_vec.real[ind11]
            im11 = self.reg.state_vec.imag[ind11]

			# apply u * {amp00, amp01, amp10, amp11}
            self.reg.state_vec.real[ind00] = \
                ureal[0][0]*re00 - uimag[0][0]*im00 + \
                ureal[0][1]*re01 - uimag[0][1]*im01 + \
                ureal[0][2]*re10 - uimag[0][2]*im10 + \
                ureal[0][3]*re11 - uimag[0][3]*im11
            self.reg.state_vec.imag[ind00] = \
                uimag[0][0]*re00 + ureal[0][0]*im00 + \
                uimag[0][1]*re01 + ureal[0][1]*im01 + \
                uimag[0][2]*re10 + ureal[0][2]*im10 + \
                uimag[0][3]*re11 + ureal[0][3]*im11
                
            self.reg.state_vec.real[ind01] = \
                ureal[1][0]*re00 - uimag[1][0]*im00 + \
                ureal[1][1]*re01 - uimag[1][1]*im01 + \
                ureal[1][2]*re10 - uimag[1][2]*im10 + \
                ureal[1][3]*re11 - uimag[1][3]*im11
            self.reg.state_vec.imag[ind01] = \
                uimag[1][0]*re00 + ureal[1][0]*im00 + \
                uimag[1][1]*re01 + ureal[1][1]*im01 + \
                uimag[1][2]*re10 + ureal[1][2]*im10 + \
                uimag[1][3]*re11 + ureal[1][3]*im11
                
            self.reg.state_vec.real[ind10] = \
                ureal[2][0]*re00 - uimag[2][0]*im00 + \
                ureal[2][1]*re01 - uimag[2][1]*im01 + \
                ureal[2][2]*re10 - uimag[2][2]*im10 + \
                ureal[2][3]*re11 - uimag[2][3]*im11
            self.reg.state_vec.imag[ind10] = \
                uimag[2][0]*re00 + ureal[2][0]*im00 + \
                uimag[2][1]*re01 + ureal[2][1]*im01 + \
                uimag[2][2]*re10 + ureal[2][2]*im10 + \
                uimag[2][3]*re11 + ureal[2][3]*im11  
                
            self.reg.state_vec.real[ind11] = \
                ureal[3][0]*re00 - uimag[3][0]*im00 + \
                ureal[3][1]*re01 - uimag[3][1]*im01 + \
                ureal[3][2]*re10 - uimag[3][2]*im10 + \
                ureal[3][3]*re11 - uimag[3][3]*im11
            self.reg.state_vec.imag[ind11] = \
                uimag[3][0]*re00 + ureal[3][0]*im00 + \
                uimag[3][1]*re01 + ureal[3][1]*im01 + \
                uimag[3][2]*re10 + ureal[3][2]*im10 + \
                uimag[3][3]*re11 + ureal[3][3]*im11
                
    def __multi_controlled_two_qubit_unitary(self, ctrl_mask: int, q1: int, q2: int, ureal, uimag):
        q1_fits_in_node = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, q1)
        q2_fits_in_node = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, q2)
        
        if q1_fits_in_node and q2_fits_in_node:  
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, q1, q2, ureal, uimag)
        elif q1_fits_in_node:
            qswap = (q1-1) if q1 > 0 else (q1+1)
            if self.__mask_contains_bit(ctrl_mask, qswap):
                ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, q2), qswap)
            
            self.__swap_qubit_amps(q2, qswap)
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, q1, qswap, ureal, uimag)
            self.__swap_qubit_amps(q2, qswap)
        elif q2_fits_in_node:
            qswap = (q2-1) if q2 > 0 else (q2+1)
            if self.__mask_contains_bit(ctrl_mask, qswap):
                ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, q1), qswap)
            
            self.__swap_qubit_amps(q1, qswap)
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, q2, qswap, ureal, uimag)
            self.__swap_qubit_amps(q1, qswap)
        else:
            swap1 = 0
            swap2 = 1
            if self.__mask_contains_bit(ctrl_mask, swap1):
                ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, swap1), q1)
            if self.__mask_contains_bit(ctrl_mask, swap2):
                ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, swap2), q2)
            
            self.__swap_qubit_amps(q1, swap1)
            self.__swap_qubit_amps(q2, swap2)
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, swap1, swap2, ureal, uimag)
            self.__swap_qubit_amps(q1, swap1)
            self.__swap_qubit_amps(q2, swap2)
            
    def __apply_matrix4(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__validate_multi_qubit_matrix_fits_in_node(2)
        self.__multi_controlled_two_qubit_unitary(0, target_qubit1, target_qubit2, ureal, uimag)