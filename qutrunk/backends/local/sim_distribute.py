"""Implementation of quantum compute simulator for distributed running mode."""
import math
from typing import Union

import numpy
from mpi4py import MPI

from .sim_local import SimCpu, PauliOpType

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
        self.sim_cpu = None
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

        total_num_amps = 2 ** num_qubits
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
        self.sim_cpu.chunk_id = self.reg.chunk_id

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

    def __half_matrix_block_fits_in_chunk(self, chunk_size, target_qubit):
        size_half_block = (1 << target_qubit)
        return 1 if (chunk_size > size_half_block) else 0

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
        # print(self.reg.chunk_id, self.reg.state_vec.real, self.reg.state_vec.imag)
        self.reg.pair_state_vec = self.comm.sendrecv(self.reg.state_vec, pair_rank, source=pair_rank)

    def __is_chunk_to_skip_in_find_prob_zero(self, chunk_id, chunk_size, measure_qubit):
        size_half_block = (1 << measure_qubit)
        num_chunks_to_skip = size_half_block // chunk_size
        # calculate probability by summing over numChunksToSkip, then skipping numChunksToSkip, etc
        bit_to_check = chunk_id & num_chunks_to_skip
        return bit_to_check

    def __find_Prob_of_zero_distributed(self):
        total_prob = 0.0
        for this_task in range(self.reg.num_amps_per_chunk):
            total_prob += (self.reg.state_vec.real[this_task] * self.reg.state_vec.real[this_task] + self.reg.state_vec.imag[this_task] * self.reg.state_vec.imag[this_task])

        return total_prob

    def __collapse_to_known_prob_outcome_distributed_renorm(self, measure_qubit, total_probability):
        renorm = 1 / math.sqrt(total_probability)
        for this_task in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec.real[this_task] = self.reg.state_vec.real[this_task] * renorm
            self.reg.state_vec.imag[this_task] = self.reg.state_vec.imag[this_task] * renorm

    def __collapse_to_outcome_distributed_set_zero(self):
        for this_task in range(self.reg.num_amps_per_chunk):
            self.reg.state_vec.real[this_task] = 0
            self.reg.state_vec.imag[this_task] = 0

    def __calc_prob_of_outcome(self, measure_qubit, outcome):
        skip_values_within_rank = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, measure_qubit)
        if skip_values_within_rank:
            state_prob = self.sim_cpu.find_prob_of_zero(measure_qubit)
        else:
            if not self.__is_chunk_to_skip_in_find_prob_zero(self.reg.chunk_id, self.reg.num_amps_per_chunk, measure_qubit):
                state_prob = self.__find_Prob_of_zero_distributed()
            else:
                state_prob = 0

        total_state_prob = self.comm.allreduce(state_prob, MPI.SUM)
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
        if self.reg.chunk_id == 0:
            outcome, outcome_prob = self.sim_cpu.generate_measure_outcome(zero_prob)
        else:
            outcome = None
            outcome_prob = None

        outcome = self.comm.bcast(outcome, root=0)
        outcome_prob = self.comm.bcast(outcome_prob, root=0)

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
                self.__hadamard_distributed(self.reg.state_vec,  # upper
                                            self.reg.pair_state_vec,  # lower
                                            self.reg.state_vec, rank_isupper)  # output
            else:
                self.__hadamard_distributed(self.reg.pair_state_vec,  # upper
                                            self.reg.state_vec,  # lower
                                            self.reg.state_vec, rank_isupper)  # output

    def __hadamard_distributed(self, state_vec_up, state_vec_lo, state_vec_out, update_upper):
        sign = 1 if update_upper else -1
        rec_root2 = 1.0 / math.sqrt(2)

        for this_task in range(self.reg.num_amps_per_chunk):
            # store current state vector values in temp variables
            state_real_up = state_vec_up.real[this_task]
            state_imag_up = state_vec_up.imag[this_task]

            state_real_lo = state_vec_lo.real[this_task]
            state_imag_lo = state_vec_lo.imag[this_task]

            state_vec_out.real[this_task] = rec_root2 * (state_real_up + sign * state_real_lo)
            state_vec_out.imag[this_task] = rec_root2 * (state_imag_up + sign * state_imag_lo)

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
            control_bit = self.sim_cpu.extract_bit(control_qubit,
                                                   this_task + self.reg.chunk_id * self.reg.num_amps_per_chunk)
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

        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.apply_matrix2(target_qubit, ureal, uimag)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_isupper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
            rot1, rot2 = self.__get_rot_angle_from_unitary_matrix(rank_isupper, ureal, uimag)

            pair_rank = self.__get_chunk_pair_id(rank_isupper, self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
            self.__exchange_state_vectors(pair_rank)

            # this rank's values are either in the upper of lower half of the block.
            # send values to compactUnitaryDistributed in the correct order
            if rank_isupper:
                self.__apply_matrix2_distributed(rot1, rot2, self.reg.state_vec, self.reg.pair_state_vec,
                                                 self.reg.state_vec)
            else:
                self.__apply_matrix2_distributed(rot1, rot2, self.reg.pair_state_vec, self.reg.state_vec,
                                                 self.reg.state_vec)

    def __get_rot_angle_from_unitary_matrix(self, rank_isupper, ureal, uimag):
        if rank_isupper:
            rot1 = complex(ureal[0][0], uimag[0][0])
            rot2 = complex(ureal[0][1], uimag[0][1])
        else:
            rot1 = complex(ureal[1][0], uimag[1][0])
            rot2 = complex(ureal[1][1], uimag[1][1])
        return rot1, rot2

    def __apply_matrix2_distributed(self, rot1, rot2, state_vec_up, state_vec_lo, state_vec_out):
        for this_task in range(self.reg.num_amps_per_chunk):
            # store current state vector values in temp variables
            state_real_up = state_vec_up.real[this_task]
            state_imag_up = state_vec_up.imag[this_task]

            state_real_lo = state_vec_lo.real[this_task]
            state_imag_lo = state_vec_lo.imag[this_task]

            state_vec_out.real[this_task] = rot1.real * state_real_up - rot1.imag * state_imag_up \
                                       + rot2.real * state_real_lo - rot2.imag * state_imag_lo
            state_vec_out.imag[this_task] = rot1.real * state_imag_up + rot1.imag * state_real_up \
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
            raise ValueError(
                "The specified matrix targets too many qubits; "
                "the batches of amplitudes to modify cannot all fit in a single distributed node's memory allocation.")

    def __validate_matrix(self, ureal, uimag, row_num: int, column_num: int):
        if len(ureal) != row_num or len(ureal[0]) != column_num or len(uimag) != row_num or len(uimag[0]) != column_num:
            raise ValueError("expected arrary length of ureal or uimag")

    def __validate_numamps(self, start_ind: int, numamps: int):
        if start_ind < 0 or start_ind >= self.reg.num_qubits_in_state_vec:
            raise ValueError("Invalid state index. Must be >=0 and <2^numQubits.")
        if numamps < 0 or numamps > self.reg.num_amps_total:
            raise ValueError("Invalid number of amplitudes. Must be >=0 and <=2^numQubits.")
        if numamps + start_ind > self.reg.num_amps_total:
            raise ValueError("More amplitudes given than exist in the statevector from the given starting index.")

    def __mask_contains_bit(self, mask: int, bit_ind: int):
        return mask & (1 << bit_ind)

    def __flip_bit(self, number: int, bit_ind: int):
        return self.sim_cpu.flip_bit(number, bit_ind)

    def __insert_zero_bit(self, number: int, index: int):
        return self.sim_cpu.insert_zero_bit(number, index)

    def __insert_two_zero_bits(self, number: int, bit1: int, bit2: int):
        return self.sim_cpu.insert_two_zero_bits(number, bit1, bit2)

    def __extract_bit(self, ctrl: int, index: int):
        return self.sim_cpu.extract_bit(ctrl, index)

    def __is_odd_parity(self, number: int, qb1: int, qb2: int):
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

    def __swap_qubit_amps_distributed(self, pair_rank: int, qb1: int, qb2: int):
        global_start_ind = self.reg.chunk_id * self.reg.num_amps_per_chunk
        pair_global_start_ind = pair_rank * self.reg.num_amps_per_chunk
        for local_ind in range(self.reg.num_amps_per_chunk):
            global_ind = global_start_ind + local_ind
            if self.__is_odd_parity(global_ind, qb1, qb2):
                pair_global_ind = self.__flip_bit(self.__flip_bit(global_ind, qb1), qb2)
                pair_local_ind = pair_global_ind - pair_global_start_ind

                self.reg.state_vec.real[local_ind] = self.reg.pair_state_vec.real[pair_local_ind]
                self.reg.state_vec.imag[local_ind] = self.reg.pair_state_vec.imag[pair_local_ind]

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
                ureal[0][0] * re00 - uimag[0][0] * im00 + \
                ureal[0][1] * re01 - uimag[0][1] * im01 + \
                ureal[0][2] * re10 - uimag[0][2] * im10 + \
                ureal[0][3] * re11 - uimag[0][3] * im11
            self.reg.state_vec.imag[ind00] = \
                uimag[0][0] * re00 + ureal[0][0] * im00 + \
                uimag[0][1] * re01 + ureal[0][1] * im01 + \
                uimag[0][2] * re10 + ureal[0][2] * im10 + \
                uimag[0][3] * re11 + ureal[0][3] * im11

            self.reg.state_vec.real[ind01] = \
                ureal[1][0] * re00 - uimag[1][0] * im00 + \
                ureal[1][1] * re01 - uimag[1][1] * im01 + \
                ureal[1][2] * re10 - uimag[1][2] * im10 + \
                ureal[1][3] * re11 - uimag[1][3] * im11
            self.reg.state_vec.imag[ind01] = \
                uimag[1][0] * re00 + ureal[1][0] * im00 + \
                uimag[1][1] * re01 + ureal[1][1] * im01 + \
                uimag[1][2] * re10 + ureal[1][2] * im10 + \
                uimag[1][3] * re11 + ureal[1][3] * im11

            self.reg.state_vec.real[ind10] = \
                ureal[2][0] * re00 - uimag[2][0] * im00 + \
                ureal[2][1] * re01 - uimag[2][1] * im01 + \
                ureal[2][2] * re10 - uimag[2][2] * im10 + \
                ureal[2][3] * re11 - uimag[2][3] * im11
            self.reg.state_vec.imag[ind10] = \
                uimag[2][0] * re00 + ureal[2][0] * im00 + \
                uimag[2][1] * re01 + ureal[2][1] * im01 + \
                uimag[2][2] * re10 + ureal[2][2] * im10 + \
                uimag[2][3] * re11 + ureal[2][3] * im11

            self.reg.state_vec.real[ind11] = \
                ureal[3][0] * re00 - uimag[3][0] * im00 + \
                ureal[3][1] * re01 - uimag[3][1] * im01 + \
                ureal[3][2] * re10 - uimag[3][2] * im10 + \
                ureal[3][3] * re11 - uimag[3][3] * im11
            self.reg.state_vec.imag[ind11] = \
                uimag[3][0] * re00 + ureal[3][0] * im00 + \
                uimag[3][1] * re01 + ureal[3][1] * im01 + \
                uimag[3][2] * re10 + ureal[3][2] * im10 + \
                uimag[3][3] * re11 + ureal[3][3] * im11

    def __multi_controlled_two_qubit_unitary(self, ctrl_mask: int, q1: int, q2: int, ureal, uimag):
        q1_fits_in_node = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, q1)
        q2_fits_in_node = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, q2)

        if q1_fits_in_node and q2_fits_in_node:
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, q1, q2, ureal, uimag)
        elif q1_fits_in_node:
            qswap = (q1 - 1) if q1 > 0 else (q1 + 1)
            if self.__mask_contains_bit(ctrl_mask, qswap):
                ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, q2), qswap)

            self.__swap_qubit_amps(q2, qswap)
            self.__multi_controlled_two_qubit_unitary_local(ctrl_mask, q1, qswap, ureal, uimag)
            self.__swap_qubit_amps(q2, qswap)
        elif q2_fits_in_node:
            qswap = (q2 - 1) if q2 > 0 else (q2 + 1)
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

    def apply_matrix4(self, target1, target2, ureal, uimag):
        self.__apply_matrix4(target1, target2, ureal, uimag)

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

    def rotate_around_axis(self, target_bit, angle, unit_axis):
        mag = math.sqrt(
            unit_axis[0] * unit_axis[0]
            + unit_axis[1] * unit_axis[1]
            + unit_axis[2] * unit_axis[2]
        )
        unit_vec = [unit_axis[0] / mag, unit_axis[1] / mag, unit_axis[2] / mag]

        rot1 = complex()
        rot2 = complex()
        alpha = complex()
        beta = complex()

        alpha.real = math.cos(angle / 2.0)
        alpha.imag = -math.sin(angle / 2.0) * unit_vec[2]
        beta.real = math.sin(angle / 2.0) * unit_vec[1]
        beta.imag = -math.sin(angle / 2.0) * unit_vec[0]

        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_bit)

        # rank's chunk is in upper half of block
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.rotate_around_axis(target_bit, angle, unit_axis)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_isupper = self.__exchange_state()
            self.__get_rot_angle(rank_isupper, rot1, rot2, alpha, beta)

            # this rank's values are either in the upper of lower half of the block.
            # send values to compactUnitaryDistributed in the correct order
            if rank_isupper:
                self.__rotate_around_axis_distributed(rot1, rot2, self.reg.state_vec, self.reg.pair_state_vec,
                                                      self.reg.state_vec)
            else:
                self.__rotate_around_axis_distributed(rot1, rot2, self.reg.pair_state_vec, self.reg.state_vec,
                                                      self.reg.state_vec)

    def __get_rot_angle(rank_isupper, rot1, rot2, alpha, beta):
        if rank_isupper:
            rot1.real = alpha.real
            rot1.imag = alpha.imag
            rot2.real = -beta.real
            rot2.imag = -beta.imag
        else:
            rot1.real = beta.real
            rot1.imag = beta.imag
            rot2.real = alpha.real
            rot2.imag = alpha.imag

    def __rotate_around_axis_distributed(self, rot1, rot2, state_vec_up, state_vec_lo, state_vec_out):
        for this_task in range(self.reg.num_amps_per_chunk):
            # store current state vector values in temp variables
            state_real_up = state_vec_up.real[this_task]
            state_imag_up = state_vec_up.imag[this_task]

            state_real_lo = state_vec_lo.real[this_task]
            state_imag_lo = state_vec_lo.imag[this_task]

            state_vec_out.real[
                this_task] = rot1.real * state_real_up - rot1.imag * state_imag_up + rot2.real * state_real_lo + rot2.imag * state_imag_lo
            state_vec_out.imag[
                this_task] = rot1.real * state_imag_up + rot1.imag * state_real_up + rot2.real * state_imag_lo - rot2.imag * state_real_lo

    def pauli_x(self, target_bit):
        """The single-qubit Pauli-X gate."""
        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_bit)
        # rank's chunk is in upper half of block
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.sim_cpu.pauli_x(target_bit)
        else:
            # need to get corresponding chunk of state vector from other rank
            self.__exchange_state(target_bit)
            # this rank's values are either in the upper of lower half of the block. pauliX just replaces
            # this rank's values with pair values
            self.__pauliX_distributed(self.reg.pair_state_vec, self.reg.state_vec)

    def __pauliX_distributed(self, stateVecIn, stateVecOut):
        for this_task in range(self.reg.num_amps_per_chunk):
            stateVecOut.real[this_task] = stateVecIn.real[this_task]
            stateVecOut.imag[this_task] = stateVecIn.imag[this_task]

    def pauli_y(self, target_bit):
        """The single-qubit Pauli-Y gate."""
        conj_fac = 1

        # flag to require memory exchange. 1: an entire block fits on one rank, 0: at most half a block fits on one rank
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_bit)
        if use_local_data_only:
            self.sim_cpu.pauli_y(target_bit)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_isupper = self.__exchange_state()
            # this rank's values are either in the upper of lower half of the block
            self.__pauliY_distributed(self.reg.pair_state_vec, self.reg.state_vec, rank_isupper, conj_fac)

    def __pauliY_distributed(self, stateVecIn, stateVecOut, update_upper, conj_fac):
        real_sign = 1
        imag_sign = 1
        if update_upper:
            imag_sign = -1
        else:
            real_sign = -1

        for this_task in range(self.reg.num_amps_per_chunk):
            stateVecOut.real[this_task] = conj_fac * real_sign * stateVecIn.imag[this_task]
            stateVecOut.imag[this_task] = conj_fac * imag_sign * stateVecIn.real[this_task]

    def pauli_z(self, target):
        """The single-qubit Pauli-Z gate."""
        real = -1
        imag = 0
        self.__phase_shift_by_term(target, real, imag)

    def s_gate(self, target):
        """The single-qubit S gate."""
        real = 0
        imag = 1
        self.__phase_shift_by_term(target, real, imag)

    def t_gate(self, target):
        """The single-qubit T gate."""
        real = 1 / math.sqrt(2)
        imag = 1 / math.sqrt(2)
        self.__phase_shift_by_term(target, real, imag)

    def __phase_shift_by_term(self, target_qubit, real, imag):
        for index in range(self.reg.num_amps_per_chunk):
            # update the coeff of the |1> state of the target qubit
            target_bit = self.sim_cpu.extract_bit(target_qubit, index + self.reg.chunk_id * self.reg.num_amps_per_chunk)
            if target_bit:
                state_real_lo = self.reg.state_vec.real[index]
                state_imag_lo = self.reg.state_vec.imag[index]

                self.reg.state_vec.real[index] = real * state_real_lo - imag * state_imag_lo
                self.reg.state_vec.imag[index] = imag * state_real_lo + real * state_imag_lo

    def swap(self, qb1: int, qb2: int):
        qb_big = max(qb1, qb2)
        if self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, qb_big):
            return self.sim_cpu.swap(qb1, qb2)

        odd_parity_global_idx = self.__get_global_idx_of_odd_parity_in_chunk(qb1, qb2)
        if odd_parity_global_idx == -1:
            return

        pair_rank = self.sim_cpu.flip_bit(self.sim_cpu.flip_bit(odd_parity_global_idx, qb1),
                                          qb2) // self.reg.num_amps_per_chunk
        self.__exchange_state_vectors(pair_rank)
        self.__swap_qubit_amp_dis(pair_rank, qb1, qb2)

    def __swap_qubit_amp_dis(self, pair_rank: int, qb1: int, qb2: int):
        re_vec = self.reg.state_vec.real
        im_vec = self.reg.state_vec.imag
        re_pair_vec = self.reg.pair_state_vec.real
        im_pair_vec = self.reg.pair_state_vec.imag

        num_local_amps = self.reg.num_amps_per_chunk
        global_start_idx = self.reg.chunk_id * num_local_amps
        pair_global_start_idx = pair_rank * num_local_amps

        for local_idx in range(num_local_amps):
            global_idx = global_start_idx + local_idx
            if self.__is_odd_parity(global_idx, qb1, qb2):
                pair_global_idx = self.sim_cpu.flip_bit(self.sim_cpu.flip_bit(global_idx, qb1), qb2)
                pair_local_idx = pair_global_idx - pair_global_start_idx

                re_vec[local_idx] = re_pair_vec[pair_local_idx]
                im_vec[local_idx] = im_pair_vec[pair_local_idx]

    def __get_global_idx_of_odd_parity_in_chunk(self, qb1: int, qb2: int):
        chunk_start_idx = self.reg.num_amps_per_chunk * self.reg.chunk_id
        chunk_end_idx = chunk_start_idx + self.reg.num_amps_per_chunk

        if self.sim_cpu.extract_bit(qb1, chunk_start_idx) != self.sim_cpu.extract_bit(qb2, chunk_start_idx):
            return chunk_start_idx

        odd_parity_idx = self.sim_cpu.flip_bit(chunk_start_idx, qb1)
        if chunk_start_idx <= odd_parity_idx < chunk_end_idx:
            return odd_parity_idx

        odd_parity_idx = self.sim_cpu.flip_bit(chunk_start_idx, qb2)
        if chunk_start_idx <= odd_parity_idx < chunk_end_idx:
            return odd_parity_idx

        return -1

    def __is_odd_parity(self, number: int, qb1: int, qb2: int):
        return self.sim_cpu.extract_bit(qb1, number) != self.sim_cpu.extract_bit(qb2, number)

    def get_prob(self, index):
        real = self.__get_real_amp(index)
        imag = self.__get_imag_Amp(index)
        return real * real + imag * imag

    def get_probs(self, qubits):
        outcome_probs = self.sim_cpu.get_probs(qubits)
        outcome_probs = self.comm.allreduce(outcome_probs, MPI.SUM)
        return outcome_probs

    def __get_real_amp(self, index):
        chunk_id = index // self.reg.num_amps_per_chunk
        el = 0.0
        if self.reg.chunk_id == chunk_id:
            el = self.reg.state_vec.real[index - chunk_id * self.reg.num_amps_per_chunk]

        el = self.comm.bcast(el)
        return el

    def __get_imag_Amp(self, index):
        chunk_id = index // self.reg.num_amps_per_chunk
        el = 0.0
        if self.reg.chunk_id == chunk_id:
            el = self.reg.state_vec.imag[index - chunk_id * self.reg.num_amps_per_chunk]

        el = self.comm.bcast(el)
        return el

    def get_statevector(self):
        state_vectors = self.comm.allgather(self.reg.state_vec)
        state_list = []
        for state_vec in state_vectors:
            for i in range(self.reg.num_amps_per_chunk):
                real = state_vec.real[i]
                imag = state_vec.imag[i]
                if -1e-15 < real < 1e-15:
                    real = 0
                if -1e-15 < imag < 1e-15:
                    imag = 0
                state = str(real) + ", " + str(imag)
                state_list.append(state)
        return state_list

    def __controlled_unitary_local(self, control_qubit: int, target_qubit: int, ureal, uimag):
        num_tasks = self.reg.num_amps_per_chunk >> 1
        size_half_block = 1 << target_qubit
        size_block = 2 * size_half_block
        for this_task in range(num_tasks):
            this_block = this_task / size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            control_bit = self.__extract_bit(control_qubit, index_up + self.reg.chunk_id * self.reg.num_amps_per_chunk)
            if control_bit:
                # store current state vector values in temp variables
                state_real_up = self.reg.state_vec.real[index_up]
                state_imag_Up = self.reg.state_vec.imag[index_up]

                state_real_lo = self.reg.state_vec.real[index_lo]
                state_imag_lo = self.reg.state_vec.imag[index_lo]

                # state[index_up] = u00 * state[index_up] + u01 * state[index_lo]
                self.reg.state_vec.real[index_up] = ureal[0][0] * state_real_up - uimag[0][0] * state_imag_Up + \
                                                    ureal[0][1] * state_real_lo - uimag[0][1] * state_imag_lo
                self.reg.state_vec.imag[index_up] = ureal[0][0] * state_imag_Up + uimag[0][0] * state_real_up + \
                                                    ureal[0][1] * state_imag_lo + uimag[0][1] * state_real_lo

                # state[index_lo] = u10  * state[index_up] + u11 * state[index_lo]
                self.reg.state_vec.real[index_lo] = ureal[1][0] * state_real_up - uimag[1][0] * state_imag_Up + \
                                                    ureal[1][1] * state_real_lo - uimag[1][1] * state_imag_lo
                self.reg.state_vec.imag[index_lo] = ureal[1][0] * state_imag_Up + uimag[1][0] * state_real_up + \
                                                    ureal[1][1] * state_imag_lo + uimag[1][1] * state_real_lo

    def __get_rot_angle_grom_unitary_matrix(self, chunk_is_upper: int, rot1: complex, rot2: complex, ureal, uimag):
        if chunk_is_upper:
            rot1.real = ureal[0][0]
            rot1.imag = uimag[0][0]
            rot2.real = ureal[0][1]
            rot2.imag = uimag[0][1]
        else:
            rot1.real = ureal[1][0]
            rot1.imag = uimag[1][0]
            rot2.real = ureal[1][1]
            rot2.imag = uimag[1][1]

    def __controlled_unitary_distributed(self, control_qubit: int, rot1: complex, rot2: complex, state_vec_up: StateVec,
                                         state_vec_lo: StateVec):
        for this_task in range(self.reg.num_amps_per_chunk):
            control_bit = self.__extract_bit(control_qubit, this_task + self.reg.chunk_id * self.reg.num_amps_per_chunk)
            if control_bit:
                # store current state vector values in temp variables
                stateRealUp = state_vec_up.real[this_task]
                stateImagUp = state_vec_up.imag[this_task]

                stateRealLo = state_vec_lo.real[this_task]
                stateImagLo = state_vec_lo.imag[this_task]

                self.reg.state_vec.real[
                    this_task] = rot1.real * stateRealUp - rot1.imag * stateImagUp + rot2.real * stateRealLo - rot2.imag * stateImagLo
                self.reg.state_vec.imag[
                    this_task] = rot1.real * stateImagUp + rot1.imag * stateRealUp + rot2.real * stateImagLo + rot2.imag * stateRealLo

    def __controlled_unitary(self, control_qubit: int, target_qubit: int, ureal, uimag):
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, target_qubit)
        rot1 = complex()
        rot2 = complex()
        if use_local_data_only:
            # all values required to update state vector lie in this rank
            self.__controlled_unitary_local(control_qubit, target_qubit, ureal, uimag)
        else:
            # need to get corresponding chunk of state vector from other rank
            rank_is_upper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, target_qubit)
            self.__get_rot_angle_grom_unitary_matrix(rank_is_upper, rot1, rot2, ureal, uimag)
            pair_rank = self.__get_chunk_pair_id(rank_is_upper, self.reg.chunk_id, self.reg.num_amps_per_chunk,
                                                 target_qubit)
            # get corresponding values from my pair
            self.__exchange_state_vectors(pair_rank)

            # this rank's values are either in the upper of lower half of the block. send values to controlledUnitaryDistributed in the correct order
            if rank_is_upper:
                self.__controlled_unitary_distributed(control_qubit, rot1, rot2, self.reg.state_vec,
                                                      self.reg.pair_state_vec)
            else:
                self.__controlled_unitary_distributed(control_qubit, rot1, rot2, self.reg.pair_state_vec,
                                                      self.reg.state_vec)

    def __controlled_two_qubit_unitary(self, control_qubit: int, target_qubit1: int, target_qubit2: int, ureal, uimag):
        ctrl_mask = 1 << control_qubit
        self.__multi_controlled_two_qubit_unitary(ctrl_mask, target_qubit1, target_qubit2, ureal, uimag)

    def z1(self, target: int, ureal, uimag):
        self.__validate_target(target)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__apply_matrix2(target, ureal, uimag)

    def cu1(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__apply_matrix4(target_qubit1, target_qubit2, ureal, uimag)

    def cu3(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__apply_matrix4(target_qubit1, target_qubit2, ureal, uimag)

    def cu(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__apply_matrix4(target_qubit1, target_qubit2, ureal, uimag)

    def cr(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__apply_matrix4(target_qubit1, target_qubit2, ureal, uimag)

    def iswap(self, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__apply_matrix4(target_qubit1, target_qubit2, ureal, uimag)

    def sqrtx(self, target: int, ureal, uimag):
        self.__validate_target(target)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__apply_matrix2(target, ureal, uimag)

    def ch(self, control_qubit: int, target_qubit: int, ureal, uimag):
        self.__validate_target(control_qubit)
        self.__validate_target(target_qubit)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__controlled_unitary(control_qubit, target_qubit, ureal, uimag)

    def sqrtxdg(self, target: int, ureal, uimag):
        self.__validate_target(target)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__apply_matrix2(target, ureal, uimag)

    def csqrtx(self, control_qubit: int, target_qubit: int, ureal, uimag):
        self.__validate_target(control_qubit)
        self.__validate_target(target_qubit)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__controlled_unitary(control_qubit, target_qubit, ureal, uimag)

    def cswap(self, control_qubit: int, target_qubit1: int, target_qubit2: int, ureal, uimag):
        self.__validate_target(control_qubit)
        self.__validate_target(target_qubit1)
        self.__validate_target(target_qubit2)
        self.__validate_matrix(ureal, uimag, 4, 4)
        self.__controlled_two_qubit_unitary(control_qubit, target_qubit1, target_qubit2, ureal, uimag)

    def amp(self, reals, imags, startindex: int, numamps: int):
        self.__validate_numamps(startindex, numamps)
        # this is actually distributed, since the user's code runs on every node
        # local start/end indices of the given amplitudes, assuming they fit in this chunk
        # these may be negative or above reg.num_amps_per_chunk
        local_start_ind = startindex - self.reg.chunk_id * self.reg.num_amps_per_chunk
        local_end_ind = local_start_ind + numamps

        # add this to a local index to get corresponding elem in reals & imags
        offset = self.reg.chunk_id * self.reg.num_amps_per_chunk - startindex

        # restrict these indices to fit into this chunk
        if local_start_ind < 0:
            local_start_ind = 0
        if local_end_ind > self.reg.num_amps_per_chunk:
            local_end_ind = self.reg.num_amps_per_chunk

        # they may now be out of order = no iterations
        # iterate these local inds - this might involve no iterations
        for index in range(local_start_ind, local_end_ind, 1):
            self.reg.state_vec.real[index] = reals[index + offset]
            self.reg.state_vec.imag[index] = imags[index + offset]

    def sdg(self, target_bit, ureal, uimag):
        self.__apply_matrix2(target_bit, ureal, uimag)

    def tdg(self, target_bit, ureal, uimag):
        self.__apply_matrix2(target_bit, ureal, uimag)

    def sqrtswap(self, target_bit0, target_bit1, ureal, uimag):
        self.__apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def multi_controlled_multi_qubit_not(self, ctrl_qbs, ctrl_cnt, targ_qbs, tar_cnt):
        ctrl_mask = self.sim_cpu.get_qubit_bit_mask(ctrl_qbs, ctrl_cnt)
        targ_mask = self.sim_cpu.get_qubit_bit_mask(targ_qbs, tar_cnt)

        first_idx = self.reg.chunk_id * self.reg.num_amps_per_chunk

        if (first_idx | ctrl_mask) - first_idx >= self.reg.num_amps_per_chunk:
            return

        pair_idx = first_idx ^ targ_mask
        pair_rank = pair_idx / self.reg.num_amps_per_chunk

        use_local_data_only = (pair_rank == self.reg.chunk_id)

        if use_local_data_only:
            self.sim_cpu.multi_controlled_multi_qubit_not(ctrl_qbs, ctrl_cnt, targ_qbs, tar_cnt)
        else:
            self.__exchange_state_vectors(pair_rank)
            self.__multi_controlled_multi_qubit_not_dis(
                ctrl_mask, targ_mask, self.reg.pair_state_vec, self.reg.state_vec)

    def __multi_controlled_multi_qubit_not_dis(self, ctrl_mask, targ_mask, state_vec_in, state_vec_out):
        num_amps = self.reg.num_amps_per_chunk
        global_offset = self.reg.chunk_id * num_amps

        in_real = state_vec_in.real
        in_imag = state_vec_in.imag
        out_real = state_vec_out.real
        out_imag = state_vec_out.imag

        for out_idx in range(num_amps):
            out_idx_global = out_idx + global_offset
            if ctrl_mask and (ctrl_mask & out_idx_global) != ctrl_mask:
                continue

            in_idx_global = out_idx_global ^ targ_mask
            in_idx = in_idx_global % num_amps
            out_real[out_idx] = in_real[in_idx]
            out_imag[out_idx] = in_imag[in_idx]

    def cy(self, ctrl_bit, targ_bit):
        conj_fac = 1

        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, targ_bit)

        if use_local_data_only:
            self.sim_cpu.cy(ctrl_bit, targ_bit)
        else:
            rank_isupper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, targ_bit)
            pair_rank = self.__get_chunk_pair_id(rank_isupper, self.reg.chunk_id, self.reg.num_amps_per_chunk, targ_bit)

            self.__exchange_state_vectors(pair_rank)

            if rank_isupper:
                self.__cy_dis(ctrl_bit, self.reg.pair_state_vec, self.reg.state_vec, conj_fac)
            else:
                self.__cy_dis(ctrl_bit, self.reg.pair_state_vec, self.reg.state_vec, -conj_fac)

    def __cy_dis(self, ctrl_bit, state_vec_in, state_vec_out, conj_fac):
        num_tasks = self.reg.num_amps_per_chunk
        chunk_size = self.reg.num_amps_per_chunk
        chunk_id = self.reg.chunk_id

        state_vec_real_in = state_vec_in.real
        state_vec_imag_in = state_vec_in.imag
        state_vec_real_out = state_vec_out.real
        state_vec_imag_out = state_vec_out.imag

        for this_task in range(num_tasks):
            ctrl_bit = self.__extract_bit(ctrl_bit, this_task + chunk_id * chunk_size)
            if ctrl_bit:
                state_vec_real_out[this_task] = conj_fac * state_vec_imag_in[this_task]
                state_vec_imag_out[this_task] = conj_fac * -state_vec_real_in[this_task]

    def cz(self, ctrl_bits, ctrl_cnt):
        self.sim_cpu.cz(ctrl_bits, ctrl_cnt)

    def u3(self, target_bit, ureal, uimag):
        self.__apply_matrix2(target_bit, ureal, uimag)

    def u2(self, target_bit, ureal, uimag):
        self.__apply_matrix2(target_bit, ureal, uimag)

    def u1(self, target_bit, ureal, uimag):
        self.__apply_matrix2(target_bit, ureal, uimag)

    def crx(self, control_bit, target_bit, angle):
        unit_axis = [1, 0, 0]
        self.__controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def cry(self, control_bit, target_bit, angle):
        unit_axis = [0, 1, 0]
        self.__controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def crz(self, control_bit, target_bit, angle):
        unit_axis = [0, 0, 1]
        self.__controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def __controlled_rotate_around_axis(self, ctrl_bit, targ_bit, angle, axis):
        use_local_data_only = self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, targ_bit)

        if use_local_data_only:
            self.sim_cpu.controlled_rotate_around_axis(ctrl_bit, targ_bit, angle, axis)
        else:
            mag = math.sqrt(
                axis[0] * axis[0]
                + axis[1] * axis[1]
                + axis[2] * axis[2]
            )
            unit_vec = [axis[0] / mag, axis[1] / mag, axis[2] / mag]

            alpha_real = math.cos(angle / 2.0)
            alpha_imag = -math.sin(angle / 2.0) * unit_vec[2]
            beta_real = math.sin(angle / 2.0) * unit_vec[1]
            beta_imag = -math.sin(angle / 2.0) * unit_vec[0]

            rank_isupper = self.__chunk_isupper(self.reg.chunk_id, self.reg.num_amps_per_chunk, targ_bit)
            if rank_isupper:
                rot = [alpha_real, alpha_imag, -beta_real, -beta_imag]
            else:
                rot = [beta_real, beta_imag, alpha_real, alpha_imag]

            pair_rank = self.__get_chunk_pair_id(rank_isupper, self.reg.chunk_id, self.reg.num_amps_per_chunk, targ_bit)

            self.__exchange_state_vectors(pair_rank)

            if rank_isupper:
                self.__controlled_rotate_around_axis_dis(ctrl_bit, rot,
                                                         self.reg.state_vec,  # upper
                                                         self.reg.pair_state_vec,  # lower
                                                         self.reg.state_vec)  # output
            else:
                self.__controlled_rotate_around_axis_dis(ctrl_bit, rot,
                                                         self.reg.pair_state_vec,  # upper
                                                         self.reg.state_vec,  # lower
                                                         self.reg.state_vec)  # output

    def __controlled_rotate_around_axis_dis(self, ctrl_bit, rot, state_vec_up, state_vec_lo, state_vec_out):
        num_tasks = self.reg.num_amps_per_chunk
        chunk_size = self.reg.num_amps_per_chunk
        chunk_id = self.reg.chunk_id

        rot1_real = rot[0]
        rot1_imag = rot[1]
        rot2_real = rot[2]
        rot2_imag = rot[3]

        state_vec_real_up = state_vec_up.real
        state_vec_image_up = state_vec_up.imag
        state_vec_real_lo = state_vec_lo.real
        state_vec_imag_lo = state_vec_lo.imag
        state_vec_real_out = state_vec_out.real
        state_vec_imag_out = state_vec_out.imag

        for this_task in range(num_tasks):
            ctrl_bit = self.__extract_bit(ctrl_bit, this_task + chunk_id * chunk_size)
            if ctrl_bit:
                state_real_up = state_vec_real_up[this_task]
                state_imag_up = state_vec_image_up[this_task]

                state_real_lo = state_vec_real_lo[this_task]
                state_imag_lo = state_vec_imag_lo[this_task]

                state_vec_real_out[this_task] = (rot1_real * state_real_up - rot1_imag * state_imag_up +
                                                 rot2_real * state_real_lo + rot2_imag * state_imag_lo)
                state_vec_imag_out[this_task] = (rot1_real * state_imag_up + rot1_imag * state_real_up +
                                                 rot2_real * state_imag_lo - rot2_imag * state_real_lo)

    def x1(self, target, ureal, uimag):
        self.__validate_target(target)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__apply_matrix2(target, ureal, uimag)

    def y1(self, target, ureal, uimag):
        self.__validate_target(target)
        self.__validate_matrix(ureal, uimag, 2, 2)
        self.__apply_matrix2(target, ureal, uimag)
        
    def __multi_controlled_multi_qubit_unitary_local(self, ctrl_mask: int, targets, ureals, uimags):
        num_targs = len(targets)
        num_tasks = self.reg.num_amps_per_chunk >> num_targs
        num_targ_amps = 1 << num_targs
        global_ind_start = self.reg.chunk_id*self.reg.num_amps_per_chunk
        
        sorted_targs = [0]*num_targs
        amp_inds = [0]*num_targ_amps
        re_amps = [0]*num_targ_amps
        im_amps = [0]*num_targ_amps
        
        for t in range(num_targs):
            sorted_targs[t] = targets[t]
        sorted_targs.sort()
        
        for this_task in range(num_tasks):
            # find this task's start index (where all targs are 0)
            this_ind00 = this_task
            for t in range(num_targs):
                this_ind00 = self.__insert_zero_bit(this_ind00, sorted_targs[t])
                
            # this task only modifies amplitudes if control qubits are 1 for this state
            this_global_ind00 = this_ind00 + global_ind_start
            if (ctrl_mask and ((ctrl_mask & this_global_ind00) != ctrl_mask)):
                continue
                
            # determine the indices and record values of this tasks's target amps
            for i in range(num_targ_amps):
                # get statevec index of current target qubit assignment
                ind = this_ind00
                for t in range(num_targs):
                    if (self.__extract_bit(t, i)):
                        ind = self.__flip_bit(ind, targets[t])
                
                # update this tasks's private arrays
                amp_inds[i] = ind
                re_amps [i] = self.reg.state_vec.real[ind]
                im_amps [i] = self.reg.state_vec.imag[ind]
            
            # modify this tasks's target amplitudes
            for r in range(num_targ_amps):
                ind = amp_inds[r]
                self.reg.state_vec.real[ind] = 0
                self.reg.state_vec.imag[ind] = 0
                
                for c in range(num_targ_amps):
                    re_elem = ureals[r][c]
                    im_elem = uimags[r][c]
                    self.reg.state_vec.real[ind] += re_amps[c]*re_elem - im_amps[c]*im_elem
                    self.reg.state_vec.imag[ind] += re_amps[c]*im_elem + im_amps[c]*re_elem
        
    def __multi_controlled_multi_qubit_unitary(self, ctrl_mask: int, targets, ureals, uimags):
        num_targs = len(targets)
        targ_mask = self.sim_cpu.get_qubit_bit_mask(targets, num_targs)
        free_qb = 0
        while self.__mask_contains_bit(targ_mask, free_qb):
            free_qb += 1
            
        swap_targs = [0]*num_targs
        for t in range(num_targs):
            if self.__half_matrix_block_fits_in_chunk(self.reg.num_amps_per_chunk, targets[t]):
                swap_targs[t] = targets[t]
            else:
                swap_targs[t] = free_qb
                if self.__mask_contains_bit(ctrl_mask, swap_targs[t]):
                    ctrl_mask = self.__flip_bit(self.__flip_bit(ctrl_mask, swap_targs[t]), targets[t])
                    
                free_qb += 1
                while self.__mask_contains_bit(targ_mask, free_qb):
                    free_qb += 1
                    
        for t in range(num_targs):
            if swap_targs[t] != targets[t]:
                self.__swap_qubit_amps(targets[t], swap_targs[t])
            
        self.__multi_controlled_multi_qubit_unitary_local(ctrl_mask, swap_targs, ureals, uimags)
        
        for t in range(num_targs):
            self.__swap_qubit_amps(targets[t], swap_targs[t])
            
    def __apply_multi_controlled_matrix_n(self, controls, targets, ureals, uimags):
        num_ctrls = len(controls)
        ctrl_mask = self.sim_cpu.get_qubit_bit_mask(controls, num_ctrls)
        self.__multi_controlled_multi_qubit_unitary(ctrl_mask, targets, ureals, uimags)
        
    def matrix(self, controls, targets, reals, imags):
        self.__apply_multi_controlled_matrix_n(controls, targets, reals, imags)