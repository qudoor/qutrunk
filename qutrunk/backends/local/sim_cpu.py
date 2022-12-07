"""Implementation of quantum compute simulator for cpu running mode."""


import math
import random
from enum import Enum

REAL_EPS = 1e-13


class BitEncoding(Enum):
    """Bit Encoding"""

    UNSIGNED = 0
    TWOS_COMPLEMENT = 1


class PhaseFunc(Enum):
    """PhaseFunc"""

    NORM = 0
    SCALED_NORM = 1
    INVERSE_NORM = 2
    SCALED_INVERSE_NORM = 3
    SCALED_INVERSE_SHIFTED_NORM = 4
    PRODUCT = 5
    SCALED_PRODUCT = 6
    INVERSE_PRODUCT = 7
    SCALED_INVERSE_PRODUCT = 8
    DISTANCE = 9
    SCALED_DISTANCE = 10
    INVERSE_DISTANCE = 11
    SCALED_INVERSE_DISTANCE = 12
    SCALED_INVERSE_SHIFTED_DISTANCE = 13


class PauliOpType(Enum):
    """PauliOpType"""

    PAULI_I = 0
    PAULI_X = 1
    PAULI_Y = 2
    PAULI_Z = 3


class SimCpu:
    """Simulator-cpu implement."""

    def __init__(self):
        self.real = []  # real
        self.imag = []  # imag
        self.qubits = 0
        self.total_num_amps = 0  # numAmpsPerChunk

    def create_qureg(self, num_qubits):
        """Allocate resource.

        Args:
            num_qubits: number of qubits
        """
        self.qubits = num_qubits
        num_amps = 2**num_qubits
        self.total_num_amps = num_amps
        # TODO: need to improve.
        self.real = [0] * num_amps
        self.imag = [0] * num_amps

    def init_blank_state(self):
        """Init blank state"""
        for i in range(self.total_num_amps):
            self.real[i] = 0.0
            self.imag[i] = 0.0

    def init_zero_state(self):
        """Init zero state"""
        self.init_blank_state()
        # TODO:??
        self.real[0] = 1.0

    def init_plus_state(self):
        """Init plus state"""
        # dimension of the state vector
        chunk_size = self.total_num_amps
        state_vec_size = chunk_size
        normFactor = 1.0 / math.sqrt(state_vec_size)

        # initialise the state to |+++..+++> = 1/normFactor {1, 1, 1, ...}
        for this_task in range(chunk_size):
            self.real[this_task] = normFactor
            self.imag[this_task] = 0.0

    def init_classical_state(self):
        """Init classical state"""

        # initialise the state to vector to all zeros
        self.init_blank_state()

        # give the specified classical state prob 1
        self.real[0] = 1.0
        self.imag[0] = 0.0

    def amp(self, reals, imags, startindex, numamps):
        """Init amplitudes state"""
        
        for index in range(numamps):
            self.real[startindex] = reals[index]
            self.imag[startindex] = imags[index]
            startindex += 1

    def matrix(self, controls, targets, reals, imags):
        """Apply custom matrix"""
        ctrlMask = self.get_qubit_bit_mask(controls, len(controls))

        numTargs = len(targets)
        numTasks = self.total_num_amps >> numTargs
        numTargAmps = 1 << numTargs

        ampInds = [0]*numTargAmps
        reAmps = [0]*numTargAmps
        imAmps = [0]*numTargAmps

        sortedTargs = [0]*numTargs
        for t in range(numTargs):
            sortedTargs[t] = targets[t]
        sortedTargs.sort()

        for thisTask in range(numTasks):
            # find this task's start index (where all targs are 0)
            thisInd00 = thisTask
            for t in range(numTargs):
                thisInd00 = self.insert_zero_bit(thisInd00, sortedTargs[t])
                
            # this task only modifies amplitudes if control qubits are 1 for this state
            thisGlobalInd00 = thisInd00
            if (ctrlMask and ((ctrlMask & thisGlobalInd00) != ctrlMask)):
                continue
                
            # determine the indices and record values of this tasks's target amps
            for i in range(numTargAmps):
                # get statevec index of current target qubit assignment
                ind = thisInd00
                for t in range(numTargs):
                    if (self.extract_bit(t, i)):
                        ind = self.flip_bit(ind, targets[t])
                
                # update this tasks's private arrays
                ampInds[i] = ind
                reAmps [i] = self.real[ind]
                imAmps [i] = self.imag[ind]
            
            # modify this tasks's target amplitudes
            for r in range(numTargAmps):
                ind = ampInds[r]
                self.real[ind] = 0
                self.imag[ind] = 0
                
                for c in range(numTargAmps):
                    reElem = reals[r][c]
                    imElem = imags[r][c]
                    self.real[ind] += reAmps[c]*reElem - imAmps[c]*imElem
                    self.imag[ind] += reAmps[c]*imElem + imAmps[c]*reElem

    # TODO:need to improve.
    def hadamard(self, target):
        """
        Apply hadamard gate.

        Args:
            target: target qubit.
        """
        size_half_block = 2**target
        size_block = size_half_block * 2
        num_task = self.total_num_amps // 2

        rec_root = 1.0 / math.sqrt(2)
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            state_real_up = self.real[index_up]
            state_imag_up = self.imag[index_up]

            state_real_lo = self.real[index_lo]
            state_imag_lo = self.imag[index_lo]

            self.real[index_up] = rec_root * (state_real_up + state_real_lo)
            self.imag[index_up] = rec_root * (state_imag_up + state_imag_lo)

            self.real[index_lo] = rec_root * (state_real_up - state_real_lo)
            self.imag[index_lo] = rec_root * (state_imag_up - state_imag_lo)

    def ch(self, control_bit, target_bit, ureal, uimage):
        self.controlled_unitary(control_bit, target_bit, ureal, uimage)

    def phase_shift(self, target, angle):
        """Shift the phase between |0> and |1> of a single qubit by a given angle.

        Args:
            target: qubit to undergo a phase shift.
            angle:  amount by which to shift the phase in radians.
        """
        real = math.cos(angle)
        imag = math.sin(angle)
        self.phase_shift_by_term(self.real, self.imag, target, real, imag)

    def controlled_phase_shift(self, ctrl, target, angle):
        """
        Controlled-Phase gate.

        Args:
            ctrl: control qubit
            target: target qubit
            angle: amount by which to shift the phase in radians.
        """
        state_vec_size = self.total_num_amps
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        for index in range(state_vec_size):
            bit1 = self.extract_bit(ctrl, index)
            bit2 = self.extract_bit(target, index)

            if bit1 and bit2:
                state_real_lo = self.real[index]
                state_imag_lo = self.imag[index]

                self.real[index] = cos_angle * state_real_lo - sin_angle * state_imag_lo
                self.imag[index] = sin_angle * state_real_lo + cos_angle * state_imag_lo

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

    def pauli_x(self, target):
        """The single-qubit Pauli-X gate."""
        self.paulix_local(self.real, self.imag, target)

    def pauli_y(self, target):
        """The single-qubit Pauli-Y gate."""
        conj_fac = 1
        self.pauliy_local(self.real, self.imag, target, conj_fac)

    def pauli_z(self, target):
        """The single-qubit Pauli-Z gate."""
        real = -1
        imag = 0
        self.phase_shift_by_term(self.real, self.imag, target, real, imag)

    def s_gate(self, target):
        """The single-qubit S gate."""
        real = 0
        imag = 1
        self.phase_shift_by_term(self.real, self.imag, target, real, imag)

    def t_gate(self, target):
        """The single-qubit T gate."""
        real = 1 / math.sqrt(2)
        imag = 1 / math.sqrt(2)
        self.phase_shift_by_term(self.real, self.imag, target, real, imag)

    def control_not(self, ctrl, target):
        """Control not gate"""
        num_task = self.total_num_amps // 2
        size_half_block = 2**target
        size_block = size_half_block * 2

        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            control_bit = self.extract_bit(ctrl, index_up)
            if control_bit:
                state_real_up = self.real[index_up]
                state_imag_up = self.imag[index_up]

                self.real[index_up] = self.real[index_lo]
                self.imag[index_up] = self.imag[index_lo]

                self.real[index_lo] = state_real_up
                self.imag[index_lo] = state_imag_up

    def sdg(self, target_bit, ureal, uimag):
        self.apply_matrix2(target_bit, ureal, uimag)

    def tdg(self, target_bit, ureal, uimag):
        self.apply_matrix2(target_bit, ureal, uimag)

    def sqrtswap(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def swap(self, target_bit0, target_bit1):
        num_task = self.total_num_amps >> 2
        for this_task in range(num_task):
            # determine ind00 of |..0..0..>, |..0..1..> and |..1..0..>
            ind00 = self.insert_two_zero_bits(this_task, target_bit0, target_bit1)
            ind01 = self.flip_bit(ind00, target_bit0)
            ind10 = self.flip_bit(ind00, target_bit1)

            # extract statevec amplitudes
            re01 = self.real[ind01]
            im01 = self.imag[ind01]
            re10 = self.real[ind10]
            im10 = self.imag[ind10]

            # swap 01 and 10 amps
            self.real[ind01] = re10
            self.real[ind10] = re01
            self.imag[ind01] = im10
            self.imag[ind10] = im01

    def cswap(self, control_bit, target_bit0, targetbi_bit1, ureal, uimage):
        self.controlled_two_qubit_unitary(
            control_bit, target_bit0, targetbi_bit1, ureal, uimage
        )

    def cy(self, control_bit, target_bit):
        conj_fac = 1
        num_task = self.total_num_amps >> 1
        size_half_block = 2**target_bit
        size_block = size_half_block * 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            controlBit = self.extract_bit(control_bit, index_up)
            if controlBit:
                state_real_up = self.real[index_up]
                state_imag_up = self.imag[index_up]

                # update under +-{{0, -i}, {i, 0}}
                self.real[index_up] = conj_fac * self.imag[index_lo]
                self.imag[index_up] = conj_fac * -self.real[index_lo]
                self.real[index_lo] = conj_fac * -state_imag_up
                self.imag[index_lo] = conj_fac * state_real_up

    def cz(self, control_bits, num_control_bits):
        state_vec_size = self.total_num_amps
        mask = self.get_qubit_bit_mask(control_bits, num_control_bits)
        for index in range(state_vec_size):
            if mask == (mask & index):
                self.real[index] = -self.real[index]
                self.imag[index] = -self.imag[index]

    def u3(self, target_bit, ureal, uimag):
        self.unitary(target_bit, ureal, uimag)

    def u2(self, target_bit, ureal, uimag):
        self.unitary(target_bit, ureal, uimag)

    def u1(self, target_bit, ureal, uimag):
        self.unitary(target_bit, ureal, uimag)

    def unitary(self, target_bit, ureal, uimag):
        num_task = self.total_num_amps >> 1
        size_half_block = 2**target_bit
        size_block = size_half_block * 2

        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            # store current state vector values in temp variables
            state_real_up = self.real[index_up]
            state_imag_up = self.imag[index_up]

            state_real_lo = self.real[index_lo]
            state_imag_lo = self.imag[index_lo]

            self.real[index_up] = (
                ureal[0][0] * state_real_up
                - uimag[0][0] * state_imag_up
                + ureal[0][1] * state_real_lo
                - uimag[0][1] * state_imag_lo
            )
            self.imag[index_up] = (
                ureal[0][0] * state_imag_up
                + uimag[0][0] * state_real_up
                + ureal[0][1] * state_imag_lo
                + uimag[0][1] * state_real_lo
            )

            self.real[index_lo] = (
                ureal[1][0] * state_real_up
                - uimag[1][0] * state_imag_up
                + ureal[1][1] * state_real_lo
                - uimag[1][1] * state_imag_lo
            )
            self.imag[index_lo] = (
                ureal[1][0] * state_imag_up
                + uimag[1][0] * state_real_up
                + ureal[1][1] * state_imag_lo
                + uimag[1][1] * state_real_lo
            )

    def crx(self, control_bit, target_bit, angle):
        unit_axis = [1, 0, 0]
        self.controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def cry(self, control_bit, target_bit, angle):
        unit_axis = [0, 1, 0]
        self.controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def crz(self, control_bit, target_bit, angle):
        unit_axis = [0, 0, 1]
        self.controlled_rotate_around_axis(control_bit, target_bit, angle, unit_axis)

    def controlled_rotate_around_axis(self, control_bit, target_bit, angle, unit_axis):
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

        num_task = self.total_num_amps >> 1
        size_half_block = 2**target_bit
        size_block = size_half_block * 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            control_bit = self.extract_bit(control_bit, index_up)
            if control_bit:
                # store current state vector values in temp variables
                state_real_up = self.real[index_up]
                state_imag_up = self.imag[index_up]

                state_real_lo = self.real[index_lo]
                state_imag_lo = self.imag[index_lo]

                self.real[index_up] = (
                    alpha_real * state_real_up
                    - alpha_imag * state_imag_up
                    - beta_real * state_real_lo
                    - beta_imag * state_imag_lo
                )
                self.imag[index_up] = (
                    alpha_real * state_imag_up
                    + alpha_imag * state_real_up
                    - beta_real * state_imag_lo
                    + beta_imag * state_real_lo
                )

                self.real[index_lo] = (
                    beta_real * state_real_up
                    - beta_imag * state_imag_up
                    + alpha_real * state_real_lo
                    + alpha_imag * state_imag_lo
                )
                self.imag[index_lo] = (
                    beta_real * state_imag_up
                    + beta_imag * state_real_up
                    + alpha_real * state_imag_lo
                    - alpha_imag * state_real_lo
                )

    def rotate_around_axis(self, target_bit, angle, unit_axis):
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

        num_task = self.total_num_amps >> 1
        size_half_block = 2**target_bit
        size_block = size_half_block * 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            # store current state vector values in temp variables
            state_real_up = self.real[index_up]
            state_imag_up = self.imag[index_up]

            state_real_lo = self.real[index_lo]
            state_imag_lo = self.imag[index_lo]

            self.real[index_up] = (
                alpha_real * state_real_up
                - alpha_imag * state_imag_up
                - beta_real * state_real_lo
                - beta_imag * state_imag_lo
            )
            self.imag[index_up] = (
                alpha_real * state_imag_up
                + alpha_imag * state_real_up
                - beta_real * state_imag_lo
                + beta_imag * state_real_lo
            )

            self.real[index_lo] = (
                beta_real * state_real_up
                - beta_imag * state_imag_up
                + alpha_real * state_real_lo
                + alpha_imag * state_imag_lo
            )
            self.imag[index_lo] = (
                beta_real * state_imag_up
                + beta_imag * state_real_up
                + alpha_real * state_imag_lo
                - alpha_imag * state_real_lo
            )

    def x1(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def y1(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def z1(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def sqrtx(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def sqrtxdg(self, target, ureal, uimag):
        self.apply_matrix2(target, ureal, uimag)

    def csqrtx(self, control_bit, target_bit, ureal, uimage):
        self.controlled_unitary(control_bit, target_bit, ureal, uimage)

    def cu1(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def cu3(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def cu(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def cr(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def iswap(self, target_bit0, target_bit1, ureal, uimag):
        self.apply_matrix4(target_bit0, target_bit1, ureal, uimag)

    def get_qubit_bit_mask(self, qubits, numqubit):
        mask = 0
        for index in range(numqubit):
            mask = mask | (1 << qubits[index])

        return mask

    def apply_matrix2(self, target, ureal, uimag):
        num_task = self.total_num_amps // 2
        size_half_block = 2**target
        size_block = size_half_block * 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            state_real_up = self.real[index_up]
            state_imag_up = self.imag[index_up]

            state_real_lo = self.real[index_lo]
            state_imag_lo = self.imag[index_lo]

            self.real[index_up] = (
                ureal[0][0] * state_real_up
                - uimag[0][0] * state_imag_up
                + ureal[0][1] * state_real_lo
                - uimag[0][1] * state_imag_lo
            )
            self.imag[index_up] = (
                ureal[0][0] * state_imag_up
                + uimag[0][0] * state_real_up
                + ureal[0][1] * state_imag_lo
                + uimag[0][1] * state_real_lo
            )
            self.real[index_lo] = (
                ureal[1][0] * state_real_up
                - uimag[1][0] * state_imag_up
                + ureal[1][1] * state_real_lo
                - uimag[1][1] * state_imag_lo
            )
            self.imag[index_lo] = (
                ureal[1][0] * state_imag_up
                + uimag[1][0] * state_real_up
                + ureal[1][1] * state_imag_lo
                + uimag[1][1] * state_real_lo
            )

    def apply_matrix4(self, target0, target1, ureal, uimag):
        ctrl_mask = 0
        global_ind_start = self.total_num_amps
        num_task = self.total_num_amps >> 2
        for this_task in range(num_task):
            ind00 = self.insert_two_zero_bits(this_task, target0, target1)
            this_global_ind00 = ind00 + global_ind_start
            if ctrl_mask and ((ctrl_mask & this_global_ind00) != ctrl_mask):
                continue

            # inds of |..0..1..>, |..1..0..> and |..1..1..>
            ind01 = self.flip_bit(ind00, target0)
            ind10 = self.flip_bit(ind00, target1)
            ind11 = self.flip_bit(ind01, target1)

            # extract statevec amplitudes
            re00 = self.real[ind00]
            im00 = self.imag[ind00]
            re01 = self.real[ind01]
            im01 = self.imag[ind01]
            re10 = self.real[ind10]
            im10 = self.imag[ind10]
            re11 = self.real[ind11]
            im11 = self.imag[ind11]

            # apply u * {amp00, amp01, amp10, amp11}
            self.real[ind00] = (
                ureal[0][0] * re00
                - uimag[0][0] * im00
                + ureal[0][1] * re01
                - uimag[0][1] * im01
                + ureal[0][2] * re10
                - uimag[0][2] * im10
                + ureal[0][3] * re11
                - uimag[0][3] * im11
            )
            self.imag[ind00] = (
                uimag[0][0] * re00
                + ureal[0][0] * im00
                + uimag[0][1] * re01
                + ureal[0][1] * im01
                + uimag[0][2] * re10
                + ureal[0][2] * im10
                + uimag[0][3] * re11
                + ureal[0][3] * im11
            )

            self.real[ind01] = (
                ureal[1][0] * re00
                - uimag[1][0] * im00
                + ureal[1][1] * re01
                - uimag[1][1] * im01
                + ureal[1][2] * re10
                - uimag[1][2] * im10
                + ureal[1][3] * re11
                - uimag[1][3] * im11
            )
            self.imag[ind01] = (
                uimag[1][0] * re00
                + ureal[1][0] * im00
                + uimag[1][1] * re01
                + ureal[1][1] * im01
                + uimag[1][2] * re10
                + ureal[1][2] * im10
                + uimag[1][3] * re11
                + ureal[1][3] * im11
            )

            self.real[ind10] = (
                ureal[2][0] * re00
                - uimag[2][0] * im00
                + ureal[2][1] * re01
                - uimag[2][1] * im01
                + ureal[2][2] * re10
                - uimag[2][2] * im10
                + ureal[2][3] * re11
                - uimag[2][3] * im11
            )
            self.imag[ind10] = (
                uimag[2][0] * re00
                + ureal[2][0] * im00
                + uimag[2][1] * re01
                + ureal[2][1] * im01
                + uimag[2][2] * re10
                + ureal[2][2] * im10
                + uimag[2][3] * re11
                + ureal[2][3] * im11
            )

            self.real[ind11] = (
                ureal[3][0] * re00
                - uimag[3][0] * im00
                + ureal[3][1] * re01
                - uimag[3][1] * im01
                + ureal[3][2] * re10
                - uimag[3][2] * im10
                + ureal[3][3] * re11
                - uimag[3][3] * im11
            )
            self.imag[ind11] = (
                uimag[3][0] * re00
                + ureal[3][0] * im00
                + uimag[3][1] * re01
                + ureal[3][1] * im01
                + uimag[3][2] * re10
                + ureal[3][2] * im10
                + uimag[3][3] * re11
                + ureal[3][3] * im11
            )

    def controlled_unitary(self, control_bit, target_bit, ureal, uimag):
        num_task = self.total_num_amps >> 1
        size_half_block = 1 << target_bit
        size_block = 2 * size_half_block

        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            control_bit = self.extract_bit(control_bit, index_up)
            if control_bit:
                # store current state vector values in temp variables.
                state_real_up = self.real[index_up]
                state_imag_up = self.imag[index_up]

                state_real_lo = self.real[index_lo]
                state_imag_lo = self.imag[index_lo]

                self.real[index_up] = (
                    ureal[0][0] * state_real_up
                    - uimag[0][0] * state_imag_up
                    + ureal[0][1] * state_real_lo
                    - uimag[0][1] * state_imag_lo
                )
                self.imag[index_up] = (
                    ureal[0][0] * state_imag_up
                    + uimag[0][0] * state_real_up
                    + ureal[0][1] * state_imag_lo
                    + uimag[0][1] * state_real_lo
                )

                self.real[index_lo] = (
                    ureal[1][0] * state_real_up
                    - uimag[1][0] * state_imag_up
                    + ureal[1][1] * state_real_lo
                    - uimag[1][1] * state_imag_lo
                )
                self.imag[index_lo] = (
                    ureal[1][0] * state_imag_up
                    + uimag[1][0] * state_real_up
                    + ureal[1][1] * state_imag_lo
                    + uimag[1][1] * state_real_lo
                )

    def insert_two_zero_bits(self, number, bit1, bit2):
        small = bit1 if bit1 < bit2 else bit2
        big = bit2 if bit1 < bit2 else bit1
        return self.insert_zero_bit(self.insert_zero_bit(number, small), big)

    def insert_zero_bit(self, number, index):
        left = (number >> index) << index
        right = number - left
        return (left << 1) ^ right

    def flip_bit(self, number, bit_ind):
        return number ^ (1 << bit_ind)

    def extract_bit(self, ctrl, index):
        return (index & (2**ctrl)) // (2**ctrl)

    def multi_controlled_multi_qubit_not(
        self, control_bits, num_control_bits, target_bits, num_target_bits
    ):
        ctrl_mask = self.get_qubit_bit_mask(control_bits, num_control_bits)
        targ_mask = self.get_qubit_bit_mask(target_bits, num_target_bits)

        for amp_ind in range(self.total_num_amps):

            # /* it may be a premature optimisation to remove the seemingly wasteful continues below,
            #  * because the maximum skipped amplitudes is 1/2 that stored in the node
            #  * (e.g. since this function is not called if all amps should be skipped via controls),
            #  * and since we're memory-bandwidth bottlenecked.
            #  */

            # // although amps are local, we may still be running in distributed mode,
            # // and hence need to consult the global index to determine the values of
            # // the control qubits
            global_ind = amp_ind

            # // modify amplitude only if control qubits are 1 for this state
            if ctrl_mask and ((ctrl_mask & global_ind) != ctrl_mask):
                continue

            mate_ind = amp_ind ^ targ_mask

            # // if the mate is behind, it was already processed
            if mate_ind < amp_ind:
                continue

            mate_re = self.real[mate_ind]
            mate_im = self.imag[mate_ind]

            # // swap amp with mate
            self.real[mate_ind] = self.real[amp_ind]
            self.imag[mate_ind] = self.imag[amp_ind]
            self.real[amp_ind] = mate_re
            self.imag[amp_ind] = mate_im

    def controlled_two_qubit_unitary(
        self, control_bit, target_bit0, targetbi_bit1, ureal, uimag
    ):
        num_task = self.total_num_amps >> 2
        for this_task in range(num_task):
            #  determine ind00 of |..0..0..>.
            ind00 = self.insert_two_zero_bits(this_task, target_bit0, targetbi_bit1)

            #  skip amplitude if controls aren't in 1 state (overloaded for speed).
            this_global_ind00 = ind00
            if control_bit and ((control_bit & this_global_ind00) != control_bit):
                continue

            #  inds of |..0..1..>, |..1..0..> and |..1..1..>.
            ind01 = self.flip_bit(ind00, target_bit0)
            ind10 = self.flip_bit(ind00, targetbi_bit1)
            ind11 = self.flip_bit(ind01, targetbi_bit1)

            #  extract statevec amplitudes
            re00 = self.real[ind00]
            im00 = self.imag[ind00]
            re01 = self.real[ind01]
            im01 = self.imag[ind01]
            re10 = self.real[ind10]
            im10 = self.imag[ind10]
            re11 = self.real[ind11]
            im11 = self.imag[ind11]

            #  apply u * {amp00, amp01, amp10, amp11}.
            self.real[ind00] = (
                ureal[0][0] * re00
                - uimag[0][0] * im00
                + ureal[0][1] * re01
                - uimag[0][1] * im01
                + ureal[0][2] * re10
                - uimag[0][2] * im10
                + ureal[0][3] * re11
                - uimag[0][3] * im11
            )
            self.imag[ind00] = (
                uimag[0][0] * re00
                + ureal[0][0] * im00
                + uimag[0][1] * re01
                + ureal[0][1] * im01
                + uimag[0][2] * re10
                + ureal[0][2] * im10
                + uimag[0][3] * re11
                + ureal[0][3] * im11
            )

            self.real[ind01] = (
                ureal[1][0] * re00
                - uimag[1][0] * im00
                + ureal[1][1] * re01
                - uimag[1][1] * im01
                + ureal[1][2] * re10
                - uimag[1][2] * im10
                + ureal[1][3] * re11
                - uimag[1][3] * im11
            )
            self.imag[ind01] = (
                uimag[1][0] * re00
                + ureal[1][0] * im00
                + uimag[1][1] * re01
                + ureal[1][1] * im01
                + uimag[1][2] * re10
                + ureal[1][2] * im10
                + uimag[1][3] * re11
                + ureal[1][3] * im11
            )

            self.real[ind10] = (
                ureal[2][0] * re00
                - uimag[2][0] * im00
                + ureal[2][1] * re01
                - uimag[2][1] * im01
                + ureal[2][2] * re10
                - uimag[2][2] * im10
                + ureal[2][3] * re11
                - uimag[2][3] * im11
            )
            self.imag[ind10] = (
                uimag[2][0] * re00
                + ureal[2][0] * im00
                + uimag[2][1] * re01
                + ureal[2][1] * im01
                + uimag[2][2] * re10
                + ureal[2][2] * im10
                + uimag[2][3] * re11
                + ureal[2][3] * im11
            )

            self.real[ind11] = (
                ureal[3][0] * re00
                - uimag[3][0] * im00
                + ureal[3][1] * re01
                - uimag[3][1] * im01
                + ureal[3][2] * re10
                - uimag[3][2] * im10
                + ureal[3][3] * re11
                - uimag[3][3] * im11
            )
            self.imag[ind11] = (
                uimag[3][0] * re00
                + ureal[3][0] * im00
                + uimag[3][1] * re01
                + ureal[3][1] * im01
                + uimag[3][2] * re10
                + ureal[3][2] * im10
                + uimag[3][3] * re11
                + ureal[3][3] * im11
            )

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
            outcome = 1 if random.random() > zero_prob else 0

        outcome_prob = zero_prob if outcome == 0 else 1 - zero_prob
        return outcome, outcome_prob

    def collapse_to_know_prob_outcome(self, target, outcome, outcome_prob):
        num_task = self.total_num_amps // 2
        size_half_block = 2**target
        size_block = 2 * size_half_block

        renorm = 1 / math.sqrt(outcome_prob)
        if outcome == 0:
            for this_task in range(num_task):
                this_block = this_task // size_half_block
                index = this_block * size_block + this_task % size_half_block

                self.real[index] = self.real[index] * renorm
                self.imag[index] = self.imag[index] * renorm

                self.real[index + size_half_block] = 0
                self.imag[index + size_half_block] = 0
        else:
            for this_task in range(num_task):
                this_block = this_task // size_half_block
                index = this_block * size_block + this_task % size_half_block

                self.real[index] = 0
                self.imag[index] = 0

                self.real[index + size_half_block] = (
                    self.real[index + size_half_block] * renorm
                )
                self.imag[index + size_half_block] = (
                    self.imag[index + size_half_block] * renorm
                )

    def calc_prob_of_outcome(self, target, outcome):
        outcome_prob = self.find_prob_of_zero(target)
        if outcome == 1:
            outcome_prob = 1.0 - outcome_prob
        return outcome_prob

    def find_prob_of_zero(self, target):
        num_task = self.total_num_amps // 2
        size_half_block = 2**target
        size_block = size_half_block * 2
        total_prob = 0.0

        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index = this_block * size_block + this_task % size_half_block

            total_prob += (
                self.real[index] * self.real[index]
                + self.imag[index] * self.imag[index]
            )

        return total_prob

    def get_prob(self, index):
        """
        Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: index in state vector of probability amplitudes

        Returns:
            the probability of target index
        """
        if index < 0 or index >= self.total_num_amps:
            raise ValueError(f"{index} is illegal parameter.")

        real = self.real[index]
        imag = self.imag[index]
        # TODO:doing
        # print("in pysim=", real * real + imag * imag)
        return real * real + imag * imag

    def get_probs(self, qubits):
        """Get all probabilities of circuit.

        Returns:
            An array contains all probabilities of circuit.
        """
        num_outcome_probs = len(qubits)
        outcome_probs = [0] * (2**num_outcome_probs)
        for i in range(self.total_num_amps):
            outcome_ind = 0
            for q in range(num_outcome_probs):
                outcome_ind += self.extract_bit(qubits[q], i) * (2**q)

            real = self.real[i]
            imag = self.imag[i]
            prob = real * real + imag * imag
            outcome_probs[outcome_ind] += prob

        return outcome_probs

    # TODO: to matrix
    def get_statevector(self):
        """
        Get the current state vector of probability amplitudes for a set of qubits
        """
        # todo better in float or ndarray
        state_list = []
        for i in range(self.total_num_amps):
            real = self.real[i]
            imag = self.imag[i]
            # TODO: need to improve.
            if self.real[i] > -1e-15 and self.real[i] < 1e-15:
                real = 0
            if self.imag[i] > -1e-15 and self.imag[i] < 1e-15:
                imag = 0
            state = str(real) + ", " + str(imag)
            state_list.append(state)
        return state_list

    def apply_param_named_phase(
        self,
        qubits,
        num_qubits_per_reg,
        num_regs,
        encoding,
        phase_func_name,
        params,
        override_inds,
        override_phases,
        num_overrides,
        conj,
    ):
        i = 0
        max_num_regs_apply_arbitrary_phase = 100
        phase_inds = [0] * max_num_regs_apply_arbitrary_phase
        for index in range(self.total_num_amps):
            flat_ind = 0
            for r in range(num_regs):
                phase_inds[r] = 0
                if encoding == BitEncoding.UNSIGNED:
                    for q in range(num_qubits_per_reg[r]):
                        phase_inds[r] += (2**q) * self.extract_bit(
                            qubits[flat_ind], index
                        )
                        flat_ind += 1
                elif encoding == BitEncoding.TWOS_COMPLEMENT:
                    for q in range(num_qubits_per_reg[r] - 1):
                        phase_inds[r] += (2**q) * self.extract_bit(
                            qubits[flat_ind], index
                        )
                        flat_ind += 1
                    if self.extract_bit(qubits[flat_ind], index) == 1:
                        phase_inds[r] -= 2 ** (num_qubits_per_reg[r] - 1)
                    flat_ind += 1

            for i in range(num_overrides):
                found = 1
                for r in range(num_regs):
                    if phase_inds[r] != override_inds[i * num_regs + r]:
                        found = 0
                        break
                if 1 == found:
                    break

            phase = 0
            if i < num_overrides:
                phase = override_phases[i]
            else:
                if (
                    phase_func_name == PhaseFunc.NORM
                    or phase_func_name == PhaseFunc.INVERSE_NORM
                    or phase_func_name == PhaseFunc.SCALED_NORM
                    or phase_func_name == PhaseFunc.SCALED_INVERSE_NORM
                    or phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_NORM
                ):
                    norm = 0
                    if phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_NORM:
                        for r in range(num_regs):
                            norm += (phase_inds[r] - params[2 + r]) * (
                                phase_inds[r] - params[2 + r]
                            )
                    else:
                        for r in range(num_regs):
                            norm += phase_inds[r] * phase_inds[r]
                    norm = math.sqrt(norm)

                    if phase_func_name == PhaseFunc.NORM:
                        phase = norm
                    elif phase_func_name == PhaseFunc.INVERSE_NORM:
                        if norm == 0.0:
                            phase = params[0]
                        else:
                            phase = 1 // norm
                    elif phase_func_name == PhaseFunc.SCALED_NORM:
                        phase = params[0] * norm
                    elif (
                        phase_func_name == PhaseFunc.SCALED_INVERSE_NORM
                        or phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_NORM
                    ):
                        if norm <= REAL_EPS:
                            phase = params[1]
                        else:
                            phase = params[0] // norm
                elif (
                    phase_func_name == PhaseFunc.PRODUCT
                    or phase_func_name == PhaseFunc.INVERSE_PRODUCT
                    or phase_func_name == PhaseFunc.SCALED_PRODUCT
                    or phase_func_name == PhaseFunc.SCALED_INVERSE_PRODUCT
                ):
                    prod = 1
                    for r in range(num_regs):
                        prod *= phase_inds[r]

                    if phase_func_name == PhaseFunc.PRODUCT:
                        phase = prod
                    elif phase_func_name == PhaseFunc.INVERSE_PRODUCT:
                        if prod == 0.0:
                            phase = params[0]
                        else:
                            phase = 1 // prod
                    elif phase_func_name == PhaseFunc.SCALED_PRODUCT:
                        phase = params[0] * prod
                    elif phase_func_name == PhaseFunc.SCALED_INVERSE_PRODUCT:
                        if prod == 0.0:
                            phase = params[1]
                        else:
                            phase = params[0] // prod
                elif (
                    phase_func_name == PhaseFunc.DISTANCE
                    or phase_func_name == PhaseFunc.INVERSE_DISTANCE
                    or phase_func_name == PhaseFunc.SCALED_DISTANCE
                    or phase_func_name == PhaseFunc.SCALED_INVERSE_DISTANCE
                    or phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_DISTANCE
                ):
                    dist = 0
                    if phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_DISTANCE:
                        for r in range(0, num_regs, 2):
                            dist += (
                                phase_inds[r] - phase_inds[r + 1] - params[2 + r / 2]
                            ) * (phase_inds[r] - phase_inds[r + 1] - params[2 + r / 2])
                    else:
                        for r in range(0, num_regs, 2):
                            dist += (phase_inds[r + 1] - phase_inds[r]) * (
                                phase_inds[r + 1] - phase_inds[r]
                            )
                    dist = math.sqrt(dist)

                    if phase_func_name == PhaseFunc.DISTANCE:
                        phase = dist
                    elif phase_func_name == PhaseFunc.INVERSE_DISTANCE:
                        if dist == 0.0:
                            phase = params[0]
                        else:
                            phase = 1 // dist
                    elif phase_func_name == PhaseFunc.SCALED_DISTANCE:
                        phase = params[0] * dist
                    elif (
                        phase_func_name == PhaseFunc.SCALED_INVERSE_DISTANCE
                        or phase_func_name == PhaseFunc.SCALED_INVERSE_SHIFTED_DISTANCE
                    ):
                        if dist <= REAL_EPS:
                            phase = params[1]
                        else:
                            phase = params[0] // dist

            if conj:
                phase *= -1

            c = math.cos(phase)
            s = math.sin(phase)
            re = self.real[index]
            im = self.imag[index]

            self.real[index] = re * c - im * s
            self.imag[index] = re * s + im * c

    def insert_zero_bit(self, number, index):
        left = (number >> index) << index
        right = number - left
        return (left << 1) ^ right

    def insert_two_zero_bits(self, number, bit1, bit2):
        small = 0
        if bit1 < bit2:
            small = bit1
        else:
            small = bit2
        big = 0
        if bit1 < bit2:
            big = bit2
        else:
            big = bit1
        return self.insert_zero_bit(self.insert_zero_bit(number, small), big)

    def swap_qubit_amps(self, qb1, qb2):
        num_task = self.total_num_amps >> 2
        for this_task in range(num_task):
            ind00 = self.insert_two_zero_bits(this_task, qb1, qb2)
            ind01 = self.flip_bit(ind00, qb1)
            ind10 = self.flip_bit(ind00, qb2)
            re01 = self.real[ind01]
            im01 = self.imag[ind01]
            re10 = self.real[ind10]
            im10 = self.imag[ind10]
            self.real[ind01] = re10
            self.real[ind10] = re01
            self.imag[ind01] = im10
            self.imag[ind10] = im01

    # TODO:need to improve.
    def paulix_local(self, work_real, work_imag, target_qubit):
        """pauli-X"""
        # jpaulix_local(target_qubit, self.total_num_amps, List(work_real), List(work_imag))
        size_half_block = 2**target_qubit
        size_block = 2 * size_half_block

        num_task = self.total_num_amps // 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            state_real_up = work_real[index_up]
            state_imag_up = work_imag[index_up]

            work_real[index_up] = work_real[index_lo]
            work_imag[index_up] = work_imag[index_lo]

            work_real[index_lo] = state_real_up
            work_imag[index_lo] = state_imag_up

    def pauliy_local(self, work_real, work_imag, target_qubit, conj_fac):
        """pauli-Y"""
        size_half_block = 2**target_qubit
        size_block = 2 * size_half_block

        num_task = self.total_num_amps // 2
        for this_task in range(num_task):
            this_block = this_task // size_half_block
            index_up = this_block * size_block + this_task % size_half_block
            index_lo = index_up + size_half_block

            state_real_up = work_real[index_up]
            state_imag_up = work_imag[index_up]

            work_real[index_up] = conj_fac * work_imag[index_lo]
            work_imag[index_up] = conj_fac * (-work_real[index_lo])

            work_real[index_lo] = conj_fac * (-state_imag_up)
            work_imag[index_lo] = conj_fac * state_real_up

    def phase_shift_by_term(self, real, imag, target_qubit, term_real, term_imag):
        state_vec_size = self.total_num_amps
        cos_angle = term_real
        sin_angle = term_imag
        for index in range(state_vec_size):
            target_bit = self.extract_bit(target_qubit, index)
            if target_bit:
                state_real_lo = real[index]
                state_imag_lo = imag[index]

                real[index] = cos_angle * state_real_lo - sin_angle * state_imag_lo
                imag[index] = sin_angle * state_real_lo + cos_angle * state_imag_lo

    def pauliz_local(self, work_real, work_imag, target_qubit):
        term_real = -1
        term_imag = 0
        self.phase_shift_by_term(
            work_real, work_imag, target_qubit, term_real, term_imag
        )

    def calc_inner_product_local(self, bra_real, bra_imag, ket_real, ket_imag):
        inner_prod_real = 0
        inner_prod_imag = 0
        for index in range(self.total_num_amps):
            bra_re = bra_real[index]
            bra_im = bra_imag[index]
            ket_re = ket_real[index]
            ket_im = ket_imag[index]
            inner_prod_real += bra_re * ket_re + bra_im * ket_im
            inner_prod_imag += bra_re * ket_im - bra_im * ket_re
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
        work_real = [0] * self.total_num_amps
        work_imag = [0] * self.total_num_amps
        for i in range(self.total_num_amps):
            work_real[i] = self.real[i]
            work_imag[i] = self.imag[i]
        for pauli_op in pauli_prod_list:
            op_type = pauli_op["oper_type"]
            if op_type == PauliOpType.PAULI_X.value:
                self.paulix_local(work_real, work_imag, pauli_op["target"])
            elif op_type == PauliOpType.PAULI_Y.value:
                self.pauliy_local(work_real, work_imag, pauli_op["target"], 1)
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
        num_qb = self.qubits
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
