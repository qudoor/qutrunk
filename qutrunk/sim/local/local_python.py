import math
from copy import deepcopy

import numpy as np
from numpy import pi

from qutrunk.sim.local.pysim import Simulator
from qutrunk.tools.function_time import timefn
from .exceptions import LocalBackendError


class MeasureResult:
    def __init__(self, id=0, value=0):
        # TODO: id是关键字，不建议使用
        self.id = id
        self.value = value


class OutcomeResult:
    def __init__(self, bit_str="", count=0):
        self.bitstr = bit_str
        self.count = count


class Result:
    def __init__(self):
        self.measureSet = []
        self.outcomeSet = []


class BackendLocalPython:
    def __init__(self):
        self.sim = Simulator()
        self.gate_map = {
            "H": "h",
            "P": "p",
            "CP": "cp",
            "R": "r",
            "Rx": "rx",
            "Rxx": "rxx",
            "Ry": "ry",
            "Ryy": "ryy",
            "Rz": "rz",
            "Rzz": "rzz",
            "NOT": "x",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "S": "s",
            "T": "t",
            "Sdg": "sdg",
            "Tdg": "tdg",
            "SqrtSwap": "sqrtswap",
            "Swap": "swap",
            "CNOT": "cnot",
            "MCX": "cnot",
            "CY": "cy",
            "MCZ": "cz",
            "U3": "u3",
            "U2": "u2",
            "U1": "u1",
            "CRx": "crx",
            "CRy": "cry",
            "CRz": "crz",
            "X1": "x1",
            "Y1": "y1",
            "Z1": "z1",
            "CU1": "cu1",
            "CU3": "cu3",
            "U": "u",
            "CU": "cu",
            "CR": "cr",
            "iSwap": "iswap",
            "Barrier": "barrier",
            "SqrtX": "sprtx",
            "Id": "id",
            "CH": "ch",
            "SqrtXdg": "sqrtxdg",
            "CSqrtX": "csqrtx",
            "CSwap": "cswap",
            "AMP": "amp"
        }
        self.cmds = []
        self.result = Result()

    @timefn
    def init(self, qubits, show):
        self.sim.create_qureg(qubits)
        self.sim.init_zero_state()

    @timefn
    def send_circuit(self, circuit, final):
        """
        Send the quantum circuit to local backend.

        Args:
            circuit: quantum circuit to send.
            final: True if quantum circuit finish, default False, \
                when final==True The backend program will release the computing resources.
        """
        start = circuit.cmd_cursor
        cmds = circuit.cmds[start:]
        for cmd in cmds:
            self.cmds.append(cmd)
            self.exec_cmd(cmd)
        if final:
            self.pack_result()

    @timefn
    def run(self, shots):
        """Run quantum circuit.

        Args:
            shots: circuit run times, for sampling.

        Returns:
            result: the Result object contain circuit running outcome.
        """
        run_times = shots - 1
        while run_times > 0:
            self.result.measureSet = []

            # reset state
            self.sim.init_zero_state()

            for cmd in self.cmds:
                self.exec_cmd(cmd)
            self.pack_result()
            run_times -= 1

        self.result.outcomeSet.sort(key=lambda a: a.bitstr)
        return self.result

    @timefn
    def get_prob_amp(self, index):
        """
        Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: index in state vector of probability amplitudes

        Returns:
            the probability of target index
        """
        return self.sim.get_prob_amp(index)

    @timefn
    def get_prob_outcome(self, target, outcome):
        """
        Get the probability of a specified qubit being measured in the given outcome (0 or 1)

        Args:
            qubit: the specified qubit to be measured
            outcome: the qubit measure result(0 or 1)

        Returns:
            the probability of target qubit
        """
        return self.sim.get_prob_outcome(target, outcome)

    @timefn
    def get_prob_all_outcome(self, qubits):
        """
        Get outcomeProbs with the probabilities of every outcome of the sub-register contained in qureg

        Args:
            qubits: the sub-register contained in qureg

        Returns:
            An array contains probability of target qubits
        """
        return self.sim.get_prob_all_outcome(qubits)

    @timefn
    def get_all_state(self):
        """
        Get the current state vector of probability amplitudes for a set of qubits
        """
        return self.sim.get_all_state()

    @timefn
    def qft(self, qubits):
        """
        Applies the quantum Fourier transform (QFT) to a specific subset of qubits of the register qureg

        Args:
            qubits: a list of the qubits to operate the QFT upon
        """
        if qubits:
            self.sim.apply_qft(qubits, len(qubits))
        else:
            self.sim.apply_full_qft()

    @timefn
    def get_expec_pauli_prod(self, pauli_prod_list):
        """
        Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: a list contains the indices of the target qubits,\
                the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            the expected value of a product of Pauli operators.
        """
        return self.sim.get_expec_pauli_prod(pauli_prod_list)

    @timefn
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
        return self.sim.get_expec_pauli_sum(oper_type_list, term_coeff_list)

    def exec_cmd(self, cmd):
        if str(cmd.gate) == "Measure":
            res = self.sim.measure(cmd.targets[0])
            mr = MeasureResult(cmd.targets[0], res)
            self.result.measureSet.append(mr)
            return

        return getattr(self, self.gate_map[str(cmd.gate)])(cmd)

    def pack_result(self):
        self.result.measureSet.sort(key=lambda a: a.id)
        bit_str = ""
        measureSet = self.result.measureSet
        for i in range(len(measureSet)):
            m = measureSet[i]
            bit_str += str(m.value)

        index = -1
        outcomeSet = self.result.outcomeSet
        for i in range(len(outcomeSet)):
            out = outcomeSet[i]
            if out.bitstr == bit_str:
                index = i
                break

        if index >= 0:
            self.result.outcomeSet[index].count += 1
        else:
            out = OutcomeResult(bit_str, 1)
            self.result.outcomeSet.append(out)

    def h(self, cmd):
        """the single-qubit Hadamard gate.

        Args:
           cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"h gate takes exactly one targets argument({targets_len} given)."
            )

        self.sim.hadamard(cmd.targets[0])

    def ch(self, cmd):
        """the controlled single-qubit Hadamard gate.

        Args:
           cmd: the Command object.

        # inverse is the same.
        """
        if len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        factor = 1 / math.sqrt(2)
        # real part of complex number.
        ureal = np.array(
            [
                [1 * factor, 1 * factor],
                [1 * factor, -1 * factor],
            ]
        )

        # imaginary part of complex number.
        uimag = np.array(
            [
                [0, 0],
                [0, 0],
            ]
        )

        self.sim.ch(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def cnot(self, cmd):
        if len(cmd.targets) == 1 and len(cmd.controls) == 1:
            self.sim.control_not(cmd.controls[0], cmd.targets[0])
        else:
            self.sim.multi_controlled_multi_qubit_not(
                cmd.controls, len(cmd.controls), cmd.targets, len(cmd.targets)
            )

    def p(self, cmd):
        """Shift the phase between |0> and |1> of a single qubit by a given angle.

        Args:
            cmd: the Command object.
        """
        t_num = len(cmd.targets)
        r_num = len(cmd.rotation)
        if (t_num != 1) or (r_num != 1):
            raise LocalBackendError(
                f"p gate takes exactly one targets and one rotation argument(targets:{t_num}, rotation:{r_num} given)."
            )

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation
        self.sim.phase_shift(cmd.targets[0], rotation)

    def cp(self, cmd):
        """Controlled-Phase gate.

        Args:
            cmd: the Command object.
        """
        t_num = len(cmd.targets)
        r_num = len(cmd.rotation)
        c_num = len(cmd.controls)
        if (t_num != 1) or (r_num != 1) or (c_num != 1):
            raise LocalBackendError(
                f"cp gate takes exactly one targets, one rotation and one controls argument(targets:{t_num}, "
                f"rotation:{r_num}, controls:{c_num} given)."
            )

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation
        self.sim.controlled_phase_shift(cmd.controls[0], cmd.targets[0], rotation)

    def r(self, cmd):
        """r gate.

        Args:
            cmd: the Command object.
        """
        t_num = len(cmd.targets)
        r_num = len(cmd.rotation)
        if (t_num != 1) or (r_num != 2):
            raise LocalBackendError(
                f"r gate takes exactly one targets and two rotation argument(targets:{t_num}, rotation:{r_num} given)."
            )

        theta = cmd.rotation[0]
        if cmd.inverse:
            theta = -theta
        phi = cmd.rotation[1]

        # real part of complex number
        ureal = np.array(
            [
                [math.cos(theta / 2), math.sin(-phi) * math.sin(theta / 2)],
                [math.sin(phi) * math.sin(theta / 2), math.cos(theta / 2)],
            ]
        )

        # imaginary part of complex number
        uimag = np.array(
            [
                [0, -1 * math.cos(-phi) * math.sin(theta / 2)],
                [-1 * math.cos(phi) * math.sin(theta / 2), 0],
            ]
        )

        self.sim.rotate(cmd.targets[0], ureal, uimag)

    def rx(self, cmd):
        """Rotate a single qubit by a given angle around the X-axis of the Bloch-sphere.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"rx gate takes exactly one targets argument({targets_len} given)."
            )

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation
        self.sim.rotate_x(cmd.targets[0], rotation)

    def rxx(self, cmd):
        """rxx gate.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 2:
            raise LocalBackendError(
                f"rxx gate takes exactly two targets argument({targets_len} given)."
            )

        angle = cmd.rotation[0]
        if cmd.inverse:
            angle = -angle

        ureal = np.array(
            [
                [math.cos(0.5 * angle), 0, 0, 0],
                [0, math.cos(0.5 * angle), 0, 0],
                [0, 0, math.cos(0.5 * angle), 0],
                [0, 0, 0, math.cos(0.5 * angle)],
            ]
        )
        uimag = np.array(
            [
                [0, 0, 0, -1 * math.sin(0.5 * angle)],
                [0, 0, -1 * math.sin(0.5 * angle), 0],
                [0, -1 * math.sin(0.5 * angle), 0, 0],
                [-1 * math.sin(0.5 * angle), 0, 0, 0],
            ]
        )

        self.sim.apply_matrix4(cmd.targets[0], cmd.targets[1], ureal, uimag)

    def ry(self, cmd):
        """ry gate.

        Args:
           cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"ry gate takes exactly one targets argument({targets_len} given)."
            )

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation

        self.sim.rotate_y(cmd.targets[0], rotation)

    def ryy(self, cmd):
        """ryy gate.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 2:
            raise LocalBackendError(
                f"ryy gate takes exactly two targets argument({targets_len} given)."
            )

        angle = cmd.rotation[0]
        if cmd.inverse:
            angle = -angle

        # real part of complex number
        ureal = np.array(
            [
                [math.cos(0.5 * angle), 0, 0, 0],
                [0, math.cos(0.5 * angle), 0, 0],
                [0, 0, math.cos(0.5 * angle), 0],
                [0, 0, 0, math.cos(0.5 * angle)],
            ]
        )
        # imaginary part of complex number
        uimag = np.array(
            [
                [0, 0, 0, 1 * math.sin(0.5 * angle)],
                [0, 0, -1 * math.sin(0.5 * angle), 0],
                [0, -1 * math.sin(0.5 * angle), 0, 0],
                [1 * math.sin(0.5 * angle), 0, 0, 0],
            ]
        )

        self.sim.apply_matrix4(cmd.targets[0], cmd.targets[1], ureal, uimag)

    def rz(self, cmd):
        """rz gate.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"rz gate takes exactly one targets argument({targets_len} given)."
            )

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation

        self.sim.rotate_z(cmd.targets[0], rotation)

    def rzz(self, cmd):
        """rzz gate.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 2:
            raise LocalBackendError(
                f"rzz gate takes exactly two targets argument({targets_len} given)."
            )

        angle = cmd.rotation[0]
        if cmd.inverse:
            angle = -angle

        # real part of complex number
        ureal = np.array(
            [
                [math.cos(0.5 * angle), 0, 0, 0],
                [0, math.cos(0.5 * angle), 0, 0],
                [0, 0, math.cos(0.5 * angle), 0],
                [0, 0, 0, math.cos(0.5 * angle)],
            ]
        )
        # imaginary part of complex number
        uimag = np.array(
            [
                [-1 * math.sin(0.5 * angle), 0, 0, 0],
                [0, math.sin(0.5 * angle), 0, 0],
                [0, 0, math.sin(0.5 * angle), 0],
                [0, 0, 0, -1 * math.sin(0.5 * angle)],
            ]
        )

        self.sim.apply_matrix4(cmd.targets[0], cmd.targets[1], ureal, uimag)

    def x(self, cmd):
        """The single-qubit Pauli-X gate.

        Args:
            cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"x gate takes exactly one targets argument({targets_len} given)."
            )

        self.sim.pauli_x(cmd.targets[0])

    def y(self, cmd):
        """
        The single-qubit Pauli-Y gate.

            Args:
                cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"y gate takes exactly one targets argument({targets_len} given)."
            )

        self.sim.pauli_y(cmd.targets[0])

    def z(self, cmd):
        """The single-qubit Pauli-Z gate.

        Args:
          cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"z gate takes exactly one targets argument({targets_len} given)."
            )

        self.sim.pauli_z(cmd.targets[0])

    def s(self, cmd):
        """the single-qubit S gate.

        Args:
          cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"s gate takes exactly one targets argument({targets_len} given)."
            )

        if cmd.inverse:
            cmd = deepcopy(cmd)
            cmd.inverse = False
            self.sdg(cmd)
        else:
            self.sim.s_gate(cmd.targets[0])

    def t(self, cmd):
        """the single-qubit T gate.

        Args:
          cmd: the Command object.
        """
        targets_len = len(cmd.targets)
        if targets_len != 1:
            raise LocalBackendError(
                f"t gate takes exactly one targets argument({targets_len} given)."
            )

        if cmd.inverse:
            cmd = deepcopy(cmd)
            cmd.inverse = False
            self.tdg(cmd)
        else:
            self.sim.t_gate(cmd.targets[0])

    def sdg(self, cmd):
        if cmd.inverse:
            cmd_temp = deepcopy(cmd)
            cmd_temp.inverse = False
            self.s(cmd_temp)
        else:
            ureal = np.array([[1, 0], [0, 0]])
            uimag = np.array([[0, 0], [0, -1]])
            self.sim.sdg(cmd.targets[0], ureal, uimag)

    def tdg(self, cmd):
        if cmd.inverse:
            cmd_temp = deepcopy(cmd)
            cmd_temp.inverse = False
            self.t(cmd_temp)
        else:
            ureal = np.array([[1, 0], [0, 1 / math.sqrt(2)]])
            uimag = np.array([[0, 0], [0, -1 / math.sqrt(2)]])
            self.sim.tdg(cmd.targets[0], ureal, uimag)

    def sqrtswap(self, cmd):
        targets_len = len(cmd.targets)
        if targets_len != 2:
            raise LocalBackendError(
                f"sqrtswap gate takes exactly one targets argument({targets_len} given)."
            )

        ureal = []
        uimag = []
        if cmd.inverse:
            ureal = np.array(
                [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
            )
            uimag = np.array(
                [[0, 0, 0, 0], [0, -0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0, 0, 0]]
            )
        else:
            ureal = np.array(
                [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
            )
            uimag = np.array(
                [[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]]
            )
        self.sim.sqrtswap(cmd.targets[0], cmd.targets[1], ureal, uimag)

    def swap(self, cmd):
        # inverse is the same
        if len(cmd.targets) != 2:
            return

        self.sim.swap(cmd.targets[0], cmd.targets[1])

    def cy(self, cmd):
        # inverse is the same
        if len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        self.sim.cy(cmd.controls[0], cmd.targets[0])

    def cz(self, cmd):
        # inverse is the same
        ctrls = cmd.controls[:]
        ctrls.extend(cmd.targets)

        self.sim.cz(ctrls, len(ctrls))

    def u3(self, cmd):
        if len(cmd.rotation) != 3 or len(cmd.targets) != 1:
            return

        theta = cmd.rotation[0]
        phi = cmd.rotation[1]
        lam = cmd.rotation[2]
        if cmd.inverse:
            theta1 = -theta
            phi1 = -lam
            lam1 = -phi
            theta = theta1
            phi = phi1
            lam = lam1

        ureal = np.array(
            [
                [math.cos(theta / 2), -1 * math.cos(lam) * math.sin(theta / 2)],
                [
                    math.cos(phi) * math.sin(theta / 2),
                    math.cos(phi + lam) * math.cos(theta / 2),
                ],
            ]
        )
        uimag = np.array(
            [
                [0, -1 * math.sin(lam) * math.sin(theta / 2)],
                [
                    math.sin(phi) * math.sin(theta / 2),
                    math.sin(phi + lam) * math.cos(theta / 2),
                ],
            ]
        )

        self.sim.u3(cmd.targets[0], ureal, uimag)

    def u2(self, cmd):
        if len(cmd.rotation) != 2 or len(cmd.targets) != 1:
            return

        phi = cmd.rotation[0]
        lam = cmd.rotation[1]

        if cmd.inverse:
            phi1 = -lam - pi
            lam1 = -phi + pi
            phi = phi1
            lam = lam1

        factor = 1 / math.sqrt(2)
        ureal = np.array(
            [
                [1 * factor, -factor * math.cos(lam)],
                [factor * math.cos(phi), factor * math.cos(phi + lam)],
            ]
        )
        uimag = np.array(
            [
                [0, -factor * math.sin(lam)],
                [factor * math.sin(phi), factor * math.sin(phi + lam)],
            ]
        )

        self.sim.u2(cmd.targets[0], ureal, uimag)

    def u1(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1:
            return

        alpha = cmd.rotation[0]
        if cmd.inverse:
            alpha = -alpha

        ureal = np.array([[1, 0], [0, math.cos(alpha)]])
        uimag = np.array([[0, 0], [0, math.sin(alpha)]])

        self.sim.u1(cmd.targets[0], ureal, uimag)

    def crx(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation

        self.sim.crx(cmd.controls[0], cmd.targets[0], rotation)

    def cry(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation

        self.sim.cry(cmd.controls[0], cmd.targets[0], rotation)

    def crz(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        rotation = cmd.rotation[0]
        if cmd.inverse:
            rotation = -rotation

        self.sim.crz(cmd.controls[0], cmd.targets[0], rotation)

    def x1(self, cmd):
        if len(cmd.targets) != 1:
            return

        ureal = []
        uimag = []
        if cmd.inverse:
            factor = math.sqrt(2)
            ureal = np.array([[0.5 * factor, 0], [0, 0.5 * factor]])
            uimag = np.array([[0, 0.5 * factor], [0.5 * factor, 0]])
        else:
            factor = 1 / math.sqrt(2)
            ureal = np.array([[1 * factor, 0], [0, 1 * factor]])
            uimag = np.array([[0, -1 * factor], [-1 * factor, 0]])

        self.sim.x1(cmd.targets[0], ureal, uimag)

    def y1(self, cmd):
        if len(cmd.targets) != 1:
            return

        ureal = []
        uimag = []
        if cmd.inverse:
            factor = math.sqrt(2)
            ureal = np.array(
                [[0.5 * factor, 0.5 * factor], [-0.5 * factor, 0.5 * factor]]
            )
            uimag = np.array([[0, 0], [0, 0]])
        else:
            factor = 1 / math.sqrt(2)
            ureal = np.array([[1 * factor, -1 * factor], [1 * factor, 1 * factor]])
            uimag = np.array([[0, 0], [0, 0]])

        self.sim.y1(cmd.targets[0], ureal, uimag)

    def z1(self, cmd):
        if len(cmd.targets) != 1:
            return

        rotation = pi / 4.0
        if cmd.inverse:
            rotation = -rotation

        ureal = np.array([[math.cos(-rotation), 0], [0, math.cos(rotation)]])
        uimag = np.array([[math.sin(-rotation), 0], [0, math.sin(rotation)]])

        self.sim.z1(cmd.targets[0], ureal, uimag)

    def cu1(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        alpha = cmd.rotation[0]

        if cmd.inverse:
            alpha = -alpha

        ureal = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, math.cos(alpha)]]
        )
        uimag = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, math.sin(alpha)]]
        )

        self.sim.cu1(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def cu3(self, cmd):
        if len(cmd.rotation) != 3 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        theta = cmd.rotation[0]
        phi = cmd.rotation[1]
        lam = cmd.rotation[2]
        if cmd.inverse:
            theta1 = -theta
            phi1 = -lam
            lam1 = -phi
            theta = theta1
            phi = phi1
            lam = lam1

        ureal = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(theta / 2), 0, -math.cos(lam) * math.sin(theta / 2)],
                [0, 0, 1, 0],
                [
                    0,
                    math.cos(phi) * math.sin(theta / 2),
                    0,
                    math.cos(phi + lam) * math.cos(theta / 2),
                ],
            ]
        )
        uimag = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, -math.sin(lam) * math.sin(theta / 2)],
                [0, 0, 0, 0],
                [
                    0,
                    math.sin(phi) * math.sin(theta / 2),
                    0,
                    math.sin(phi + lam) * math.cos(theta / 2),
                ],
            ]
        )

        self.sim.cu3(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def u(self, cmd):
        self.u3(cmd)

    def cu(self, cmd):
        if len(cmd.rotation) != 4 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        theta = cmd.rotation[0]
        phi = cmd.rotation[1]
        lam = cmd.rotation[2]
        gamma = cmd.rotation[3]
        if cmd.inverse:
            theta1 = -theta
            phi1 = -lam
            lam1 = -phi
            gamma1 = -gamma
            theta = theta1
            phi = phi1
            lam = lam1
            gamma = gamma1

        ureal = np.array(
            [
                [1, 0, 0, 0],
                [
                    0,
                    math.cos(gamma) * math.cos(theta / 2),
                    0,
                    -math.cos(gamma + lam) * math.sin(theta / 2),
                ],
                [0, 0, 1, 0],
                [
                    0,
                    math.cos(gamma + phi) * math.sin(theta / 2),
                    0,
                    math.cos(gamma + phi + lam) * math.cos(theta / 2),
                ],
            ]
        )
        uimag = np.array(
            [
                [0, 0, 0, 0],
                [
                    0,
                    math.sin(gamma) * math.cos(theta / 2),
                    0,
                    -math.sin(gamma + lam) * math.sin(theta / 2),
                ],
                [0, 0, 0, 0],
                [
                    0,
                    math.sin(gamma + phi) * math.sin(theta / 2),
                    0,
                    math.sin(gamma + phi + lam) * math.cos(theta / 2),
                ],
            ]
        )

        self.sim.cu(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def cr(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        theta = cmd.rotation[0]
        if cmd.inverse:
            theta = -theta

        ureal = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, math.cos(theta)]]
        )
        uimag = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, math.sin(theta)]]
        )

        self.sim.cr(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def iswap(self, cmd):
        if len(cmd.rotation) != 1 or len(cmd.targets) != 2:
            return

        theta = cmd.rotation[0]
        if cmd.inverse:
            theta = -theta

        ureal = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(theta), 0, 0],
                [0, 0, math.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        uimag = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, -math.sin(theta), 0],
                [0, -math.sin(theta), 0, 0],
                [0, 0, 0, 0],
            ]
        )

        self.sim.iswap(cmd.targets[0], cmd.targets[1], ureal, uimag)

    def barrier(self, cmd):
        return

    def sprtx(self, cmd):
        if len(cmd.targets) != 1:
            return

        ureal = []
        uimag = []
        if cmd.inverse:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[-0.5, 0.5], [0.5, -0.5]])
        else:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[0.5, -0.5], [-0.5, 0.5]])

        self.sim.sqrtx(cmd.targets[0], ureal, uimag)

    def sqrtxdg(self, cmd):
        if len(cmd.targets) != 1:
            return

        ureal = []
        uimag = []
        if cmd.inverse:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[0.5, -0.5], [-0.5, 0.5]])
        else:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[-0.5, 0.5], [0.5, -0.5]])

        self.sim.sqrtxdg(cmd.targets[0], ureal, uimag)

    def csqrtx(self, cmd):
        if len(cmd.targets) != 1 or len(cmd.controls) != 1:
            return

        ureal = []
        uimag = []
        if cmd.inverse:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[-0.5, 0.5], [0.5, -0.5]])
        else:
            ureal = np.array([[0.5, 0.5], [0.5, 0.5]])
            uimag = np.array([[0.5, -0.5], [-0.5, 0.5]])

        self.sim.csqrtx(cmd.controls[0], cmd.targets[0], ureal, uimag)

    def cswap(self, cmd):
        if len(cmd.targets) != 2 or len(cmd.controls) != 1:
            return

        ureal = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        uimag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.sim.cswap(cmd.controls[0], cmd.targets[0], cmd.targets[1], ureal, uimag)

    def id(self, cmd):
        # do nothing
        pass

    def amp(self, cmd):
        """the set amplitudes gate.

        Args:
           cmd: the Command object.
        """

        self.sim.amp(cmd.cmdex.amp.reals, cmd.cmdex.amp.imags, cmd.cmdex.amp.startind, cmd.cmdex.amp.numamps)
