"""Definition of some meta operator."""
from typing import Union, Optional
import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import QuBit, Qureg
from qutrunk.circuit.command import Command, CmdEx, Mat


# need to improve.
class All(BasicGate):
    """Meta operator, provides unified operation of multiple qubits.

    Args:
        gate: The gate will apply to all qubits.

    Example:
        .. code-block:: python

            All(H) * qureg
            All(Measure) * qureg
    """

    def __init__(self, gate):
        self.gate = gate

    def __or__(self, qureg):
        """Quantum logic gate operation.

        Args:
            qureg: The qureg(represent a set of qubit) to apply gate.

        Example:
            .. code-block:: python

                All(H) * qureg
                All(Measure) * qureg
        """
        for q in qureg:
            self.gate * q

    def __mul__(self, qureg):
        """Overwrite * operator to achieve quantum logic gate operation, \
            reuse __or__ operator implement."""
        self.__or__(qureg)


class Power(BasicGate):
    """Power Gate.

    Args:
        power: The power to raise target gate to.
        gate: The target gate to raise.

    Example:
        .. code-block:: python

            Power(2, gate) * q[0]
    """

    def __init__(self, power, gate):
        self.power = power
        self.gate = gate

    def __or__(self, qubits: Union[QuBit, Qureg, tuple]):
        """Quantum logic gate operation."""
        if self.power < 0:
            raise ValueError("power should >= 0.")

        if not isinstance(qubits, (QuBit, Qureg, tuple)):
            raise TypeError("qubits should be type of QuBit, Qureg or tuple.")

        for _ in range(self.power):
            self.gate * qubits

    def __mul__(self, qubits: Union[QuBit, Qureg, tuple]):
        self.__or__(qubits)


class Matrix(BasicGate):
    """Custom matrix gate.

    Example:
            .. code-block:: python

                Matrix([[0.5, 0.5], [0.5, -0.5]]) * qr[0]  -- No controlled bit
                Matrix([[0.5, 0.5], [0.5, -0.5]], 1) * (qr[0], qr[1])  -- qr[0] is controlled bit
                Matrix([[0.5, 0.5], [0.5, -0.5]], 2) * (qr[0], qr[1], qr[2])  -- qr[0], qr[1] are controlled bits
    """

    def __init__(self, matrix, ctrl_cnt=0):
        super().__init__()
        self.matrix = matrix
        self.ctrl_cnt = ctrl_cnt

    def __str__(self):
        return "Matrix"

    def __or__(self, qubits):
        """Quantum logic gate operation.

        Args:
            qubit: The quantum bit to apply X gate.

        Example:
            .. code-block:: python

                Matrix([[0.5, 0.5], [0.5, -0.5]]) * qr[0]  -- No controlled bit
                Matrix([[0.5, 0.5], [0.5, -0.5]], 1) * (qr[0], qr[1])  -- qr[0] is controlled bit
                Matrix([[0.5, 0.5], [0.5, -0.5]], 2) * (qr[0], qr[1], qr[2])  -- qr[0], qr[1] are controlled bits

        Raises:
            NotImplementedError: If the argument is not a Qubit object.
        """
        if not isinstance(qubits, QuBit) and not all(
            isinstance(qubit, QuBit) for qubit in qubits
        ):
            raise TypeError("The argument must be Qubit object.")

        if (isinstance(qubits, QuBit) and self.ctrl_cnt > 0) or (
            not isinstance(qubits, QuBit) and (len(qubits) <= self.ctrl_cnt)
        ):
            raise ValueError("The parameter miss controlled or target qubit(s).")

        controls = (
            None if self.ctrl_cnt <= 0 else [q.index for q in qubits[0 : self.ctrl_cnt]]
        )
        targets = (
            [qubits.index]
            if isinstance(qubits, QuBit)
            else [q.index for q in qubits[self.ctrl_cnt :]]
        )

        if not self.check_matrix_format(len(targets)):
            raise ValueError(
                "The matrix is not in the right format by specified target(s)."
            )

        e = np.matrix(self.matrix)
        if self.is_inverse:
            if not self.is_unitary(e):
                raise ValueError(
                    "Only unitary matrices support invertible operations"
                )
            else:
                e = e.T.conjugate()

        cmd = Command(self, targets, controls, cmdex=CmdEx(mat=Mat()))
        cmd.cmdex.mat.reals = np.real(e).tolist()
        cmd.cmdex.mat.imags = np.imag(e).tolist()
        cmd.cmdex.mat.unitary = self.is_unitary(e)

        self.commit(qubits.circuit, cmd) if isinstance(qubits, QuBit) else self.commit(
            qubits[0].circuit, cmd
        )

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)

    def inv(self):
        """Apply inverse gate."""
        gate = Matrix(self.matrix, self.ctrl_cnt)
        gate.is_inverse = not self.is_inverse
        return gate

    def ctrl(self, ctrl_cnt=1):
        """Apply controlled gate.

        Args:
            ctrl_cnt: The number of control qubits, default: 1.
        """
        gate = Matrix(self.matrix, ctrl_cnt)
        gate.is_inverse = self.is_inverse
        return gate

    def check_matrix_format(self, numtargets):
        len_targets = 2**numtargets
        len_matrix = len(self.matrix)
        if len_targets != len_matrix:
            return False

        for row in range(len(self.matrix)):
            len_row = len(self.matrix[row])
            if len_targets != len_row:
                return False

        return True

    def is_unitary(self, mat):
        """Test a matrix is unitary or not.

         Example:
            .. code-block:: python

                m = [[1, 0], [0, 1]]
                m = np.matrix(m)
                print(is_unitary(m))
        """
        return np.allclose(np.eye(mat.shape[0]), mat.H * mat)


class Gate(BasicGate):
    """Definition of custom gate.

    Implement by composing some basic logic gates or define specific matrix.

    Example:
        .. code-block:: python

            @Gate
            def my_gate(a, b, c, d):
                return Gate() << (Matrix([[-0.5, 0.5], [0.5, 0.5]], 2).inv(), (a, b, c)) \
                    << (Matrix([[0.5, -0.5], [0.5, 0.5]]).ctrl().inv(), (a, c)) \
                    << (Matrix([[0.5, 0.5], [-0.5, 0.5]]), b)

            my_gate * (q[3], q[1], q[0], q[2])
    """

    def __init__(self, func: Optional[callable] = None):
        super().__init__()
        self.gates = []
        self.func = func

    def __lshift__(self, gate_define):
        if not isinstance(gate_define[0], BasicGate):
            raise TypeError("The first parameter is not a gate object.")

        if not isinstance(gate_define[1], QuBit) and not all(
            isinstance(qubit, QuBit) for qubit in gate_define[1]
        ):
            raise TypeError("The argument must be Qubit object.")

        self.gates.append({"gate": gate_define[0], "qubits": gate_define[1]})

        return self

    def __or__(self, qubits: Union[QuBit, tuple]):
        """Quantum logic gate operation."""
        if not isinstance(qubits, QuBit) and not all(
            isinstance(qubit, QuBit) for qubit in qubits
        ):
            raise TypeError("The argument must be Qubit object.")

        if isinstance(qubits, QuBit):
            custom_gate = self.func(qubits)
        else:
            custom_gate = self.func(*qubits)

        for c in custom_gate.gates:
            c["gate"] * c["qubits"]

    def __mul__(self, qubits: Union[QuBit, tuple]):
        self.__or__(qubits)


# note: 该方法会导致部分门操作产生状态污染，比如通过对象实例调用的门操作
# 只要设置过状态，那么后续所有该量子门操作都带了这个状态
# def Inv(gate):
#     """Inverse gate.

#     Args:
#         gate: The gate will apply inverse operator.

#     Example:
#         .. code-block:: python

#             Inv(H) * q[0]
#     """
#     if isinstance(gate, BasicGate):
#         gate.is_inverse = not gate.is_inverse

#     return gate
