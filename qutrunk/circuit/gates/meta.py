"""Definition of some meta operator."""
from typing import Union
import numpy as np

from .basicgate import BasicGate
from qutrunk.circuit import QuBit, Qureg
from qutrunk.circuit.command import Command, CmdEx, Mat

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
            raise ValueError("power should >= 0")

        if not isinstance(qubits, (QuBit, Qureg, tuple)):
            raise TypeError("qubits should be type of QuBit, Qureg or tuple.")

        for _ in range(self.power):
            self.gate * qubits

    def __mul__(self, qubits: Union[QuBit, Qureg, tuple]):
        self.__or__(qubits)


class Matrix(BasicGate):
    """Custom matrix gate."""
    def __init__(self, matrix, ctrl_cnt = 0):
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
        if not isinstance(qubits, QuBit) and not all(isinstance(qubit, QuBit) for qubit in qubits):
            raise NotImplementedError("The argument must be Qubit object.")

        if (isinstance(qubits, QuBit) and self.ctrl_cnt > 0) or (not isinstance(qubits, QuBit) and (len(qubits) <= self.ctrl_cnt)):
            raise AttributeError("The parameter miss controlled or target qubit(s).")

        controls = None if self.ctrl_cnt <= 0 else [q.index for q in qubits[0 : self.ctrl_cnt]]
        targets = [qubits.index] if isinstance(qubits, QuBit) else [q.index for q in qubits[self.ctrl_cnt:]]
        
        cmd = Command(self, targets, controls, cmdex=CmdEx(mat=Mat()))

        e = self.matrix
        if self.is_inverse:
            e = np.matrix(self.matrix)
            e = e.T.conjugate()

        cmd.cmdex.mat.reals = np.real(e)
        cmd.cmdex.mat.imags = np.imag(e)

        self.commit(qubits.circuit, cmd) if isinstance(qubits, QuBit) else self.commit(qubits[0].circuit, cmd)

    def __mul__(self, qubit):
        """Overwrite * operator to achieve quantum logic gate operation, reuse __or__ operator implement."""
        self.__or__(qubit)
    
    def inv(self):
        """Apply inverse gate"""
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


# class Gate:
#     """Used to define custom gate.

#     Example:
#         .. code-block:: python

#             from qutrunk.circuit import QCircuit
#             from qutrunk.circuit.gates import H, CNOT, CustomGate, All, Measure, gate

#             circuit = QCircuit()
#             q = circuit.allocate(2)

#             @def_gate
#             def my_gate(a, b):
#                 return Gate() << (H, a) << (CNOT, (a, b))

#             my_gate * (q[0], q[1])
#             All(Measure) * q
#             circuit.print()
#             res = circuit.run(shots=100)
#             print(res.get_counts()) 
            
#     """

#     def __init__(self):
#         super().__init__()
#         self.gates = []

#     def append_gate(self, gate, qubits):
#         """Append basic gate to custom gate.
        
#         Args:
#             gate: Basic gate.
#             qubits: The target qubits of quantum gate to apply.
#         """
#         self.gates.append({"gate": gate, "qubits": qubits})

#     def __lshift__(self, gate_define):
#         self.append_gate(gate_define[0], gate_define[1])
#         return self


class def_gate(BasicGate):
    """Definition of custom gate.

    Implement by composing some basic logic gates or define specific matrix.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, CNOT, CustomGate, All, Measure, gate

            circuit = QCircuit()
            q = circuit.allocate(2)

            @def_gate
            def my_gate(a, b):
                return Gate() << (H, a) << (CNOT, (a, b))

            my_gate * (q[0], q[1])
            All(Measure) * q
            circuit.print()
            res = circuit.run(shots=100)
            print(res.get_counts()) 
    """

    def __init__(self):
        super().__init__()
        #self.ctrl_cnt = 0
        #self.callable = func
        #self.gates = []

    # def _get_operand(self, ctrl_qubits, target_qubits)->tuple:
    #     if len(ctrl_qubits) == 0:
    #         return target_qubits
            
    #     if isinstance(target_qubits, QuBit):
    #         return (*ctrl_qubits, target_qubits)

    #     return (*ctrl_qubits, *target_qubits)

    # def __or__(self, qubits: Union[QuBit, tuple]):
    #     """Quantum logic gate operation."""
    #     if not isinstance(qubits, (QuBit, tuple)):
    #         raise TypeError("qubits should be type of QuBit or tuple")

    #     ctrl_qubits = []
    #     target_qubits = []
    #     if self.ctrl_cnt > 0:
    #         ctrl_qubits = qubits[0:self.ctrl_cnt]
    #         target_qubits = qubits[self.ctrl_cnt:]
    #         custom_gate = self.callable(*target_qubits)
    #     else:
    #         if isinstance(qubits, QuBit):
    #             custom_gate = self.callable(qubits)
    #         else:
    #             custom_gate = self.callable(*qubits)

    #     for c in custom_gate.gates:
    #         operand = self._get_operand(ctrl_qubits, c['qubits'])
    #         if self.is_inverse and self.ctrl_cnt > 0:
    #             c['gate'].ctrl(self.ctrl_cnt).inv() * operand
    #         elif self.is_inverse:
    #             c['gate'].inv() * operand
    #         elif self.ctrl_cnt > 0:
    #             c['gate'].ctrl(self.ctrl_cnt) * operand
    #         else:
    #             c['gate'] * operand

    # def __mul__(self, qubits: Union[QuBit, tuple]):
    #     self.__or__(qubits)

    # def append_gate(self, gate, qubits):
    #     """Append basic gate to custom gate.
        
    #     Args:
    #         gate: Basic gate.
    #         qubits: The target qubits of quantum gate to apply.
    #     """
    #     self.gates.append({"gate": gate, "qubits": qubits})

    def __lshift__(self, gate_define):
        gate_define[0] * gate_define[1]
        return self

    # def inv(self):
    #     """Apply inverse gate.

    #     Note: Ensure all the basic gates used to compose custom gate has a inverse version, \
    #         otherwise, an exception will be raised.
    #     """
    #     gate = def_gate(func=self.callable)
    #     gate.is_inverse = not self.is_inverse
    #     gate.ctrl_cnt = self.ctrl_cnt
    #     return gate

    # def ctrl(self, ctrl_cnt=1):
    #     """Apply controlled gate.

    #     Note: Ensure all the basic gates used to compose custom gate has a control version, \
    #         otherwise, an exception will be raised.
        
    #     Args:
    #         ctrl_cnt: The number of control qubits, default: 1.
    #     """
    #     gate = def_gate(func=self.callable)
    #     gate.is_inverse = self.is_inverse
    #     gate.ctrl_cnt = ctrl_cnt
    #     return gate


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
