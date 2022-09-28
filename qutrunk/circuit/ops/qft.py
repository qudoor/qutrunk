from math import pi
from typing import Iterable, Union

from qutrunk.circuit import QuBit, Qureg
from qutrunk.circuit.gates import CP, H, Swap
from qutrunk.circuit.ops.operator import Operator, OperatorContext


class QFTOps(Operator):
    """Quantum Fourier Transfer Operator.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import All, Measure
            from qutrunk.circuit.ops import QFT

            circuit = QCircuit()
            qreg = circuit.allocate(5)

            # for full qft
            # QFT * qreg
            qubits = list(qreg)[::-2]
            QFT * qubits

            print(circuit.draw(line_length=1000))
            state = circuit.get_all_state()
            print(state)

            All(Measure) * qreg

            res = circuit.run(shots=1000)
            print(res.get_counts())

    """

    def __init__(self):
        super().__init__()

    def __mul__(self, qubits: Union[Qureg, Iterable[QuBit]]):
        """
        Args:
            qubits: Union[Qureg, Iterable[QuBit]], can be partial QFT.
        """
        if not all(isinstance(qb, QuBit) for qb in qubits):
            raise TypeError("The operand must be Qureg or Iterable of QuBit.")

        with OperatorContext(qubits[0].circuit) as oc:
            qb_cnt = len(qubits)
            for ctrl_qb_num in reversed(range(qb_cnt)):
                H * qubits[ctrl_qb_num]

                for target_qb_num in reversed(range(ctrl_qb_num)):
                    # Use negative exponents so that the angle safely underflows to zero, rather than
                    # using a temporary variable that overflows to infinity in the worst case.
                    lam = pi * (2.0 ** (target_qb_num - ctrl_qb_num))
                    CP(lam) * (qubits[ctrl_qb_num], qubits[target_qb_num])
            for i in range(qb_cnt // 2):
                Swap * (qubits[i], qubits[qb_cnt - i - 1])

        # TODO: need to improve
        operand = ""
        if isinstance(qubits, Qureg):
            operand = "q"
        else:
            operand += "("
            for q in qubits:
                operand += "q[" + str(q.index) + "], "
            operand = operand[0:-2]
            operand += ")"

        qubits[0].circuit.append_statement("QFT * " + operand)


class IQFTOps(Operator):
    def __init__(self):
        super().__init__()

    def __mul__(self, qubits: Union[Qureg, Iterable[QuBit]]):
        """
        Args:
            qubits: Union[Qureg, Iterable[QuBit]], can be partial QFT.
        """
        with OperatorContext(qubits[0].circuit) as oc:
            qb_cnt = len(qubits)
            for i in range(qb_cnt // 2):
                # swap is same as its inverse
                Swap * (qubits[i], qubits[qb_cnt - i - 1])

            for target_qb_num in range(qb_cnt):
                # H is same as its inverse
                H * qubits[target_qb_num]

                for ctrl_qb_num in range(target_qb_num + 1, qb_cnt):
                    lam = -pi * (2.0 ** (target_qb_num - ctrl_qb_num))
                    CP(lam) * (qubits[ctrl_qb_num], qubits[target_qb_num])

        # TODO: need to improve
        operand = ""
        if isinstance(qubits, Qureg):
            operand = "q"
        else:
            operand += "("
            for q in qubits:
                operand += "q[" + str(q.index) + "], "
            operand = operand[0:-2]
            operand += ")"

        qubits[0].circuit.append_statement("IQFT * " + operand)


QFT = QFTOps()
IQFT = IQFTOps()
