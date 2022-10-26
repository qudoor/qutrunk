"""Definition of the quantum amplitude amplification Operator."""

from .operator import Operator, OperatorContext
from qutrunk.circuit import Qureg
from qutrunk.circuit.gates import X, MCZ, All, H


class QAA(Operator):
    """Quantum Amplitude Amplification Operator.

    Args:
        iterations: The number of QAA iteration.
        marked_index: Index of the marked qubit.

    Example:
        .. code-block:: python

            from qutrunk.circuit.ops import QAA
            from qutrunk.circuit import QCircuit
            from qutrunk.circuit.gates import H, All

            circuit = QCircuit()
            qureg = circuit.allocate(4)
            All(H) * qureg
            QAA(3, 7) * qureg
            for i in range(2 ** len(qureg)):
                print(circuit.get_prob(i))

        First, the four qubits are uniformly superposed,
        and then the state value of 7 is selected as the marker value.
        Three times of QAA iterative calculation are performed.
        The result obtained after the operation is that the probability of
        the corresponding state of 7 exceeds 96%.
    """

    def __init__(self, iterations, marked_index):
        super().__init__()
        self.iterations = iterations
        self.marked_index = marked_index

    def __mul__(self, qureg: Qureg):
        """Apply the QAA operator."""
        if not isinstance(qureg, Qureg):
            raise TypeError("The operand must be Qureg.")

        if self.marked_index < 0 or self.marked_index >= 2 ** len(qureg):
            raise ValueError("The marked index value exceed 2 ** len(qureg).")

        with OperatorContext(qureg.circuit):
            for i in range(self.iterations):
                self._flip_process(qureg)
                self._imag_process(qureg)
                # show intermediate process
                prob_amp = qureg.circuit.get_prob(self.marked_index)
                print(f"prob of state |{self.marked_index}> = {prob_amp}")

        qureg.circuit.append_statement(
            f"QAA({self.iterations}, {self.marked_index}) * q"
        )

    def _flip_process(self, qureg):
        """Flit the phase of target qubit.

        Args:
            qureg: The qureg apply flip process.
        """
        # 0b111
        bit_flip = bin(self.marked_index)
        # 00000111
        bit_flip = bit_flip[2:].zfill(len(qureg))
        # 11100000
        bit_flip = bit_flip[::-1]

        for i in range(len(bit_flip)):
            if bit_flip[i] == "0":
                X * qureg[i]

        MCZ(len(qureg) - 1) * qureg

        for i in range(len(bit_flip)):
            if bit_flip[i] == "0":
                X * qureg[i]

    def _imag_process(self, qureg):
        """Apply image process.

        Args:
            qureg: The qureg apply imag process.
        """
        All(H) * qureg
        All(X) * qureg
        MCZ(len(qureg) - 1) * qureg
        All(X) * qureg
        All(H) * qureg

