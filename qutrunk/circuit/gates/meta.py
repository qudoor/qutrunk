"""Definition of some meta operator."""

from typing import Union

from .basicgate import BasicGate
from qutrunk.circuit import QuBit, Qureg


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


class gate(BasicGate):
    """Definition of custom gate.

    Implement by composing some basic logic gates.

    Example:
        .. code-block:: python

            @gate
            def my_gate(q):
                H * q[0]
                CNOT * (q[0], q[1])

            circuit = QCircuit()
            q = circuit.allocate(2)
            my_gate * q
            All(Measure) * q
            res = circuit.run(shots=100)
            print(res.get_counts())
    """

    def __init__(self, func):
        self.compose_gate = True
        self.callable = func

    def __or__(self, qubits: Union[QuBit, Qureg, tuple]):
        """Quantum logic gate operation."""
        if not isinstance(qubits, (QuBit, Qureg, tuple)):
            raise TypeError("qubits should be type of QuBit, Qureg or tuple")

        self.callable(qubits)

    def __mul__(self, qubits: Union[QuBit, Qureg, tuple]):
        self.__or__(qubits)


def Inv(gate):
    """Inverse gate.

    Args:
        gate: The gate will apply inverse operator.

    Example:
        .. code-block:: python

            Inv(H) * q[0]
    """
    if isinstance(gate, BasicGate):
        gate.is_inverse = True

    return gate
