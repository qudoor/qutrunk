import itertools

from .qubit import QuBit


class SubQureg:
    """SubQureg."""

    def __init__(self, circuit):
        self.circuit = circuit
        self.qubits = []

    def __getitem__(self, idx):
        """Return a Qubit instance."""
        if not isinstance(idx, int):
            raise ValueError("expected integer index into register")
        return self.qubits[idx]

    def append(self, qubit: QuBit):
        """Append QuBit to SubQureg.

        Args:
            qubit: The target qubit append to SubQureg.
        """
        self.qubits.append(qubit)

    def get_indexs(self) -> set:
        """Get qubits indexs of SubQureg"""
        start = self.qubits[0].index
        end = self.qubits[-1].index
        res = set()
        for i in range(start, end + 1):
            res.add(i)

        return res

    def __len__(self):
        return len(self.qubits)


class Qureg:
    """Register, maintains a set of qubits.

    Args:
        circuit: A quantum circuit.
        name: The name of the register.
        size: The size of the register.
    """

    prefix = "q"
    bit_type = "QuBit"
    instances_counter = itertools.count()

    def __init__(self, circuit=None, name=None, size=None):
        if name is None:
            name = f"{self.prefix}{self.instances_counter}"
        self.name = name

        self.qubits = []
        # TODO: delete size ?
        self.size = size
        self.circuit = circuit

        for index in range(self.size):
            self.qubits.append(QuBit(self, index))

    def __getitem__(self, idx):
        """Return a Qubit instance.

        Arg:
            idx: The index of Qubit.

        Returns:
            Qubit instance.

        Raises:
          ValueError: If the index of Qubit is not a integer data.
        """
        if not isinstance(idx, int):
            raise ValueError("expected integer index into register")
        return self.qubits[idx]

    def __len__(self):
        return len(self.qubits)

    def qasm(self):
        """Return OPENQASM string for this register."""
        return f"qreg q[{len(self)}];"

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.size})"

    def index(self, qubit):
        """Find the index of the provided qubit within this register.

        Args:
            qubit: The qubit to get index.
        """
        return self.qubits.index(qubit)

    def get_indexs(self):
        """Get qubits indexs of Qureg."""
        res = set()
        for i in range(len(self.qubits)):
            res.add(i)
        return res

    def split(self, sections: list):
        """Split Qureg into subqureg.

        Args:
            sections: Tell how to split the original Qureg, \
                each value item represents the size of corresponding subqureg.

        Returns:
            The tuple contains all subqureg.
        """
        if not isinstance(sections, list):
            raise TypeError("sections parameter should be a list.")

        if sum(sections) != len(self.qubits):
            raise ValueError("The sum of sections should be equal to len(qubits).")

        res = []
        start_index = 0
        for sec in sections:
            slices = SubQureg(self.circuit)
            for i in range(sec):
                # note: share the same QuBit
                slices.append(self.qubits[start_index + i])
            res.append(slices)
            start_index += sec

        return tuple(res)
