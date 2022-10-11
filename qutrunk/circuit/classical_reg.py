"""Classical register reference object."""

import itertools

from .classical_bit import CBit


class CReg:
    """A classical register.

    Args:
        name (str): The name of the register.
        size: The size of the register.
        circuit: A quantum circuit.
    """

    prefix = "c"
    bit_type = "CBit"
    instances_counter = itertools.count()

    def __init__(self, circuit=None, name=None, size=None):
        if name is None:
            name = f"{self.prefix}{self.instances_counter}"
        self.name = name

        self.cbits = []
        # TODO: delete size ?
        self.size = size
        self.circuit = circuit

        for index in range(self.size):
            self.cbits.append(CBit(self, index))

    def append(self, cbit):
        """Add a CBit instance.

        Args:
            cbit: A list which store CBit instance.
        """
        self.cbits.append(cbit)

    def qasm(self):
        """Return OPENQASM string for this register."""
        return f"creg {self.name}[{self.size}];"

    def __getitem__(self, idx):
        """Return a Cbit instance.

        Args:
            idx: The index of Cbit.

        Returns:
            CBit instance.
        """
        if not isinstance(idx, int):
            raise ValueError("expected integer index into register")
        return self.cbits[idx]

    def __len__(self):
        return len(self.cbits)

    def index(self, cbit):
        """Find the index of the provided cbit within this register.

        Args:
            cbit: A list which store CBit instance.
        """
        return self.cbits.index(cbit)

    def __setitem__(self, key, value):
        self.cbits[key] = value

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.size})"
