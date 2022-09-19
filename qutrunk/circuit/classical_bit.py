"""Classical Bit reference object."""
from .bit import Bit


class CBit(Bit):
    """A classical bit.

    Args:
        creg(CReg): A classical register containing the bit.
        index (int): The index of the bit in its containing register.
        value: The value of qubit which is measured.

    Raises:
        TypeError: If the provided register is not a CReg object.
    """

    def __init__(self, creg=None, index=None, value=None):
        """Creates a classical bit."""
        from .classical_reg import CReg

        if not isinstance(creg, CReg):
            raise TypeError(
                f"CBit needs a CReg and {type(creg).__name__} eas provided."
            )

        super().__init__(creg, index)
        self._value = value

        if creg is not None:
            self.circuit = creg.circuit

    @property
    def value(self):
        """Get cbit value."""
        return self._value

    @value.setter
    def value(self, value):
        """Update cbit value when the corresponding qubit measure done."""
        self._value = value
