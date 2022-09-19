"""Quantum bit and Classical bit objects."""


class Bit:
    """A generic bit.

    Args:
        register: Bit's register.
        index: Bit's index.
    """

    def __init__(self, register=None, index=None):
        """Create a new generic bit."""
        if register is None and index is None:
            self.register = None
            self.index = None
        else:
            try:
                index = int(index)
            except Exception:
                raise TypeError(
                    f"index needs to be an int: type {type(index).__name__} was provided."
                )

            self.register = register
            self.index = index
            self._hash = hash((self.register, self.index))
            self._repr = f"{self.__class__.__name__}({self.register}, {self.index})"

    def __repr__(self):
        """Return the string representing the bit."""
        if self.register is None and self.index is None:
            return object.__repr__(self)

        return self._repr

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self.register is None and self.index is None:
            return other is self

        try:
            return self._repr == other._repr
        except AttributeError:
            return False
