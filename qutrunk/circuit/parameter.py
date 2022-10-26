"""
Parameter Class for variable parameters.
"""
from uuid import uuid4


class Parameter:
    """Parameter Class for variable parameters.

    Args:
        name(str): the name of Parameter.
    """

    def __new__(cls, name):
        obj = object.__new__(cls)
        obj._uuid = uuid4()
        obj._hash = hash(obj._uuid)

        return obj

    def __init__(self, name):
        self.name = name
        self.value = None
        self._host = None

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self._uuid == other._uuid
        else:
            return False

    def __getstate__(self):
        return {"name": self.name, "value": self.value}

    def __setstate__(self, state):
        self.name = state["name"]
        self.value = state["value"]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    def host(self):
        """Get parameter host."""
        return self._host

    @host.setter
    def host(self, host):
        """Set parameter host.
        
        Args:
            host: The Host using this parameter. 
        """
        self._host = host

    def update(self, value):
        """Update parameter value.
        
        Args:
            value: Parameter value.
        """
        self.value = value
        if self._host:
            self._host.update_parameter(self)

