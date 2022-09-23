"""
DAG Node.
"""


class DAGNode:
    """Parent class for DAGOpNode, DAGInNode, and DAGOutNode."""

    def __init__(self, wire=None, node_id=-1):
        self._wire = wire
        self._node_id = node_id

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __str__(self):
        return str(id(self))


class DAGOpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""

    def __init__(self, op, qargs=None, cargs=None):
        super().__init__()
        self._type = "op"
        self.op = op
        self.qargs = qargs
        self.cargs = cargs
        self.sort_key = str(self.qargs)

    @property
    def name(self):
        """Operation name."""
        return str(self.op)

    def __repr__(self):
        return f"DAGOpNode(op={self.op}, qargs={self.qargs}, cargs={self.cargs})"

    __str__ = __repr__


class DAGInNode(DAGNode):
    """Object to represent an incoming wire node in the DAGCircuit."""

    def __init__(self, wire):
        super().__init__()
        self._type = "in"
        self.wire = wire
        self.sort_key = str([])

    def __repr__(self):
        return f"DAGInNode(wire={self.wire})"

    __str__ = __repr__


class DAGOutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""

    def __init__(self, wire):
        super().__init__()
        self._type = "out"
        self.wire = wire
        self.sort_key = str([])

    def __repr__(self):
        return f"DAGOutNode(wire={self.wire})"

    __str__ = __repr__
