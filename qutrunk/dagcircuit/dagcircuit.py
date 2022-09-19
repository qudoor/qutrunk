"""DAG represent the Quantum circuit."""
import itertools
from collections import OrderedDict

import retworkx as rx

from qutrunk.circuit import CBit, CReg, QuBit, Qureg
from .dagnode import DAGInNode, DAGOpNode, DAGOutNode


class DAGCircuit:
    """Convert quantum circuit to a directed acyclic graph(DAG)."""

    def __init__(self):
        """Create an empty circuit."""
        self.name = None
        self.metadata = None
        self._wires = set()
        self.input_map = OrderedDict()
        self.output_map = OrderedDict()
        self._multi_graph = rx.PyDAG()
        self.qubits = []
        self.cbits = []
        self._op_names = {}
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

    def _add_wire(self, wire):
        """Add a qubit or cbit to the circuit.

        Args:
            wire: The object of qubit or cbit.

        Raises:
            ValueError.
        """

        if wire not in self._wires:
            self._wires.add(wire)

            input_node = DAGInNode(wire=wire)
            output_node = DAGOutNode(wire=wire)
            input_map_id, output_map_id = self._multi_graph.add_nodes_from(
                [input_node, output_node]
            )
            input_node._node_id = input_map_id
            output_node._node_id = output_map_id

            self.input_map[wire] = input_node
            self.output_map[wire] = output_node
            self._multi_graph.add_edge(input_node._node_id, output_node._node_id, wire)
        else:
            raise ValueError(f"duplicate wire {wire}")

    def add_qubits(self, qubits):
        """Add qubits to DAGCircuit, and add wires.

        Args:
            qubits: The qubits add to DAG.

        Raises:
            ValueError.
        """
        if any(not isinstance(qubit, QuBit) for qubit in qubits):
            raise ValueError("not a QuBit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise ValueError(f"duplicate qubits {duplicate_qubits}")

        self.qubits.extend(qubits)
        for qubit in qubits:
            self._add_wire(qubit)

    def add_cbits(self, cbits):
        """Add classical bits to DAGCircuit, and add wires.

        Args:
            cbits: The classical bits add to DAG.

        Raises:
            ValueError.
        """
        if any(not isinstance(cbit, CBit) for cbit in cbits):
            raise ValueError("not a CBit instance.")

        duplicate_cbits = set(self.cbits).intersection(cbits)
        if duplicate_cbits:
            raise ValueError(f"duplicate cbits {duplicate_cbits}")

        self.cbits.extend(cbits)
        for cbit in cbits:
            self._add_wire(cbit)

    def topological_nodes(self, key=None):
        """Return nodes in topological sort order.

        Args:
             key (Callable): A callable which will take a DAGNode object and
                 return a string sort key.

         Returns:
             Generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
        """

        def _key(x):
            return x.sort_key

        if key is None:
            key = _key

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=key))

    def topological_op_nodes(self, key=None):
        """Return op nodes in topological sort order.

        Args:
            key (Callable): A callable which will take a DAGNode object and
                return a string sort key.

        Returns:
            generator(DAGOpNode): Op node in topological order
        """
        return (nd for nd in self.topological_nodes(key) if isinstance(nd, DAGOpNode))

    @property
    def wires(self):
        """Return a list of the wires in order."""
        return self.qubits + self.cbits

    def _add_op_node(self, op, qargs, cargs):
        """Add operation node.

        Args:
            op: The operation associated with DAG node.
            qargs: A list of QuBit.
            cargs: A list of CBit.

        Returns:
            node_index: The integer node index for the new operation node on the DAG.
        """
        new_node = DAGOpNode(op=op, qargs=qargs, cargs=cargs)

        node_index = self._multi_graph.add_node(new_node)
        new_node._node_id = node_index
        self._increment_op(op)

        return node_index

    def apply_operation_back(self, op, qargs=None, cargs=None):
        """Add operation node at the end of DAG.

        Args:
            op (Commands): The operation associated with the DAG node.
            qargs (list[QuBit]): Qubits that op will be applied to.
            cargs (list[CBit]): Cbits that op will be applied to.

        Returns:
            DAGOpNode: The node for the op that was added to the dag.
        """
        if cargs is None:
            cargs = []

        if qargs is None:
            qargs = []

        node_index = self._add_op_node(op, qargs, cargs)

        all_bit = [qargs, cargs]
        self._multi_graph.insert_node_on_in_edges_multiple(
            node_index, [self.output_map[q]._node_id for q in itertools.chain(*all_bit)]
        )

        return self._multi_graph[node_index]

    def add_qreg(self, qreg):
        """Add all wires in a quantum register."""

        if not isinstance(qreg, Qureg):
            raise ValueError("not a Qureg instance.")

        if qreg.name in self.qregs:
            raise ValueError(f"duplicate register {qreg.name}")

        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for i in range(qreg.size):
            if qreg[i] not in existing_qubits:
                self.qubits.append(qreg[i])
                self._add_wire(qreg[i])

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, CReg):
            raise ValueError("not a CReg instance.")

        if creg.name in self.cregs:
            raise ValueError(f"duplicate register {creg.name}")

        self.cregs[creg.name] = creg
        existing_cbits = set(self.cbits)
        for i in range(creg.size):
            if creg[i] not in existing_cbits:
                self.cbits.append(creg[i])
                self._add_wire(creg[i])

    def _increment_op(self, op):
        """Count the number of each door operation."""
        if op.name in self._op_names:
            self._op_names[op.name] += 1
        else:
            self._op_names[op.name] = 1
