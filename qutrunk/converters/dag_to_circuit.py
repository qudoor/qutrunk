"""Helper function for converting a dag to a circuit."""

from qutrunk.circuit import QCircuit
from qutrunk.dagcircuit.dagnode import DAGOpNode


def dag_to_circuit(dag):
    """Build a ``QCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): The input dag.

    Return:
        QCircuit: The circuit representing the input dag.
    """

    circuit = QCircuit()
    if dag.name:
        circuit.name = dag.name
    circuit.allocate(len(dag.qubits))
    circuit.metadata = dag.metadata

    for node in dag.topological_op_nodes():
        if hasattr(node.op, "definition"):
            unroll_custom_gate(circuit, node.op.definition, node.qargs, node.cargs)
        else:
            append_node(circuit, node)

    return circuit


def append_node(circuit: QCircuit, node: DAGOpNode):
    """
    Interpret node and apply to circuit
    support both single qubit and multi qubit gate
    as measure gate and all gate only take qargs, cargs is omitted

    Args:
        circuit: Generated circuit.
        node: Dag node tobe interpreted.
    """
    gate = node.op
    qubit = node.qargs
    for qb in qubit:
        qb.circuit = circuit
    if len(qubit) == 1:
        qubit = qubit[0]
    gate * qubit


def unroll_custom_gate(circuit, nodes, qargs, cargs):
    """Unroll custom gate in QASM file, nested custom gate is supported.

    Args:
        circuit (QCircuit): Circuit to append gate.
        nodes: Operation node from dag.
        qargs: Qargs for the gate.
        cargs: Cargs for the gate.
    """
    for inst in nodes:
        op = inst[0]
        inner_qargs = inst[1]
        inner_cargs = inst[2]

        real_qargs = []
        for qb in inner_qargs:
            real_qargs.append(qargs[qb.index])

        real_cargs = []
        for cb in inner_cargs:
            real_cargs.append(cargs[cb.index])

        if hasattr(op, "definition"):
            unroll_custom_gate(circuit, op.definition, real_qargs, real_cargs)
        else:
            append_node(circuit, DAGOpNode(op, real_qargs, real_cargs))
