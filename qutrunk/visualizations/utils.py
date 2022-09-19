from qutrunk.converters.circuit_to_dag import circuit_to_dag


def _get_instructions(circuit):
    """Given a circuit, return a tuple(qubits, cbits, nodes).

    Args:
        circuit: A quantum circuit.

    Returns:
        qubits: A list of QuBit.
        cbits: A list of CBit.
        nodes: Nodes is a list of DAG nodes whose type is "operation".
            (qubits, cbits, nodes)
    """

    dag = circuit_to_dag(circuit)

    qubits = dag.qubits
    cbits = dag.cbits

    nodes = []
    for node in dag.topological_op_nodes():
        nodes.append([node])
    nodes = [
        [node for node in layer if any(q in qubits for q in node.qargs)]
        for layer in nodes
    ]

    return qubits, cbits, nodes
