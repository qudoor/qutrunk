"""Quantum circuit convert to DAG."""
from qutrunk.dagcircuit.dagcircuit import DAGCircuit
from qutrunk.circuit import QuBit, CBit


def circuit_to_dag(circuit):
    """Convert circuit to DAGCircuit.

     Args:
        circuit: The input circuit.

    Return:
        DAGCircuit: The DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_cbits(circuit.cbits)

    for cmd in circuit.cmds:
        qargs, cargs = __operation_command(cmd, circuit)
        dagcircuit.apply_operation_back(cmd, qargs, cargs)

    return dagcircuit


def __operation_command(command, circuit):
    """Rebuild instruction."""
    qargs = []
    cargs = []

    for c in command.controls:
        qubit = QuBit(circuit.qreg, index=c)
        qargs.append(qubit)

    for c in command.targets:
        qubit = QuBit(circuit.qreg, index=c)
        qargs.append(qubit)

    if str(command.gate) == "Measure":
        cbit = CBit(circuit.creg, index=command.targets[0])
        cargs.append(cbit)

    return qargs, cargs
