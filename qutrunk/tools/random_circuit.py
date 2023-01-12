"""random circuit."""
import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import (
    HGate,
    XGate,
    P,
    R,
    Rx,
    Ry,
    Rz,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    YGate,
    ZGate,
    X1Gate,
    Y1Gate,
    Z1Gate,
    SqrtXGate,
    IGate,
    U1,
    U2,
    U3,
    SwapGate,
    iSwapGate,
    CP,
    CR,
    CRx,
    CRy,
    CRz,
    Rxx,
    Ryy,
    Rzz,
    CU1,
    CU3,
    CHGate,
    CSqrtXGate,
    CSwapGate,
    Measure,
    All,
)


def random_circuit(num_qubits, depth, backend=None, measure=False, seed=None):
    """Generate random circuit.

     Args:
        num_qubits (int): number of qubit.
        depth (int): quantum depth.
        backend: Used to run quantum circuits.
        measure (bool): if True, measure all qubits at the end.
        seed (int): random seed.
    """
    # 1 generate random seed
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # 2 classify gates by qubits and parameters.
    one_qubit_gate = [
        HGate,
        XGate,
        P,
        R,
        Rx,
        Ry,
        Rz,
        SGate,
        SdgGate,
        TGate,
        TdgGate,
        YGate,
        ZGate,
        X1Gate,
        Y1Gate,
        Z1Gate,
        SqrtXGate,
        IGate,
        U1,
        U2,
        U3,
    ]
    two_qubit_gate = [
        SwapGate,
        iSwapGate,
        CP,
        CR,
        CRx,
        CRy,
        CRz,
        Rxx,
        Ryy,
        Rzz,
        CU1,
        CU3,
        CHGate,
        CSqrtXGate,
    ]
    three_qubit_gate = [CSwapGate]

    one_param = [
        P,
        Rx,
        Ry,
        Rz,
        CP,
        CR,
        CRx,
        CRy,
        CRz,
        Rxx,
        Ryy,
        Rzz,
        U1,
        CU1,
    ]
    two_param = [
        R,
        U2,
    ]
    three_param = [
        U3,
        CU3,
    ]

    # 3 generate random circuit
    # create quantum circuit
    qc = QCircuit(backend=backend)
    # allocate quantum qubits
    qr = qc.allocate(num_qubits)

    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), 3)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                gate = rng.choice(one_qubit_gate)
            elif num_operands == 2:
                gate = rng.choice(two_qubit_gate)
            elif num_operands == 3:
                gate = rng.choice(three_qubit_gate)
            if gate in one_param:
                num_angles = 1
            elif gate in two_param:
                num_angles = 2
            elif gate in three_param:
                num_angles = 3
            else:
                num_angles = 0
            # parameters
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            # qubit
            reg_gate = [qr[i] for i in operands]
            # gate
            g = gate(*angles)

            if len(operands) == 1:
                g * reg_gate.pop()
            elif len(operands) == 2:
                g * (reg_gate[0], reg_gate[1])
            elif len(operands) == 3:
                g * (reg_gate[0], reg_gate[1], reg_gate[2])

    if measure:
        All(Measure) * qr

    # 4 return circuit
    return qc


if __name__ == "__main__":
    circuit = random_circuit(4, 3, measure=True)
    res = circuit.run(shots=100)
    print(res.get_counts())
    circuit.draw()
