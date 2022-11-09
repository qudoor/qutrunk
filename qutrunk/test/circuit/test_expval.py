from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Ry, PauliCoeff, PauliType, PauliCoeffs
from qutrunk.test.tools.float import equal


def test_expval_pauli_sum():
    circuit = QCircuit()
    q = circuit.allocate(2)

    H * q[0]
    Ry(1.23) * q[1]
    pauli_coeffs = (
        PauliCoeffs()
        << PauliCoeff(0.12, [PauliType.PAULI_Z])
        << PauliCoeff(0.34, [PauliType.PAULI_X, PauliType.PAULI_I])
    )
    expect_sum = circuit.expval_pauli_sum(pauli_coeffs)

    # 确保前后的计算结果是一致的, 对于上述例子后端计算的结果为0.34
    assert equal(expect_sum, 0.34)
