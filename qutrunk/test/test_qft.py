"""
verify different quantum fourier transform implementation,
include decomposition, backend direct, qiskit
"""
import math

import pytest
from qiskit import BasicAer, QuantumCircuit, execute
from qiskit.circuit.library import QFT as QIS_QFT

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, X
from qutrunk.circuit.ops import IQFT, QFT

SHOT = 30
TOLERANCE = 1e-10
MAX_QB_CNT = 8


@pytest.fixture(autouse=True)
def banner():
    print()
    print("=" * 50)
    yield
    print("\n" * 5)


@pytest.fixture(params=range(2, MAX_QB_CNT))
def qb_cnt(request):
    return request.param


@pytest.fixture(params=[False, True])
def inverse(request):
    return request.param


@pytest.fixture()
def circuit(qb_cnt):
    circuit = QCircuit()
    circuit.allocate(qb_cnt)
    return circuit


@pytest.fixture()
def circuit_extra(circuit, qb_cnt):
    qreg = circuit.qreg
    H * qreg[0]
    X * qreg[qb_cnt - 1]
    return circuit


@pytest.fixture()
def decomposition_qft_state(circuit, inverse):
    """
    decomposition QFT by H and CU1

    Args:
        circuit:
        inverse:

    Returns:

    """
    if inverse:
        IQFT * circuit.qreg
    else:
        QFT * circuit.qreg
    # qft.append(circuit, qubits, inverse=inverse)
    # circuit.backend.send_circuit(circuit)
    state = circuit.get_all_state()
    print(
        f"decomposition qft, qubits: {str(len(circuit.qubits))}, inverse: {str(inverse)}"
    )
    print(circuit.draw(line_length=1000))
    print(f"depth: {circuit.depth()}, width: {circuit.width()}")
    print(state)

    # All(Measure) * circuit.qreg

    # res = circuit.run(shots=SHOT)
    # print(res.get_counts())
    return state


@pytest.fixture()
def decomposition_qft_extra_state(circuit_extra, inverse):
    """
    decomposition QFT by H and CU1

    Args:
        circuit_extra:
        inverse:

    Returns:

    """
    if inverse:
        IQFT * circuit_extra.qreg
    else:
        QFT * circuit_extra.qreg
    # qft.append(circuit_extra, qubits, inverse=inverse)
    # circuit.backend.send_circuit(circuit)
    state = circuit_extra.get_all_state()
    print(
        f"decomposition qft extra, qubits: {str(len(circuit_extra.qubits))}, inverse: {str(inverse)}"
    )
    print(circuit_extra.draw(line_length=1000))
    print(state)

    # All(Measure) * circuit.qreg

    # res = circuit.run(shots=SHOT)
    # print(res.get_counts())
    return state


@pytest.fixture()
def qiskit_qft_state(qb_cnt, inverse):
    """
    qiskit qft don't support partial qft

    Returns:
        state:

    """
    simulator = BasicAer.get_backend("statevector_simulator")

    qc = QuantumCircuit(qb_cnt, qb_cnt)
    qc += QIS_QFT(qb_cnt, inverse=inverse)

    dqc = qc.decompose()

    job = execute(qc, simulator, shots=SHOT)
    res = job.result()
    state = res.get_statevector(qc)

    print(f"qiskit qft, qubits: {str(qb_cnt)}, inverse: {str(inverse)}")
    print(dqc.draw(fold=1000))
    print(f"depth: {dqc.depth()}, width: {dqc.width()}")
    print(state)
    print(res.get_counts())

    return state


@pytest.fixture()
def qiskit_qft_extra_state(qb_cnt, inverse):
    """
    qiskit qft don't support partial qft

    Returns:
        state:

    """
    simulator = BasicAer.get_backend("statevector_simulator")

    qc = QuantumCircuit(qb_cnt, qb_cnt)
    qc.h(0)
    qc.x(qb_cnt - 1)
    qc += QIS_QFT(qb_cnt, inverse=inverse)
    # qc.measure_all()

    dqc = qc.decompose()

    job = execute(qc, simulator, shots=SHOT)
    res = job.result()
    state = res.get_statevector(qc)

    print(f"qiskit qft extra, qubits: {str(qb_cnt)}, inverse: {inverse}")
    print(dqc.draw(fold=1000))
    print(state)
    print(res.get_counts())

    return state


def verify_state(dec_states, qiskit_states):
    # this test will run (MAX_QB_CNT - 2) * 2 times for each test case
    # 2 stands for inverse=[False, True]
    for dec_state, qiskit_state in zip(dec_states, qiskit_states):
        dec_real, dec_img = (float(num.strip()) for num in dec_state.split(","))
        assert math.isclose(dec_real, qiskit_state.real, abs_tol=TOLERANCE)
        assert math.isclose(dec_img, qiskit_state.imag, abs_tol=TOLERANCE)


def test_decom2qiskit(decomposition_qft_state, qiskit_qft_state):
    verify_state(decomposition_qft_state, qiskit_qft_state)


def test_decom2qiskit_extra(decomposition_qft_extra_state, qiskit_qft_extra_state):
    verify_state(decomposition_qft_extra_state, qiskit_qft_extra_state)
