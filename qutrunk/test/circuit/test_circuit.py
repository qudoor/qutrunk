"""Test the functions of circuit."""
import math
import pytest
from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import (
    H,
    Measure,
    CNOT,
    iSwap,
    T,
    Swap,
    SqrtSwap,
    CH,
    All,
    Ry,
    Rx,
    PauliCoeff,
    PauliType,
    PauliCoeffs,
    PauliZ,
    PauliI,
)

from qutrunk.test.global_parameters import PRECISION

from qutrunk.backends import BackendLocal, BackendQuSprout


class TestCircuit:
    def test_backend(self):
        """Test the backend function in QCircuit."""
        circuit1 = QCircuit()
        q1 = circuit1.allocate(2)
        assert type(circuit1.backend) is BackendLocal

    def test_depth(self):
        """Test the depth function in QCircuit."""
        qc = QCircuit()
        qreg = qc.allocate(5)
        H * qreg[0]
        H * qreg[1]
        H * qreg[2]
        iSwap * (qreg[0], qreg[4])
        T * qreg[1]
        H * qreg[0]
        Swap * (qreg[1], qreg[4])
        CH * (qreg[0], qreg[1])
        SqrtSwap * (qreg[2], qreg[4])
        T * qreg[0]
        H * qreg[1]
        CNOT * (qreg[0], qreg[2])
        CH * (qreg[1], qreg[2])
        H * qreg[2]
        All(Measure) * qreg

        result = qc.depth()
        expect = 9
        assert result == expect

    def test_expval_pauli(self):
        """Test the expval_pauli function in QCircuit."""
        circuit = QCircuit()
        q = circuit.allocate(2)
        Ry(pi) * q[0]
        pauli_str = [PauliZ(q[0]), PauliI(q[1])]

        result = circuit.expval_pauli(pauli_str)
        expect = -1
        assert math.fabs(result - expect) < PRECISION

    def test_expval_pauli_sum(self):
        """Test the expval_pauli_sum function in QCircuit."""
        circuit = QCircuit()
        q = circuit.allocate(2)
        H * q[0]
        Ry(pi) * q[1]
        pauli_coeffs = (
            PauliCoeffs()
            << PauliCoeff(0.12, [PauliType.PAULI_Z])
            << PauliCoeff(0.34, [PauliType.PAULI_X, PauliType.PAULI_I])
        )

        result = circuit.expval_pauli_sum(pauli_coeffs)
        expect = 0.34
        assert math.fabs(result - expect) < PRECISION

    def test_dump_qusl(self):
        """Testing the dump qusl function in QCircuit."""
        qc = QCircuit()
        qreg = qc.allocate(2)
        H * qreg[0]
        CNOT * (qreg[0], qreg[1])
        Measure * qreg[0]
        Measure * qreg[1]
        qc.dump(file="bell_pair.qusl", format="qusl")
        expect = '"code": ["H * q[0]\\n", "MCX(1) * (q[0], q[1])\\n", "Measure * q[0]\\n", "Measure * q[1]\\n"]'

        with open("bell_pair.qusl", "r") as stream:
            container = stream.readline()
            result = container[container.rfind("code") - 1 : -1]

        assert result == expect

    def test_dump_openqasm(self):
        """Testing the dump qasm function in QCircuit."""
        qc = QCircuit()
        qreg = qc.allocate(2)
        H * qreg[0]
        CNOT * (qreg[0], qreg[1])
        Measure * qreg[0]
        Measure * qreg[1]
        qc.dump(file="bell_pair.qasm", format="openqasm")
        expect = [
            "OPENQASM 2.0;\n",
            'include "qelib1.inc";\n',
            "qreg q[2];\n",
            "creg c[2];\n",
            "h q[0];\n",
            "cx q[0],q[1];\n",
            "measure q[0] -> c[0];\n",
            "measure q[1] -> c[1];\n",
        ]

        with open("bell_pair.qasm", "r") as stream:
            result = stream.readlines()

        assert result == expect

    def test_load_qusl(self):
        """Test the load qusl function in QCircuit."""
        qc = QCircuit()
        qreg = qc.allocate(2)
        H * qreg[0]
        CNOT * (qreg[0], qreg[1])
        Measure * qreg[0]
        Measure * qreg[1]

        expect = qc.cmds
        result = qc.load(file="bell_pair.qusl", format="qusl")
        assert expect == result.cmds

    def test_load_openqasm(self):
        """Test the load qasm function in QCircuit."""
        qc = QCircuit()
        qreg = qc.allocate(2)
        H * qreg[0]
        CNOT * (qreg[0], qreg[1])
        Measure * qreg[0]
        Measure * qreg[1]

        expect = qc.cmds
        result = qc.load(file="bell_pair.qasm", format="openqasm")
        assert expect == result.cmds

    def test_get_prob(self):
        """Test the get_prob function in QCircuit."""
        circuit = QCircuit()
        qr = circuit.allocate(2)
        H * qr[0]
        CNOT * (qr[0], qr[1])

        result = circuit.get_prob(0)
        expect = 0.5
        assert math.fabs(result - expect) < PRECISION

    def test_get_probs(self):
        """Test the get_probs function in QCircuit."""
        circuit = QCircuit()
        qr = circuit.allocate(2)
        H * qr[0]
        CNOT * (qr[0], qr[1])

        result = circuit.get_probs()
        assert math.fabs(result[0]["prob"] - 0.5) < PRECISION
        assert math.fabs(result[1]["prob"] - 0) < PRECISION
        assert math.fabs(result[2]["prob"] - 0) < PRECISION
        assert math.fabs(result[3]["prob"] - 0.5) < PRECISION

    def test_get_statevector(self):
        """Test the get_statevector function in QCircuit."""
        circuit = QCircuit()
        qr = circuit.allocate(2)
        H * qr[0]
        CNOT * (qr[0], qr[1])

        result = circuit.get_statevector()
        assert result[0] == complex(1 / math.sqrt(2))
        assert result[1] == complex(0)
        assert result[2] == complex(0)
        assert result[3] == complex(1 / math.sqrt(2))

    def test_append_circuit(self):
        """Test the append_circuit function in QCircuit."""
        circ1 = QCircuit()
        q1 = circ1.allocate(2)
        H * q1[0]
        CNOT * (q1[0], q1[1])

        circ2 = QCircuit()
        q2 = circ2.allocate(2)
        circ2.append_circuit(circ1)

        assert circ1.cmds == circ2.cmds

    def test_bind_parameters(self):
        """Test the bind_parameters function in QCircuit."""
        circuit = QCircuit()
        q = circuit.allocate(2)
        theta, phi = circuit.create_parameters(["theta", "phi"])
        Ry(theta) * q[0]
        Ry(phi) * q[1]

        circuit.bind_parameters({"theta": pi, "phi": pi / 2})
        assert circuit.cmds[0].rotation[0] == pi
        assert circuit.cmds[1].rotation[0] == pi / 2

    def test_inverse(self):
        """Test the inverse function in QCircuit."""
        circuit = QCircuit()
        q = circuit.allocate(2)
        Rx(pi).inv() * q[0]
        Ry(pi) * q[1]

        inverse_circuit, inverse_circuit_q = circuit.inverse()

        assert inverse_circuit.cmds[0].inverse == 1
        assert inverse_circuit.cmds[1].inverse == 0
        assert inverse_circuit.cmds[0].gate == circuit.cmds[1].gate
        assert inverse_circuit.cmds[1].gate == circuit.cmds[0].gate
