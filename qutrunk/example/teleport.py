"""GHZ state example."""

from qutrunk.circuit import QCircuit, MeasureCond
from qutrunk.circuit.gates import CX, Measure, H, Barrier, All, U3, Z, X
from qutrunk.backends import BackendQuSprout
from qutrunk.backends import BackendQuRoot

def run_teleport(backend=None):
    # Create quantum circuit
    qc = QCircuit(name="teleport", backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(3)

    # Prepare an initial state
    U3(0.3, 0.2, 0.1) * qr[0]

    # Prepare a Bell pair
    H * qr[1]
    CX * (qr[1], qr[2])

    # Barrier following state preparation
    Barrier * qr

    # Measure in the Bell basis
    CX * (qr[0], qr[1])
    H * qr[0]
    Measure * qr[0]
    Measure * qr[1]

    # Apply a correction
    Barrier * qr
    Z.condition(0, 1) * qr[2]
    X.condition(1, 1) * qr[2]
    Measure * qr[2]

    # Run quantum circuit with 1024 times
    print(qc.get_statevector())
    res = qc.run(shots=1)

    # Print measure results like:
    # [{"000": 527}, {"111": 497}]
    print(res.get_counts())
    #print(res.get_bitstrs())
    print(res.get_measures())
    #print(res.get_values())
    
    return qc


if __name__ == "__main__":
    # Run locally
    #circuit = run_teleport(backend=BackendQuSprout("cpu", "192.168.158.152", 9091))
    circuit = run_teleport()

    # Dram quantum circuit
    circuit.draw()

