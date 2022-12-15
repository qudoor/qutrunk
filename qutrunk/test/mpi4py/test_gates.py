import sys
import trace

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Swap, Sdg, Tdg, CNOT, MCX, All, CY, MCZ


def test_swap():
    cir = QCircuit()
    qr = cir.allocate(2)
    H * qr[0]
    Swap * (qr[0], qr[1])
    print(cir.get_statevector())


def test_sdg():
    cir = QCircuit()
    qr = cir.allocate(1)
    H * qr[0]
    Sdg * qr[0]
    print(cir.get_statevector())


def test_tdg():
    cir = QCircuit()
    qr = cir.allocate(1)
    H * qr[0]
    Tdg * qr[0]
    print(cir.get_statevector())


def test_cnot():
    cir = QCircuit()
    qr = cir.allocate(2)
    H * qr[0]
    CNOT * (qr[0], qr[1])
    print(cir.get_statevector())


def test_mcx():
    cir = QCircuit()
    qr = cir.allocate(3)
    H * qr[0]
    H * qr[1]
    MCX(2) * qr
    print(cir.get_statevector())


def test_cy():
    cir = QCircuit()
    qr = cir.allocate(2)
    All(H) * qr
    CY * qr
    print(cir.get_statevector())


def test_mcz():
    cir = QCircuit()
    qr = cir.allocate(3)
    All(H) * qr
    MCZ(2) * qr
    print(cir.get_statevector())


# define Trace object: trace line numbers at runtime, exclude some modules
tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    ignoremods=[
        'inspect', 'contextlib', '_bootstrap',
        '_weakrefset', 'abc', 'posixpath', 'genericpath', 'textwrap'
    ],
    trace=0,
    count=1)

# by default trace goes to stdout
# redirect to a different file for each processes
# sys.stdout = open('trace_{:04d}.txt'.format(COMM_WORLD.rank), 'w')

tracer.runfunc(test_mcz)
