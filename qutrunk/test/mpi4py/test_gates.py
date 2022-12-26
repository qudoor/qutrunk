import math
import sys
import trace

from qutrunk.backends import BackendLocal
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Swap, Sdg, Tdg, CNOT, MCX, All, CY, MCZ, U3, U2, U1, CRx, CRy, CRz, SqrtSwap


def test_swap():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    H * qr[0]
    Swap * (qr[0], qr[1])
    print(cir.get_statevector())


def test_sdg():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(1)
    H * qr[0]
    Sdg * qr[0]
    print(cir.get_statevector())


def test_tdg():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(1)
    H * qr[0]
    Tdg * qr[0]
    print(cir.get_statevector())


def test_sqrtswap():
    cir = QCircuit(backend=BackendLocal("mpi"))
    # cir = QCircuit()
    qr = cir.allocate(3)
    H * qr[0]
    SqrtSwap * (qr[0], qr[1])
    print(cir.get_statevector())


def test_cnot():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    H * qr[0]
    CNOT * (qr[0], qr[1])
    print(cir.get_statevector())


def test_mcx():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(3)
    H * qr[0]
    H * qr[1]
    MCX(2) * qr
    print(cir.get_statevector())


def test_cy():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    All(H) * qr
    CY * qr
    print(cir.get_statevector())


def test_mcz():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(3)
    All(H) * qr
    MCZ(2) * qr
    print(cir.get_statevector())


def test_u3():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(1)
    U3(math.pi / 2, 0, math.pi) * qr[0]
    print(cir.get_statevector())


def test_u2():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(1)
    U2(0, math.pi) * qr[0]
    print(cir.get_statevector())


def test_u1():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(1)
    H * qr[0]
    U1(math.pi / 4) * qr[0]
    print(cir.get_statevector())


def test_crx():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    All(H) * qr
    CRx(math.pi / 2) * (qr[0], qr[1])
    print(cir.get_statevector())


def test_cry():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    All(H) * qr
    CRy(math.pi / 2) * (qr[0], qr[1])
    print(cir.get_statevector())


def test_crz():
    cir = QCircuit(backend=BackendLocal("mpi"))
    qr = cir.allocate(2)
    All(H) * qr
    CRz(math.pi / 2) * (qr[0], qr[1])
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

tracer.runfunc(test_sqrtswap)
