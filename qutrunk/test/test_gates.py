import pytest
import json

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import S, X, U1, U2, U3, CZ,\
    H, Measure, CNOT, Toffoli, P, R, Rx, Ry, Rz, S, Sdg, T, Tdg, X, Y, Z, MCX, MCZ,\
    NOT, Swap, SqrtSwap, SqrtX, All, CP, CX, CY, CZ, CRx, CRy, CRz, Rxx, Ryy, Rzz, \
    U1, U2, U3, Barrier, iSwap, CR, CU, CU1, CU3, X1, Y1

from numpy import pi
from qutrunk.backends import BackendQuSprout, backend
import math

PRECISION = 0.0000000001

def check_all_state(res, resbox):
    if len(res) != len(resbox):
        return False

    for index in range(len(res)):
        ampstr = res[index]
        ampstrbox = resbox[index]
        realstr, imagstr = ampstr.split(',')
        realstrbox, imagstrbox = ampstrbox.split(',')
        if (math.fabs(float(realstr) - float(realstrbox)) > PRECISION 
            or math.fabs(float(imagstr) - float(imagstrbox) > PRECISION)):
            return False
    
    return True

def test_h_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    H | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    H | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_p_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    P(pi/2) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    P(pi/2) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cp_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CP(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CP(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_r_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    R(pi/2, pi/2) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    R(pi/2, pi/2) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_rx_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Rx(pi/2) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Rx(pi/2) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_rxx_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    Rxx(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    Rxx(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_ryy_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    Ryy(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    Ryy(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_rzz_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    Rzz(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    Rzz(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_not_gate():
    test_x_gate()

def test_x_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    X | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    X | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_y_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Y | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_z_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Z | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Z | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_s_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    S | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    S | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_t_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    T | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    T | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_sdg_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Sdg | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Sdg | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_tdg_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Tdg | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Tdg | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_sqrtswap_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    SqrtSwap | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    SqrtSwap | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_swap_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    Swap | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    Swap | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cnot_gate():
    test_cx_gate()

def test_cx_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CX | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CX | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cy_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CY | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CY | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cz_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CZ | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CZ | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_u3_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U3(pi, 0, pi) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    U3(pi, 0, pi) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_u2_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U2(0, pi) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    U2(0, pi) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_u1_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    U1(pi/2) | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    U1(pi/2) | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_crx_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CRx(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CRx(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cry_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CRy(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CRy(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_crz_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CRz(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CRz(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_x1_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    X1 | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    X1 | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_y1_gate():
    circuit = QCircuit()
    qr = circuit.allocate(1)
    Y1 | qr[0]
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(1)
    Y1 | qrbox[0]
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cu1_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CU1(pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CU1(pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cu3_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CU3(pi/2,pi/2,pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CU3(pi/2,pi/2,pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_u_gate():
    test_u3_gate()

def test_cu_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CU(pi/2,pi/2,pi/2,pi/2) | (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CU(pi/2,pi/2,pi/2,pi/2) | (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_cr_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    CR(pi/2) * (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    CR(pi/2) * (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)

def test_iswap_gate():
    circuit = QCircuit()
    qr = circuit.allocate(2)
    iSwap(pi/2) * (qr[0], qr[1])
    res = circuit.get_all_state()

    circuitbox = QCircuit(backend=BackendQuSprout())
    qrbox = circuitbox.allocate(2)
    iSwap(pi/2) * (qrbox[0], qrbox[1])
    resbox = circuitbox.get_all_state()

    assert check_all_state(res, resbox)