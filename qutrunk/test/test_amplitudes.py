import pytest
from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendQuSprout
from qutrunk.circuit.ops import QSP
from qutrunk.test.gate.check_all_state import check_all_state

def test_amplitudes_local():
    """test set amp for local"""
    # 使用本地量子计算模拟器
    qubit_len = 3
    circuit = QCircuit()
    qr = circuit.allocate(qubit_len)

    amplist = [1-2j, 2+3j, 3-4j, 0.5+0.7j]
    startind = 0
    numamps = 3
    QSP("AMP", amplist, startind, numamps) * qr

    if numamps > len(amplist):
        numamps = len(amplist)

    res = ['0, 0'] * (2**qubit_len)
    res[0] = '1, 0'

    for i in range(numamps): 
        res[startind] = ','.join([str(qr.circuit.cmds[0].cmdex.amp.reals[i]), str(qr.circuit.cmds[0].cmdex.amp.imags[i])])
        startind += 1

    res_box = circuit.get_all_state()

    # 检查数据是否一致
    assert check_all_state(res, res_box)

test_amplitudes_local()

def test_amplitudes_qusprout():
    """test set amp for local"""
    # 使用本地量子计算模拟器
    qubit_len = 3
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(qubit_len)

    amplist = [1, 2, 3, 4, 5]
    startind = 2
    numamps = 6
    QSP("AMP", amplist, startind, numamps) * qr

    if numamps > len(amplist):
        numamps = len(amplist)

    res = ['0, 0'] * (2**qubit_len)
    res[0] = '1, 0'

    for i in range(numamps): 
        res[startind] = ','.join([str(qr.circuit.cmds[0].cmdex.amp.reals[i]), str(qr.circuit.cmds[0].cmdex.amp.imags[i])])
        startind += 1

    res_box = circuit.get_all_state()

    # 检查数据是否一致
    assert check_all_state(res, res_box)

test_amplitudes_qusprout()