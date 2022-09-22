import pytest
from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.backends import BackendQuSprout
from qutrunk.circuit.ops import QSP
from qutrunk.test.gate.check_all_state import check_all_state

def test_amplitudes_local():
    """test set amp for local"""
    # 使用本地量子计算模拟器
    circuit = QCircuit()
    qr = circuit.allocate(2)
    QSP("AMP", [1, 2, 3]) * qr

    res = []
    for i in range(len(circuit.init_amp_reals)): 
        res.append(','.join([str(circuit.init_amp_reals[i]), str(circuit.init_amp_imags[i])]))

    res_box = circuit.get_all_state()

    # 检查数据是否一致
    assert check_all_state(res, res_box)


def test_amplitudes_qusprout():
    """test set amp for qusprout"""
    # 使用qusprout量子计算模拟器
    circuit = QCircuit(backend=BackendQuSprout())
    qr = circuit.allocate(3)
    QSP("AMP", [1, 2, 3, 4, 5]) * qr

    res = []
    for i in range(len(circuit.init_amp_reals)): 
        res.append(','.join([str(circuit.init_amp_reals[i]), str(circuit.init_amp_imags[i])]))

    res_box = circuit.get_all_state()

    # 检查数据是否一致
    assert check_all_state(res, res_box)

test_amplitudes_qusprout()