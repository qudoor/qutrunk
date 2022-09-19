import pytest

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import X
from qutrunk.backends import BackendQuSprout
from check_all_state import check_all_state
from check_all_state_inverse import check_all_state_inverse


def test_x_gate():
    """测试X门"""
    # 使用本地量子计算模拟器
    circuit = QCircuit()
    qr = circuit.allocate(1)
    X * qr[0]
    res = circuit.get_all_state()

    # 使用BackendQuSprout量子计算模拟器
    circuit_box = QCircuit(backend=BackendQuSprout())
    qr_box = circuit_box.allocate(1)
    X * qr_box[0]
    res_box = circuit_box.get_all_state()

    # 检查数据是否一致
    assert check_all_state(res, res_box)


def test_x_inverse_gate():
    """测试反转电路"""
    # 使用本地量子计算模拟器
    circuit = QCircuit()
    qr = circuit.allocate(1)
    # 获取原始数据
    org_res = circuit.get_all_state()

    # 进行逆操作
    X * qr[0]
    X * qr[0]
    circuit.cmds[1].inverse = True

    # 获取逆操作后数据
    final_res = circuit.get_all_state()

    # 检查逆操作前后数据是否一致
    assert check_all_state_inverse(org_res, final_res)


if __name__ == "__main__":
    """运行test文件"""
    pytest.main(["-v", "-s", "./test_x_gate.py"])
