import math

import pytest
from numpy import pi

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CP
from qutrunk.test.gate.global_parameters import PRECISION


def check_all_state(res, res_box):
    if len(res) != len(res_box):
        return False

    for index in range(len(res)):
        amp_str = res[index]
        amp_str_box = res_box[index]
        real_str, image_str = amp_str.split(',')
        real_str_box, image_str_box = amp_str_box.split(',')
        test = float(image_str) - float(image_str_box)
        test1 = math.fabs(test)
        a = test1 > PRECISION
        if (math.fabs(float(real_str) - float(real_str_box)) > PRECISION
                or math.fabs(float(image_str) - float(image_str_box)) > PRECISION):
            return False

    return True


def test_cp_inverse_gate():
    # 使用本地量子计算模拟器
    circuit = QCircuit()
    qr = circuit.allocate(2)
    # 获取原始数据
    org_res = circuit.get_all_state()

    # 进行逆操作
    CP(pi / 2) * (qr[0], qr[1])
    CP(pi / 2) * (qr[0], qr[1])
    circuit.cmds[1].inverse = True

    # 获取逆操作后数据
    final_res = circuit.get_all_state()

    # 检查逆操作前后数据是否一致
    assert check_all_state(org_res, final_res)


if __name__ == "__main__":
    """运行test文件"""
    pytest.main(["-v", "-s", "./test_cp_inverse_gate.py"])