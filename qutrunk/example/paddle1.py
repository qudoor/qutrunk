import numpy as np
import matplotlib as plt
import random
import paddle
from typing import Union, Optional

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Rx, Rz, CNOT

def circuit_ansatz(num_qubits, depth):
    circuit = QCircuit()
    q = circuit.allocate(num_qubits)
    angles = ["theta-" + str(i) for i in range(2 * num_qubits * depth)]
    params = circuit.parameters(angles)
    print(params)
    print(angles)

    idx = 0
    for i in range(depth):
        for j in range(num_qubits):
            Rx(params[idx]) * q[j]
            idx += 1
        for j in range(num_qubits):
            Rz(params[idx]) * q[j]
            idx += 1
        for j in range(num_qubits - 1):
            CNOT * (q[j], q[j+1])
    
    return circuit

psi_target = np.kron(
    np.kron(np.array([1, 0]), np.array([0, 1])),
    np.array([1/np.sqrt(2), 1/np.sqrt(2)])
)  # <01+|
psi_target = paddle.to_tensor(psi_target, dtype='complex64')

class StateFidelity(paddle.nn.Layer):
    r"""The basic class to implement the quantum operation.

    Args:
        backend: The backend implementation of the operator.
            Defaults to ``None``, which means to use the default backend implementation.
        dtype: The data type of the operator.
            Defaults to ``None``, which means to use the default data type.
        name_scope: Prefix name used by the operator to name parameters. Defaults to ``None``.
    """
    def __init__(self, target_state):
        super().__init__()
        self.target_state = target_state

    def forward(self, state):
        r"""Compute the state fidelity between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the state fidelity with the target state.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The state fidelity between the input state and the target state.
        """
        state = paddle.unsqueeze(state, axis=1)
        target_state = paddle.unsqueeze(self.target_state, axis=1)
        target_state_dagger = paddle.conj(paddle.t(target_state))
        fidelity = paddle.abs(paddle.matmul(target_state_dagger, state))
        fidelity = paddle.reshape(fidelity, [1])
        
        return fidelity


# tensor
psi_target = paddle.to_tensor(psi_target)
fid_func = StateFidelity(psi_target)

# 首先，我们给出一些训练用参数
ITR = 115      # 学习迭代次数
LR = 0.2       # 学习速率

# 记录迭代中间过程:
loss_list = []
parameter_list = []
# 构造线路
num_qubits = 3
depth = 2
circuit = circuit_ansatz(num_qubits, depth)
# 随机构造参数，并绑定到线路上
angles = [random.random() * 10 for _ in range(2 * num_qubits * depth)]
# paddle tensor
# paddle.create_parameter()
angles = [paddle.to_tensor(angles[i], stop_gradient=False) for i in range(len(angles))]

# 选择优化器，通常选用Adam
opt = paddle.optimizer.Adam(learning_rate = LR, parameters = angles)
# 迭代优化
for itr in range(ITR):
    params = {"theta-" + str(i): angles[i] for i in range(len(angles))}
    cir = circuit.bind_parameters(params)
    state = cir.get_all_state()
    state = paddle.to_tensor(state, stop_gradient=False)
    # 计算损失函数
    loss = -fid_func(state)
    # 通过梯度下降算法优化
    loss.backward()
    opt.minimize(loss)
    opt.clear_grad()
    # 记录学习曲线
    loss_list.append(loss.numpy()[0])
    if itr % 10 == 0:
        print('iter:', itr, '  loss: %.4f' % loss.numpy())
