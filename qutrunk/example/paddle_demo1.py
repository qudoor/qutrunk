# 加入需要用到的包
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle_quantum
from paddle_quantum import Hamiltonian

def create_circuit(num_qubits, depth):
    # 步骤1.1：构建 N 量子比特线路
    circuit = paddle_quantum.ansatz.Circuit(num_qubits)
    # 步骤1.2：对每一层添加量子门
    for _ in range(0, depth):
        circuit.rx('full')
        circuit.rz('full')
        circuit.cnot('linear')
    return circuit

num_qubits = 3
depth = 2
cir = create_circuit(num_qubits, depth)
print(cir)

psi_target = np.kron(
    np.kron(np.array([1, 0]), np.array([0, 1])),
    np.array([1/np.sqrt(2), 1/np.sqrt(2)])
)  # <01+|
psi_target = paddle_quantum.state.to_state(paddle.to_tensor(psi_target), dtype=paddle_quantum.get_dtype())
fid_func = paddle_quantum.loss.StateFidelity(psi_target)

# 首先，我们给出一些训练用参数
ITR = 115      # 学习迭代次数
LR = 0.2       # 学习速率

# 记录迭代中间过程:
loss_list = []
parameter_list = []
# 选择优化器，通常选用Adam
opt = paddle.optimizer.Adam(learning_rate = LR, parameters = cir.parameters())
# 迭代优化
for itr in range(ITR):
    state = cir()
    # 计算损失函数
    loss = -fid_func(state)
    # 通过梯度下降算法优化
    loss.backward()
    opt.minimize(loss)
    opt.clear_grad()
    # 记录学习曲线
    loss_list.append(loss.numpy()[0])
    parameter_list.append(cir.param.numpy())
    if itr % 10 == 0:
        print('iter:', itr, '  loss: %.4f' % loss.numpy())

# 输出最终损失函数值
print('The minimum of the loss function:', loss_list[-1])
# 输出最终量子电路参数
theta_final = parameter_list[-1]
print("Parameters after optimizationL theta:\n", theta_final)
# 绘制最终电路与输出量子态
# 输入量子电路参数需要转化为paddle.tensor类型
theta_final = paddle.to_tensor(theta_final)
# 绘制电路
print(cir)

# 最终得到量子态
state_final = cir()
print("state_final:\n", state_final)

# 绘制迭代过程中损失函数变化曲线
plt.figure(1)
ITR_list = []
for i in range(ITR):
    ITR_list.append(i)
func = plt.plot(ITR_list, loss_list, alpha=0.7, marker='', linestyle='-', color='r')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend(labels=["loss function during iteration"], loc='best')
plt.show()