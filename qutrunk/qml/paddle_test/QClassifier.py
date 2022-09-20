# 导入 numpy、paddle 和 paddle_quantum
import numpy as np
import paddle
import paddle_quantum

# 构建量子电路
from paddle_quantum.ansatz import Circuit

# 一些用到的函数
from numpy import pi as PI
from paddle import matmul, transpose, reshape  # paddle 矩阵乘法与转置
from paddle_quantum.qinfo import pauli_str_to_matrix  # 得到 N 量子比特泡利矩阵,
from paddle_quantum.linalg import dagger # 复共轭

# 作图与计算时间
from matplotlib import pyplot as plt


import time

# 生成只作用在第一个量子比特上的泡利 Z 算符
# 其余量子比特上都作用单位矩阵
def Observable(n):
    r"""
    :param n: 量子比特数量
    :return: 局部可观测量: Z \otimes I \otimes ...\otimes I
    """
    Ob = pauli_str_to_matrix([[1.0, 'z0']], n)

    return Ob

# 用于绘制最终训练得到分类器的平面分类图
def heatmap_plot(Opt_Classifier, N):
    # 生成数据点 x_y_
    Num_points = 30
    x_y_ = []
    for row_y in np.linspace(0.9, -0.9, Num_points):
        row = []
        for row_x in np.linspace(-0.9, 0.9, Num_points):
            row.append([row_x, row_y])
        x_y_.append(row)
    x_y_ = np.array(x_y_).reshape(-1, 2).astype("float64")

    # 计算预测: heat_data
    input_state_test = paddle.to_tensor(
        datapoints_transform_to_state(x_y_, N))
    loss_useless, acc_useless, state_predict, cir = Opt_Classifier(state_in=input_state_test, label=x_y_[:, 0])
    heat_data = state_predict.reshape(Num_points, Num_points)

    # 画图
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x_label = np.linspace(-0.9, 0.9, 3)
    y_label = np.linspace(0.9, -0.9, 3)
    ax.set_xticks([0, Num_points // 2, Num_points - 1])
    ax.set_xticklabels(x_label)
    ax.set_yticks([0, Num_points // 2, Num_points - 1])
    ax.set_yticklabels(y_label)
    im = ax.imshow(heat_data, cmap=plt.cm.RdBu)
    plt.colorbar(im)
    plt.show()

# 圆形决策边界两分类数据集生成器
def circle_data_point_generator(Ntrain, Ntest, boundary_gap, seed_data):
    """
    :param Ntrain: 训练集大小
    :param Ntest: 测试集大小
    :param boundary_gap: 取值于 (0, 0.5), 两类别之间的差距
    :param seed_data: 随机种子
    :return: 四个列表：训练集x，训练集y，测试集x，测试集y
    """
    # 生成共Ntrain + Ntest组数据，x对应二维数据点，y对应编号
    # 取前Ntrain个为训练集，后Ntest个为测试集
    train_x, train_y = [], []
    num_samples, seed_para = 0, 0
    while num_samples < Ntrain + Ntest:
        np.random.seed((seed_data + 10) * 1000 + seed_para + num_samples)
        data_point = np.random.rand(2) * 2 - 1  # 生成[-1, 1]范围内二维向量

        # 如果数据点的模小于(0.7 - gap)，标为0
        if np.linalg.norm(data_point) < 0.7 - boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(0.)
            num_samples += 1

        # 如果数据点的模大于(0.7 + gap)，标为1
        elif np.linalg.norm(data_point) > 0.7 + boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(1.)
            num_samples += 1
        else:
            seed_para += 1

    train_x = np.array(train_x).astype("float64")
    train_y = np.array([train_y]).astype("float64").T

    print("训练集的维度大小 x {} 和 y {}".format(np.shape(train_x[0:Ntrain]), np.shape(train_y[0:Ntrain])))
    print("测试集的维度大小 x {} 和 y {}".format(np.shape(train_x[Ntrain:]), np.shape(train_y[Ntrain:])), "\n")

    return train_x[0:Ntrain], train_y[0:Ntrain], train_x[Ntrain:], train_y[Ntrain:]


# 搭建整个优化流程图
class Opt_Classifier(paddle_quantum.gate.Gate):
    """
    创建模型训练网络
    """
    def __init__(self, n, depth, seed_paras=1):
        # 初始化部分，通过n, depth给出初始电路
        super(Opt_Classifier, self).__init__()
        self.n = n
        self.depth = depth
        # 初始化偏置 (bias)
        self.bias = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Normal(std=0.01),
            dtype='float32',
            is_bias=False)
        
        self.circuit = Circuit(n)
        # 先搭建广义的旋转层
        for i in range(n):
            self.circuit.rz(qubits_idx=i)
            self.circuit.ry(qubits_idx=i)
            self.circuit.rz(qubits_idx=i)

        # 默认深度为 depth = 1
        # 对每一层搭建电路
        for d in range(3, depth + 3):
            # 搭建纠缠层
            for i in range(n-1):
                self.circuit.cnot(qubits_idx=[i, i + 1])
            self.circuit.cnot(qubits_idx=[n-1, 0])
            # 对每一个量子比特搭建Ry
            for i in range(n):
                self.circuit.ry(qubits_idx=i)

    # 定义前向传播机制、计算损失函数 和交叉验证正确率
    def forward(self, state_in, label):
        """
        输入： state_in：输入量子态，shape: [-1, 1, 2^n] -- 此教程中为[BATCH, 1, 2^n]
               label：输入量子态对应标签，shape: [-1, 1]
        计算损失函数:
                L = 1/BATCH * ((<Z> + 1)/2 + bias - label)^2
        """
        # 将 Numpy array 转换成 tensor
        Ob = paddle.to_tensor(Observable(self.n))
        label_pp = reshape(paddle.to_tensor(label), [-1, 1])

        # 按照随机初始化的参数 theta 
        Utheta = self.circuit.unitary_matrix()

        # 因为 Utheta是学习到的，我们这里用行向量运算来提速而不会影响训练效果
        state_out = matmul(state_in, Utheta)  # [-1, 1, 2 ** n]形式，第一个参数在此教程中为BATCH

        # 测量得到泡利 Z 算符的期望值 <Z> -- shape [-1,1,1]
        E_Z = matmul(matmul(state_out, Ob), transpose(paddle.conj(state_out), perm=[0, 2, 1]))

        # 映射 <Z> 处理成标签的估计值 
        state_predict = paddle.real(E_Z)[:, 0] * 0.5 + 0.5 + self.bias  # 计算每一个y^{i,k}与真实值得平方差
        loss = paddle.mean((state_predict - label_pp) ** 2)  # 对BATCH个得到的平方差取平均，得到L_i：shape:[1,1]

        # 计算交叉验证正确率
        is_correct = (paddle.abs(state_predict - label_pp) < 0.5).nonzero().shape[0]
        acc = is_correct / label.shape[0]

        return loss, acc, state_predict.numpy(), self.circuit

# 构建绕 Y 轴，绕 Z 轴旋转 theta 角度矩阵
def Ry(theta):
    """
    :param theta: 参数
    :return: Y 旋转矩阵
    """
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

def Rz(theta):
    """
    :param theta: 参数
    :return: Z 旋转矩阵
    """
    return np.array([[np.cos(theta / 2) - np.sin(theta / 2) * 1j, 0],
                     [0, np.cos(theta / 2) + np.sin(theta / 2) * 1j]])

# 经典 -> 量子数据编码器
def datapoints_transform_to_state(data, n_qubits):
    """
    :param data: 形状为 [-1, 2]，numpy向量形式
    :param n_qubits: 数据转化后的量子比特数量
    :return: 形状为 [-1, 1, 2 ^ n_qubits]
            形状中-1表示第一个参数为任意大小。在此教程实例分析中，对应于BATCH，用以得到Eq.(1)中平方误差的平均值
    """
    dim1, dim2 = data.shape
    res = []
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0]])
        # 角度编码
        for i in range(n_qubits):
            # 对偶数编号量子态作用 Rz(arccos(x0^2)) Ry(arcsin(x0))
            if i % 2 == 0:
                state_tmp=np.dot(zero_state, Ry(np.arcsin(data[sam][0])).T)
                state_tmp=np.dot(state_tmp, Rz(np.arccos(data[sam][0] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
            # 对奇数编号量子态作用 Rz(arccos(x1^2)) Ry(arcsin(x1))
            elif i % 2 == 1:
                state_tmp=np.dot(zero_state, Ry(np.arcsin(data[sam][1])).T)
                state_tmp=np.dot(state_tmp, Rz(np.arccos(data[sam][1] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
        res.append(res_state)
    res = np.array(res, dtype=paddle_quantum.get_dtype())

    return res


def QClassifier(Ntrain, Ntest, gap, N, DEPTH, EPOCH, LR, BATCH, seed_paras, seed_data):
    """
    量子二分类器
    输入参数：
        Ntrain,        # 规定训练集大小
        Ntest,         # 规定测试集大小
        gap,           # 设定决策边界的宽度
        N,             # 所需的量子比特数量
        DEPTH,         # 采用的电路深度
        BATCH,         # 训练时 batch 的大小
        EPOCH,         # 训练 epoch 轮数
        LR,            # 设置学习速率
        seed_paras,    # 设置随机种子用以初始化各种参数
        seed_data,     # 固定生成数据集所需要的随机种子
    """
    # 生成训练集测试集
    train_x, train_y, test_x, test_y = circle_data_point_generator(Ntrain=Ntrain, Ntest=Ntest, boundary_gap=gap, seed_data=seed_data)
    # 读取训练集的维度
    N_train = train_x.shape[0]

    paddle.seed(seed_paras)
    # 初始化寄存器存储正确率 acc 等信息
    summary_iter, summary_test_acc = [], []

    # 一般来说，我们利用Adam优化器来获得相对好的收敛
    # 当然你可以改成SGD或者是RMSprop
    myLayer = Opt_Classifier(n=N, depth=DEPTH)  # 得到初始化量子电路
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=myLayer.parameters())

    # 优化循环
    # 此处将训练集分为Ntrain/BATCH组数据，对每一组训练后得到的量子线路作为下一组数据训练的初始量子电路
    # 故通过cir记录每组数据得到的最终量子线路
    i = 0  # 记录总迭代次数
    for ep in range(EPOCH):
        # 将训练集分组，对每一组训练
        for itr in range(N_train // BATCH):
            i += 1  # 记录总迭代次数
            # 将经典数据编码成量子态 |psi>, 维度 [BATCH, 2 ** N]
            input_state = paddle.to_tensor(datapoints_transform_to_state(train_x[itr * BATCH:(itr + 1) * BATCH], N))

            # 前向传播计算损失函数
            loss, train_acc, state_predict_useless, cir \
                = myLayer(state_in=input_state, label=train_y[itr * BATCH:(itr + 1) * BATCH])  # 对此时量子电路优化
            # 显示迭代过程中performance变化
            if i % 30 == 5:
                # 计算测试集上的正确率 test_acc
                input_state_test = paddle.to_tensor(datapoints_transform_to_state(test_x, N))
                loss_useless, test_acc, state_predict_useless, t_cir \
                    = myLayer(state_in=input_state_test,label=test_y)
                print("epoch:", ep, "iter:", itr,
                      "loss: %.4f" % loss.numpy(),
                      "train acc: %.4f" % train_acc,
                      "test acc: %.4f" % test_acc)
                # 存储正确率 acc 等信息
                summary_iter.append(itr + ep * N_train)
                summary_test_acc.append(test_acc) 

            # 反向传播极小化损失函数
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()

    # 得到训练后电路
    print("训练后的电路：")
    print(cir)
    # 画出 heatmap 表示的决策边界
    heatmap_plot(myLayer, N=N)

    return summary_test_acc

def main():
    """
    主函数
    """
    time_start = time.time()
    acc = QClassifier(
        Ntrain = 200,        # 规定训练集大小
        Ntest = 100,         # 规定测试集大小
        gap = 0.5,           # 设定决策边界的宽度
        N = 4,               # 所需的量子比特数量
        DEPTH = 1,           # 采用的电路深度
        BATCH = 20,          # 训练时 batch 的大小
        EPOCH = 20,          
                             # 训练 epoch 轮数，使得总迭代次数 EPOCH * (Ntrain / BATCH) 在200左右
        LR = 0.01,            # 设置学习速率
        seed_paras = 19,     # 设置随机种子用以初始化各种参数
        seed_data = 2,       # 固定生成数据集所需要的随机种子
    )
    
    time_span = time.time() - time_start
    print('主程序段总共运行了', time_span, '秒')

if __name__ == '__main__':
    main()