import paddle
from paddle.vision.transforms import Compose, Normalize

import numpy as np

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, H, PauliZ, Ry, Rz, Rx, CNOT
from qutrunk.qml.qnn import qnn
from qutrunk.qml.paddle.paddle_layer import PaddleLayer
 
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')


@qnn(interface='paddle', grad='auto')
def circuit(params):
    batch = params.shape[0]
    res = []
    size = params.shape[1]
    circuit = QCircuit()
    qr = circuit.allocate(size)
    All(H) * qr
    par = params[0].numpy().tolist()
    for i in range(size):
        Ry(par[i]) * qr[i]
        # Rz(par[i]) * qr[i]
        # if i % 2 == 0 and i < size - 1:
        #     CNOT * (qr[i], qr[i+1])


    exp_values = [circuit.expval(PauliZ(qr[i])) for i in range(size)]

    for i in range(batch):
        res.append(exp_values)
    return res

import paddle
import paddle.nn.functional as F
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)
        self.qnn = paddleqnn = PaddleLayer(circuit())

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.qnn(x)
        return x
from paddle.metric import Accuracy
model = paddle.Model(LeNet())   # 用Model封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,
        epochs=2,
        batch_size=64,
        verbose=1
        )

model.evaluate(test_dataset, batch_size=64, verbose=1)
#模型验证

