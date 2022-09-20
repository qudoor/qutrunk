import numpy as np
#import matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.transforms import Compose, Normalize
from paddle.metric import Accuracy
from paddle.vision.datasets import MNIST

from qutrunk.qml.paddle.paddle_layer import PaddleLayer
from qutrunk.qml.circuit_qnn import CircuitQNN
from qutrunk.qml.qnn import qnn

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, H, Ry, Barrier, Measure, PauliZ


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = MNIST(mode='train', transform=transform)
test_dataset = MNIST(mode='test', transform=transform)
# Leaving only labels 0 and 1 
for item in range(len(train_dataset) - 1, -1, -1):
    img, label = train_dataset[item]
    if label != 0 and label != 1:
        train_dataset.images.pop(item)
        train_dataset.labels.pop(item)
print('load finished')


# define qnn network
@qnn(interface='paddle', grad='auto')
def circuit(params):
    circuit = QCircuit()
    phi = params[0]
    qr = circuit.allocate(1)
    H * qr[0]
    Ry(params[0]) * qr[0]
    expectation = circuit.expval(PauliZ(qr[0]))
    return np.array([expectation])

paddleqnn = PaddleLayer(circuit())

class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=2)

        # self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        # self.conv2 = nn.Conv2D(in_channels=6, out_channels=6, kernel_size=5)
        # self.dropout = nn.Dropout2D()
        # self.linear1 = nn.Linear(in_features=256, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=2)

        self.qnn = paddleqnn

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
        x = paddle.flatten(x, start_axis=0,stop_axis=-1)

        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        # x = self.dropout(x)
        # x = x.view(1, -1)
        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)

        x = self.qnn(x)
        return x


model = paddle.Model(LeNet())   # Encapsulate the Model with layer
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# Training modal
model.fit(train_dataset,
        epochs=2,
        batch_size=64,
        verbose=1
        )

# evaluate modal
model.evaluate(test_dataset, batch_size=64, verbose=1)
