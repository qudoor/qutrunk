"""load the data"""
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Concentrating on the first 100 samples
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

# 训练数据集
train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

# 测试数据集
n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


def show_data():
    n_samples_show = 6

    data_iter = iter(train_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    while n_samples_show > 0:
        images, targets = data_iter.__next__()

        axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))

        n_samples_show -= 1
    plt.show()


if __name__ == '__main__':
    show_data()