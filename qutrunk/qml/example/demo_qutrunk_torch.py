import numpy as np
#import matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from qutrunk.core.circuit import QCircuit
from qutrunk.core.gates import All, H, Ry, Barrier, Measure
from enum import Enum

class TrainingType(Enum):
    """
    Training type for quantum circuit

    Attributes:
        Measure: Training by measure 
        ProbAmp: Training by get_prob_amp
    """
    Measure = 1
    ProbAmp = 2

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, shots, trainingtype=TrainingType.ProbAmp, weight=None):
        self.qubits = n_qubits
        self.shots = shots
        self.trainingtype = trainingtype

    def create_circuit(self, weight):
        # --- Circuit definition ---
        circuit = QCircuit()
        qr = circuit.allocate(self.qubits)
        theta = weight
        All(H) | qr
        Barrier | qr
        All(Ry(theta)) | qr
        return circuit
    
    def get_counts_states_by_probamp(self, circuit):
        circuit.backend.send_circuit(circuit)
        counts = []
        states = []
        num_elems = 2 ** self.qubits
        for sol_elem in range(num_elems):
            prob_amp = circuit.get_prob_amp(sol_elem)
            counts.append(prob_amp)
            states.append(sol_elem)
        counts = np.array(counts)
        states = np.array(states).astype(float)
        return counts, states

    def get_counts_states_by_measure(self, circuit):
        All(Measure) | circuit.qreg
        result = circuit.run(shots=self.shots)
        counts = np.array(list(result.get_values()))
        states = np.array(list(result.get_states())).astype(float)
        return counts, states

    def run(self, weight):
        circuit = self.create_circuit(weight)
        if (self.trainingtype == TrainingType.ProbAmp):
            counts, states = self.get_counts_states_by_probamp(circuit)
            # Compute probabilities for each state
            probabilities = counts
        else:
            counts, states = self.get_counts_states_by_measure(circuit)
            # Compute probabilities for each state
            probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    
# simulator = qiskit.Aer.get_backend('aer_simulator')

# circuit = QuantumCircuit(1, simulator, 100)
# print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
# circuit._circuit.draw()

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
    
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(float(input[0]))
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, shots, TrainingType.Measure)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

# training dataset
# Concentrating on the first 100 samples
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

# n_samples_show = 6

# data_iter = iter(train_loader)
# fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

# while n_samples_show > 0:
#     images, targets = data_iter.__next__()

#     axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
#     axes[n_samples_show - 1].set_xticks([])
#     axes[n_samples_show - 1].set_yticks([])
#     axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))
    
#     n_samples_show -= 1


# testing dataset
n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)

# training the network
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))


# plt.plot(loss_list)
# plt.title('Hybrid NN Training Convergence')
# plt.xlabel('Training Iterations')
# plt.ylabel('Neg Log Likelihood Loss')


# testing the network
model.eval()
with torch.no_grad():
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )