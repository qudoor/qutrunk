import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Ry, PauliZ

# calculate circuit gradient
def parameter_shift(circuit_func, circuit, obs, input_data, shift=np.pi/2):
    """ 
    Backward pass computation, calculate the gradient of quantum circuit by parameter shift rule.
    """
    input_list = np.array(input_data.tolist())
    
    shift_right = input_list + np.ones(input_list.shape) * shift
    shift_left = input_list - np.ones(input_list.shape) * shift
    
    gradients = []
    for i in range(len(input_list)):
        expectation_right = circuit_func(circuit, obs, shift_right[i])
        expectation_left  = circuit_func(circuit, obs, shift_left[i])
        
        gradient = np.array([expectation_right]) - np.array([expectation_left])
        gradients.append(gradient)
    gradients = np.squeeze(np.array(gradients).T)
    return gradients

def circuit_func(circuit, obs, input_data):
    cc = circuit.bind_parameters({"phi": input_data})
    expval = cc.expval_pauli(obs)
    return expval

# define circuit layer
class CircuitLayer(nn.Cell):
    def __init__(self, circuit, obs):
        self.circuit = circuit
        self.obs = obs

    def construct(self, x):
        expval = circuit_func(self.circuit, self.obs, x)
        out = Tensor(expval, dtype=ms.float32)
        return out

    def bprop(self, input, output, grad_output):
        grad = parameter_shift(circuit_func, self.circuit, self.obs, input)
        grad_input = ops.Cast()(grad, grad_output.dtype)
        return (grad_input,)


phi = Tensor(0.12, dtype=ms.float32)

# define circuit by qutrunk
def def_circuit():
    circuit = QCircuit()
    q = circuit.allocate(1)
    phi = circuit.parameter("phi")
    Ry(phi) * q[0]
    return circuit, PauliZ(q[0])

circuit, obs = def_circuit()

train_net = CircuitLayer(circuit, obs)

# optimizer 
optim = ms.nn.Adam([phi], 0.01)
# train
net = ms.nn.TrainOneStepCell(train_net, optim)
for i in range(100):
    print(net(phi))


# final phi after optimize
print("final_phi:", phi)

