import numpy as np
import paddle
from paddle.autograd import PyLayer

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
    gradients = np.array(gradients).T
    gradients = np.squeeze(gradients)
    return gradients

# define circuit layer by PyLayer
class CircuitLayer(PyLayer):
    @staticmethod
    def forward(ctx, circuit, obs, x):
        def circuit_func(circuit, obs, input_data):
            cc = circuit.bind_parameters({"phi": input_data})
            expval = cc.expval_pauli(obs)
            return expval
        expval = circuit_func(circuit, obs, x)
        out = paddle.to_tensor(expval, stop_gradient=False)
        ctx.save_for_backward(circuit, obs, x)
        
        ctx.func = circuit_func
        return out

    @staticmethod
    def backward(ctx, dy):
        circuit, obs, input_data, = ctx.saved_tensor()
        grad = parameter_shift(ctx.func, circuit, obs, input_data)
        return paddle.to_tensor(grad, stop_gradient=False)

phi = paddle.to_tensor(0.012, stop_gradient=False)

# define circuit by qutrunk
def def_circuit():
    circuit = QCircuit()
    q = circuit.allocate(1)
    phi = circuit.parameter("phi")
    Ry(phi) * q[0]
    return circuit, PauliZ(q[0])

circuit, obs = def_circuit()

# apply circuit layer
z = CircuitLayer.apply(circuit, obs, phi)
z.mean().backward()

# get gradient
print(phi.grad)
