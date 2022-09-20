from functools import wraps
from qutrunk.qml.circuit_qnn import CircuitQNN
from qutrunk.qml.interface import MLInterface
 
 
class qnn(object):
    def __init__(self, interface, grad='auto'):
        self.interface = interface
        self.grad = grad
        self.callable = None
 
    def __call__(self, func):
        self.callable = func
        @wraps(func)
        def inner(*args, **kwargs):
            return CircuitQNN(circuit_func=self.callable, interface=self.interface, grad=self.grad)
        return inner