from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import *
from qutrunk.circuit.ops import *

def generate_circuit():
	circuit = QCircuit()
	q = circuit.allocate(2)
	H * q[0]
	MCX(1) * (q[0], q[1])
	Measure * q[0]
	Measure * q[1]
	return circuit

generate_circuit()