from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, All, Measure, def_gate, X, MCX

import numpy as np
from qutrunk.circuit.gates.meta import Matrix

circuit = QCircuit()

q = circuit.allocate(4)

def_gate() << (Matrix([[-0.5, 0.5], [0.5, 0.5]], 2).inv(), (q[0], q[1], q[2])) \
        << (Matrix([[0.5, -0.5], [0.5, 0.5]]).ctrl().inv(), (q[0], q[1], q[2])) \
        << (Matrix([[0.5, 0.5], [-0.5, 0.5]]), q[0])
# This way can use defgate reduplicate
# defgate = def_gate()
# defgate << (Matrix([[-0.5, 0.5], [0.5, 0.5]], 2).inv(), (0, 1, 2)) \
#            << (Matrix([[0.5, -0.5], [0.5, 0.5]]).ctrl().inv(), (0, 1, 2)) \
#            << (Matrix([[0.5, 0.5], [-0.5, 0.5]]), 0)
# defgate * q

All(Measure) * q
circuit.print()
res = circuit.run(shots=100)
print(res.get_counts())