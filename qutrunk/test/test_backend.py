from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure, All
from qutrunk.backends import BackendIBM

token = "247891ade16963963eb432d5ae4c7bbd1948d893f256a9f5d94af94628c5b57c73877dbf6ad4d2bd0ffc0e61d6aa001897666f1d75c3e613784ed8f2c7cafe68"

qc = QCircuit(backend=BackendIBM(token=token))

qr = qc.allocate(2)

H | qr[0]
CNOT | (qr[0], qr[1])

All(Measure) | qr

res = qc.run(shots=1024)

print(res.get_counts())
qc.draw()
