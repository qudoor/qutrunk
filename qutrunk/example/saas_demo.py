from qutrunk.backends import BackendQuSaas, BackendQuSprout
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CNOT, H, Measure

ak = "rCTaXydwknt2O0vhQscjkWt6aK6kBYJ0Rm8N8ByK"
sk = "rPy8B8Y4ebqw766wrls6VIdafwUNKYNknWhMDB0F7YqX7SiZqFDasXKwgq7a4Ano5DZnfHQgEmmDUke4IbpPrYjQcIQUtK7NB9hi5hj00LuCE8GUYLKeHmDdTvbvqRgg"
# use BackendQuSaas
be = BackendQuSaas(ak, sk)
circuit = QCircuit(be)
# circuit = QCircuit(BackendQuSprout())
qr = circuit.allocate(2)

# apply gate
H * qr[0]
CNOT * (qr[0], qr[1])

circuit.print()
circuit.draw(line_length=300)

print(circuit.get_prob(0))
print(circuit.get_probs())
print(circuit.get_statevector())

Measure * qr[0]
Measure * qr[1]

res = circuit.run(shots=100)

print(res.get_measure())
print(res.get_counts())
print(res.excute_info())


# generate random number
rands = be.get_rand(21, 2)
print(rands)
