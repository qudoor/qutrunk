import json
from collections import defaultdict
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H,CNOT,Measure
qc = QCircuit()
qr = qc.allocate(2)
H | qr[0]
CNOT | (qr[0], qr[1])
Measure | qr[0]
Measure | qr[1]
outcome = qc.get_probs()
qc.run()
Tree = lambda: defaultdict(Tree)
tree = Tree()
for _, item in enumerate(outcome):
    i = str(bin(item['idx']))[2:]
    tree['probs'][i]['probability'] = item['prob']
    tree['probs'][i]['angle'] = 0
print(json.dumps(tree))
with open(file=r'result.json', mode='w', encoding='utf-8') as f:
    file = json.dump(json.dumps(tree), f)
