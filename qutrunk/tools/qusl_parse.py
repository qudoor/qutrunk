"""QuSL parse."""

import os
import json

from qutrunk.exceptions import QuTrunkError


BASE_DIR = os.getcwd()
py_file = BASE_DIR + "/qutrunk/tools/qusl.py"


# TODO: have some problems.
def parse(qusl_file):
    """Parse QuSL file and generate quantum circuit.

    Args:
        file: The input QuSL file(json format).

    Returns:
        QCircuit object.
    """
    qusl_data = None
    with open(file=qusl_file, mode="r", encoding="utf-8") as f:
        qusl_data = json.load(f)
    if qusl_data is None:
        raise ValueError("Empty QuSL file")

    qubits = qusl_data["meta"]["qubits"]
    qusl_code_lines = qusl_data["code"]

    if int(qubits) <= 0:
        raise ValueError("Invalid qubits")

    with open(file=py_file, mode="w") as fw:
        fw.write("from qutrunk.circuit import QCircuit\n")
        fw.write("from qutrunk.circuit.gates import *\n")
        fw.write("from qutrunk.circuit.ops import *\n\n")
        fw.write("def generate_circuit():\n")
        fw.write("\tcircuit = QCircuit()\n")
        fw.write(f"\tq = circuit.allocate({qubits})\n")
        for line in qusl_code_lines:
            fw.write("\t" + line)
        fw.write("\treturn circuit\n\n")
        fw.write("generate_circuit()")


def qusl_to_circuit(file):
    """Parse QuSL file and generate quantum circuit.

    Args:
        file: The input QuSL file(json format).

    Returns:
        QCircuit object.
    """
    try:
        parse(file)
        from qutrunk.tools.qusl import generate_circuit

        circuit = generate_circuit()
        return circuit

    except Exception as e:
        raise QuTrunkError("parse QuSL error(err:", e + ")")
