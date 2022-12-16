"""QuSL parse."""

import os
import json
import random
import importlib

from qutrunk.exceptions import QuTrunkError

# TODO: 明确指定生成目录，将当前目录加入搜索路径
def _parse(file):
    """Parse QuSL file and generate quantum circuit.

    Args:
        file: The input QuSL file(json format).

    Returns:
        QCircuit object.
    """

    with open(file, mode="r", encoding="utf-8") as f:
        qusl_data = json.load(f)

    if qusl_data is None:
        raise ValueError("Empty QuSL file.")

    qubits = qusl_data["meta"]["qubits"]
    qusl_code_lines = qusl_data["code"]

    if int(qubits) <= 0:
        raise ValueError("Invalid qubits.")

    name = random.randint(1, 10000)
    filename = f"{name}.py"

    with open(file=filename, mode="w", encoding="utf-8") as fw:
        fw.write("from qutrunk.circuit import QCircuit\n")
        fw.write("from qutrunk.circuit.gates import *\n")
        fw.write("def generate_circuit():\n")
        fw.write("\tcircuit = QCircuit()\n")
        fw.write(f"\tq = circuit.allocate({qubits})\n")
        for line in qusl_code_lines:
            fw.write("\t" + line)
        fw.write("\treturn circuit\n\n")

    return str(name)


def qusl_to_circuit(file):
    """Parse QuSL file and generate quantum circuit.

    Args:

        file: The input QuSL file(json format).

    Returns:
        QCircuit object.
    """

    try:
        # import generate_circuit function.
        filename = _parse(file)
        g = importlib.import_module(filename)
        # generate circuit object.
        circuit = g.generate_circuit()
        # delete the temp file
        os.remove(f"{filename}.py")
    except Exception as e:
        raise QuTrunkError(e)
    else:
        return circuit

