import json
import sys
from subprocess import Popen, PIPE, TimeoutExpired


def read_ide_parameter(file_name: str):
    """Read the JSON file provided by QuIDE.

    Args:
        file_name: JSON file.

    Returns:
        file: Python dictionary.
    """
    # TODO:读取保内数据文件 pkgutil.get_data()
    with open(file=file_name, encoding="utf-8") as f:
        file = json.load(f)
    return file


def algorithm(file: dict):
    """According to IDE_Parameter.json generate quantum circuit.

    Args:
        file: Json file tell how to make quantum circuit.

    Returns:
        File path to store the quantum circuit.
    """
    with open(file="algorithm.py", mode="w", encoding="utf-8") as f:
        f.write("import json\n")
        f.write("from collections import defaultdict\n")
        f.write("from qutrunk.circuit import QCircuit\n")
        gt = "from qutrunk.circuit.gates import " + ",".join(file["gates"])
        f.write(gt + "\n")
        f.write("qc = QCircuit()\n")

        f.write(f"qr = qc.allocate({file['qubit']})\n")
        for c in file["cmd"]:
            f.write(c + "\n")

        f.write("outcome = qc.get_probs()\n")
        # f.write("print(qc.get_probs())\n")
        f.write("qc.run()\n")
        # f.write("print([int(q) for q in qr])\n")

        f.write("Tree = lambda: defaultdict(Tree)\n")
        f.write("tree = Tree()\n")
        c = [
            "for _, item in enumerate(outcome):\n",
            "    i = str(bin(item['idx']))[2:]\n",
            "    tree['probs'][i]['probability'] = item['prob']\n",
            "    tree['probs'][i]['angle'] = 0\n",
        ]
        f.writelines(c)
        # TODO:后期删除
        f.write("print(json.dumps(tree))\n")
        result_path = file["result_path"]
        write_file = [
            f"with open(file=r'{result_path}', mode='w', encoding='utf-8') as f:\n",
            "    file = json.dump(json.dumps(tree), f)\n",
        ]
        f.writelines(write_file)


def run_algorithm(params_file: str):
    """Run Python scripts using Tornado."""
    # 使用Tornado运行Python脚本
    # 1 读取IDE提供的参数JSON文件
    file = read_ide_parameter(params_file)

    # 2 根据ide_parameter.json生成量子算法
    algorithm(file)

    # 3 调用量子算法Python脚本
    proc = Popen([sys.executable, "algorithm.py"], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    try:
        outs, errs = proc.communicate(timeout=15)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    if errs:
        print("run error in algorithm.py")
        return
    print(outs.decode())
    print("sucess")


if __name__ == "__main__":
    run_algorithm("ide_parameter.json")
