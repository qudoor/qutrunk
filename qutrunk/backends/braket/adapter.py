"""Util function for backend."""
from typing import Dict, Callable

import numpy as np
from braket.circuits import Instruction, Circuit, result_types, gates
from numpy import pi

import qutrunk.circuit.gates as i_gates
from .exception import QuTrunkBraketException


def u3(theta, phi, lam):
    """U3Gate decomposition

    Args:
        theta: theta
        phi: phi
        lam: lam

    Returns:
        list of gates representing u3 gate
    """
    return [
        gates.Rz(lam),
        gates.Rx(pi / 2),
        gates.Rz(theta),
        gates.Rx(-pi / 2),
        gates.Rz(phi),
    ]


qutrunk_gate_names_to_braket_gates: Dict[str, Callable] = {
    "u1": lambda lam: [gates.Rz(lam)],
    "u2": lambda phi, lam: [gates.Rz(lam), gates.Ry(pi / 2), gates.Rz(phi)],
    "u3": lambda theta, phi, lam: u3(theta, phi, lam),
    "p": lambda angle: [gates.PhaseShift(angle)],
    "cp": lambda angle: [gates.CPhaseShift(angle)],
    "cx": lambda: [gates.CNot()],
    "x": lambda: [gates.X()],
    "y": lambda: [gates.Y()],
    "z": lambda: [gates.Z()],
    "t": lambda: [gates.T()],
    "tdg": lambda: [gates.Ti()],
    "s": lambda: [gates.S()],
    "sdg": lambda: [gates.Si()],
    "sx": lambda: [gates.V()],
    "sxdg": lambda: [gates.Vi()],
    "swap": lambda: [gates.Swap()],
    "r": lambda theta, phi: u3(theta, phi - pi / 2, -phi + pi / 2),
    "rx": lambda angle: [gates.Rx(angle)],
    "ry": lambda angle: [gates.Ry(angle)],
    "rz": lambda angle: [gates.Rz(angle)],
    "rzz": lambda angle: [gates.ZZ(angle)],
    "id": lambda: [gates.I()],
    "h": lambda: [gates.H()],
    "cy": lambda: [gates.CY()],
    "cz": lambda: [gates.CZ()],
    "ccx": lambda: [gates.CCNot()],
    "cswap": lambda: [gates.CSwap()],
    "rxx": lambda angle: [gates.XX(angle)],
    "ryy": lambda angle: [gates.YY(angle)],
    "ecr": lambda: [gates.ECR()],

    "x1": lambda: [gates.Rx(pi / 2)],
    "y1": lambda: [gates.Ry(pi / 2)],
    "z1": lambda: [gates.Rz(pi / 2)],
    "iswap": lambda: [gates.Unitary(np.array(i_gates.iSwap.matrix), "iSwap")],
    "sqrtx": lambda: [gates.V()],
    "cr": lambda theta: [gates.Unitary(np.array(i_gates.CR(theta).matrix), "CR")],
    "crx": lambda theta: [gates.Unitary(np.array(i_gates.CRx(theta).matrix), "CRx")],
    "cry": lambda theta: [gates.Unitary(np.array(i_gates.CRy(theta).matrix), "CRy")],
    "crz": lambda theta: [gates.Unitary(np.array(i_gates.CRz(theta).matrix), "CRz")],
    "cu": lambda theta, phi, lam, gamma: [gates.Unitary(np.array(i_gates.CU(theta, phi, lam, gamma).matrix), "CU")],
    "cu1": lambda theta: [gates.Unitary(np.array(i_gates.CU1(theta).matrix), "CU1")],
    "cu3": lambda theta, phi, lam: [gates.Unitary(np.array(i_gates.CU3(theta, phi, lam).matrix), "CU3")],
    "ch": lambda: [gates.Unitary(np.array(i_gates.CH.matrix), "CH")],
    "csqrtx": lambda: [gates.CV()],
    "sqrtxdg": lambda: [gates.Vi()],
}


def convert_qutrunk_to_braket_circuit(circuit) -> Circuit:
    """Return a Braket quantum circuit from a QuTrunk quantum circuit.
     Args:
            circuit (QCircuit): QuTrunk Quantum Cricuit

    Returns:
        Circuit: Braket circuit
    """
    quantum_circuit = Circuit()
    for qutrunk_cmd in circuit.cmds:
        name = str(qutrunk_cmd.gate).lower()
        if name == "measure":
            # TODO: change Probability result type for Sample for proper functioning # pylint:disable=fixme
            quantum_circuit.add_result_type(
                result_types.Probability(
                    target=[
                        qutrunk_cmd.targets[0],
                        qutrunk_cmd.targets[0]
                    ]
                )
            )
        elif name == "barrier":
            pass
        else:
            params = []
            if qutrunk_cmd.rotation:
                params = qutrunk_cmd.rotation

            ctr_cnt = len(qutrunk_cmd.controls)
            if name == "mcx":
                if ctr_cnt == 1:
                    gate_list = [gates.CNot()]
                elif ctr_cnt == 2:
                    gate_list = [gates.CCNot()]
                else:
                    raise QuTrunkBraketException(
                        f"{name}({ctr_cnt}):mcx with more than 2 control bit isn't supported by BRAKET")
            elif name == "mcz":
                if ctr_cnt == 1:
                    gate_list = [gates.CZ()]
                else:
                    raise QuTrunkBraketException(
                        f"{name}({ctr_cnt}):mcz with more than 1 control bit isn't supported by BRAKET")
            else:
                gate_list = qutrunk_gate_names_to_braket_gates[name](*params)

            if not gate_list:
                raise QuTrunkBraketException(f"{name} isn't supported by BRAKET")

            for gate in gate_list:
                instruction = Instruction(
                    # Getting the index from the bit mapping
                    operator=gate,
                    target=qutrunk_cmd.controls + qutrunk_cmd.targets,
                )
                quantum_circuit += instruction
    return quantum_circuit
