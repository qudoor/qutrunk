from qutrunk.circuit.gates import (
    MCX, CYGate, MCZ, SwapGate, HGate, SGate, SdgGate, SqrtXGate,
    TGate, TdgGate, P, U1, U2, U3, U, CU, CU1, CU3,
    XGate, YGate, ZGate, Rx, Ry, Rz, Rxx, Rzz, CP, CRx, CRy, CRz, R, Ryy, SqrtSwapGate, iSwap, CR, X1Gate,
    Y1Gate, Z1Gate, IGate, SqrtXdgGate, CHGate, CSqrtXGate, CSwapGate
)

# qasm的量子门映射到qutrunk的量子门
qutrunk_standard_gate = {
    # standard gate from qasm
    "u1": U1,
    "u2": U2,
    "u3": U3,
    "u": U,
    "p": P,
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "t": TGate,
    "tdg": TdgGate,
    "s": SGate,
    "sdg": SdgGate,
    "sx": SqrtXGate,
    "sxdg": SqrtXdgGate,
    "swap": SwapGate,
    "rx": Rx,
    "rxx": Rxx,
    "ry": Ry,
    "rz": Rz,
    "rzz": Rzz,
    "id": IGate,
    "h": HGate,
    "cx": MCX,
    "cy": CYGate,
    "cz": MCZ,
    "ch": CHGate,
    "crx": CRx,
    "cry": CRy,
    "crz": CRz,
    "csx": CSqrtXGate,
    "cu1": CU1,
    "cp": CP,
    "cu": CU,
    "cu3": CU3,
    # "ccx": MCX(2), # qutrunk implement by MCX
    "cswap": CSwapGate,
    # "delay": Delay,

    # qutrunk only gate
    "r": R,
    "ryy": Ryy,
    "sqrtswap": SqrtSwapGate,
    "iswap": iSwap,
    "cr": CR,
    "x1": X1Gate,
    "y1": Y1Gate,
    "z1": Z1Gate,
    "mcx": MCX,
    "mcz": MCZ
}
