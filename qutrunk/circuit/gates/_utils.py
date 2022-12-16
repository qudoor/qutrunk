"""
This module contains utility functions for circuits.
"""

import numpy as np

def _compute_control_matrix(base_mat, num_ctrl_qubits):
    """
    This function computes the controlled unitary with :math:`n` control qubits
    and :math:`m` target qubits.

    .. math::

    C^n(U) := I^{\otimes (n+k)} +\left(|1\rangle\langle1|\right)^{\otimes n}\otimes (U-I) = \begin{bmatrix} I_{2^{n+k}-2^k} & 0\\ 0 & U \end{bmatrix}.
    """
    num_target = int(np.log2(base_mat.shape[0]))
    vec1 = np.array([0, 1])
    mat = np.outer(vec1, vec1)
    for i in range(num_ctrl_qubits - 1):
        mat = np.kron(mat, np.outer(vec1, vec1))
    matrix = np.identity(2 ** (num_ctrl_qubits + num_target)) \
        + np.kron(mat, base_mat - np.identity(base_mat.shape[0]))

    return matrix


from qutrunk.circuit.gates import X

print(_compute_control_matrix(X.matrix, 2))

    
