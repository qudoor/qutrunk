import cmath
import math

import numpy as np

from qutrunk.circuit.gates import *


def test_gate_matrix():
    angle = np.pi / 2
    cos = math.cos(angle / 2)
    sin = math.sin(angle / 2)
    exp_m = np.exp(-1j * angle)
    exp_p = np.exp(1j * angle)
    assert np.array_equal(
        H.matrix, 1.0 / cmath.sqrt(2.0) * np.matrix([[1, 1], [1, -1]])
    )
    assert np.array_equal(
        P(angle).matrix,
        np.matrix([[cmath.exp(1j * angle), 0], [0, cmath.exp(1j * angle)]]),
    )
    assert np.array_equal(
        R(angle, angle).matrix,
        np.matrix([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]]),
    )
    assert np.array_equal(
        Rx(angle).matrix,
        np.matrix(
            [
                [math.cos(0.5 * angle), -1j * math.sin(0.5 * angle)],
                [-1j * math.sin(0.5 * angle), math.cos(0.5 * angle)],
            ]
        ),
    )
    assert np.array_equal(
        Ry(angle).matrix,
        np.matrix(
            [
                [math.cos(0.5 * angle), -math.sin(0.5 * angle)],
                [math.sin(0.5 * angle), math.cos(0.5 * angle)],
            ]
        ),
    )
    assert np.array_equal(
        Rz(angle).matrix,
        np.matrix(
            [
                [cmath.exp(-0.5 * 1j * angle), 0],
                [0, cmath.exp(0.5 * 1j * angle)],
            ]
        ),
    )
    assert np.array_equal(S.matrix, np.matrix([[1, 0], [0, 1j]]))
    assert np.array_equal(Sdg.matrix, np.matrix([[1, 0], [0, 1j]]).getH())
    assert np.array_equal(
        T.matrix, np.matrix([[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]])
    )
    assert np.array_equal(
        Tdg.matrix, np.matrix([[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]]).getH()
    )
    assert np.array_equal(X.matrix, np.matrix([[0, 1], [1, 0]]))
    assert np.array_equal(Y.matrix, np.matrix([[0, -1j], [1j, 0]]))
    assert np.array_equal(Z.matrix, np.matrix([[1, 0], [0, -1]]))
    assert np.array_equal(NOT.matrix, X.matrix)
    assert np.array_equal(
        Swap.matrix, np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )
    assert np.array_equal(
        SqrtSwap.matrix,
        np.matrix(
            [
                [1, 0, 0, 0],
                [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                [0, 0, 0, 1],
            ]
        ),
    )
    assert np.array_equal(
        SqrtX.matrix, 0.5 * np.matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    )
    assert np.array_equal(
        Rxx(angle).matrix,
        np.matrix(
            [
                [cmath.cos(0.5 * angle), 0, 0, -1j * cmath.sin(0.5 * angle)],
                [0, cmath.cos(0.5 * angle), -1j * cmath.sin(0.5 * angle), 0],
                [0, -1j * cmath.sin(0.5 * angle), cmath.cos(0.5 * angle), 0],
                [-1j * cmath.sin(0.5 * angle), 0, 0, cmath.cos(0.5 * angle)],
            ]
        ),
    )
    assert np.array_equal(
        Ryy(angle).matrix,
        np.matrix(
            [
                [cmath.cos(0.5 * angle), 0, 0, 1j * cmath.sin(0.5 * angle)],
                [0, cmath.cos(0.5 * angle), -1j * cmath.sin(0.5 * angle), 0],
                [0, -1j * cmath.sin(0.5 * angle), cmath.cos(0.5 * angle), 0],
                [1j * cmath.sin(0.5 * angle), 0, 0, cmath.cos(0.5 * angle)],
            ]
        ),
    )
    assert np.array_equal(
        Rzz(angle).matrix,
        np.matrix(
            [
                [cmath.exp(-0.5 * 1j * angle), 0, 0, 0],
                [0, cmath.exp(0.5 * 1j * angle), 0, 0],
                [0, 0, cmath.exp(0.5 * 1j * angle), 0],
                [0, 0, 0, cmath.exp(-0.5 * 1j * angle)],
            ]
        ),
    )
    assert np.array_equal(
        U1(angle).matrix, np.matrix([[1, 0], [0, np.exp(1j * angle)]])
    )
    isqrt2 = 1 / np.sqrt(2)
    assert np.array_equal(
        U2(angle, angle).matrix,
        np.matrix(
            [
                [isqrt2, -np.exp(1j * angle) * isqrt2],
                [np.exp(1j * angle) * isqrt2, np.exp(1j * (angle + angle)) * isqrt2],
            ]
        ),
    )
    assert np.array_equal(
        U3(angle, angle, angle).matrix,
        np.array(
            [
                [np.cos(angle / 2), -np.exp(1j * angle) * np.sin(angle / 2)],
                [
                    np.exp(1j * angle) * np.sin(angle / 2),
                    np.exp(1j * (angle + angle)) * np.cos(angle / 2),
                ],
            ]
        ),
    )
    eith = np.exp(1j * float(angle))
    assert np.array_equal(
        CP(angle).matrix,
        np.array(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]])),
    )
    half_theta = angle / 2
    cos = np.cos(half_theta)
    sin = np.sin(half_theta)
    isin = 1j * np.sin(half_theta)
    assert np.array_equal(
        CRx(angle).matrix,
        np.array(
            np.array(
                [[cos, 0, -isin, 0], [0, 1, 0, 0], [-isin, 0, cos, 0], [0, 0, 0, 1]]
            )
        ),
    )
    assert np.array_equal(
        CRy(angle).matrix,
        np.array(
            np.array([[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]])
        ),
    )
    arg = 1j * half_theta
    assert np.array_equal(
        CRz(angle).matrix,
        np.array(
            np.array(
                [
                    [np.exp(-arg), 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(arg), 0],
                    [0, 0, 0, 1],
                ]
            )
        ),
    )
    assert np.array_equal(
        CX.matrix,
        np.array(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
    )
    assert np.array_equal(
        CY.matrix,
        np.array(np.array([[0, 0, -1j, 0], [0, 1, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1]])),
    )
    assert np.array_equal(
        CZ.matrix,
        np.array(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])),
    )
