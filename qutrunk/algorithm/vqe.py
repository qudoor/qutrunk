import math
import time
from typing import Union

import numpy as np
from scipy.optimize import minimize

from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Ry, Rz, CNOT, Matrix, PauliCoeffs
from qutrunk.exceptions import QuTrunkError


class VQE:
    def __init__(self, qb_cnt, depth=2, backend=None, optimizer="slsqp", entangle="linear", max_iter=1000,
                 callback=None):
        """

        Args:
            qb_cnt: qubit count
            depth: depth for Ry/Rz
            backend: backend for circuit to run
            optimizer: optimizer supported by scipy, refer to scipy.optimize.minimize
            entangle: "linear"/"full", entangle method considering CNOT
            callback: callback method for each iter
            max_iter: maximum iteration
        """
        # construct ansatz
        self.ansatz_fac = AnsatzFac(backend, qb_cnt, depth, entangle)

        self._param_names = [f"theta-{str(i)}" for i in range(2 * depth * qb_cnt)]
        # init params
        self._init_params = np.random.uniform(math.pi * -2, math.pi * 2, 2 * qb_cnt * depth)

        self.optimizer = optimizer

        self._max_iter = max_iter

        self.callback = callback

    def compute_minimum_eigenvalue(self, operator: Union[Matrix, PauliCoeffs]):
        """
        Compute minimum eigenvalue,

        Args:
            operator: qubit operator

        Returns:
            VQEResult

        """
        operator = self._check_operator(operator)

        # define func for optimizer
        def energy_evaluation(params):
            """
            Evaluate the energy at given parameters for the ansatz

            Args:
                params: parameters optimized by scipy.optimize.minimize

            Returns:
                Energy of the hamiltonian

            """
            # apply ansatz
            ans = self.ansatz_fac.ansatz()
            params_dict = dict(zip(self._param_names, params))
            cir = ans.bind_parameters(params_dict)

            res = operator(cir)

            if self.callback:
                self.callback(params, res)
            return res

        # call minimizer
        start = time.time()
        opt_res = minimize(energy_evaluation, self._init_params, method=self.optimizer,
                           options={'maxiter': self._max_iter})
        opt_time = time.time() - start

        # return opt_res.x
        eigen_s = self.eigenstate(opt_res.x)
        return VQEResult(opt_res.x, opt_res.fun, eigen_s, opt_time)

    def print_ansatz(self):
        """Print ansatz circuit """

        cir = self.ansatz_fac.ansatz()
        cir.draw(line_length=1000)

    def eigenstate(self, params):
        """
        calculate eigenstate with params for ansatz

        Args:
            params: parameters for ansatz

        Returns:
            eigenstate

        """
        cir = self.ansatz_fac.ansatz()
        cir = cir.bind_parameters(dict(zip(self._param_names, params)))
        return cir.get_statevector()

    def _check_operator(self, op):
        """
        Check op type,and qubits for op and ansatz match

        Args:
            op:

        Returns:
            unified operator

        """
        if isinstance(op, PauliCoeffs):
            return lambda circuit: circuit.expval_pauli_sum(op)
        elif isinstance(op, Matrix):
            if not op.check_matrix_format(self.ansatz_fac.qb_cnt):
                raise QuTrunkError("Matrix shape doesn't comply with VQE ansatz qubit count")

            def operator(circuit):
                i_state = circuit.get_statevector()
                op * circuit.qreg
                o_state = circuit.get_statevector()
                res = (o_state.conj().T @ i_state).real
                return res

            return operator
        else:
            raise QuTrunkError("supported operator: PauliCoeffs/Matrix")


class AnsatzFac:
    def __init__(self, backend, qb_cnt, depth, entangle):
        """

        Args:
            backend: circuit backend
            qb_cnt: qubit count
            depth: depth of Ry/Rz
            entangle: entanglement method, "linear"/"full"
        """
        self.backend = backend
        self.qb_cnt = qb_cnt
        self.depth = depth
        self.entangle = entangle

    def ansatz(self):
        """
        Construct ansatz circuit

        Returns:
            ansatz circuit

        """
        cir = QCircuit(self.backend)
        qreg = cir.allocate(self.qb_cnt)
        params = [f"theta-{i}" for i in range(2 * self.qb_cnt * self.depth)]

        params = cir.create_parameters(params)
        for d in range(0, 2 * self.depth, 2):
            for i in range(self.qb_cnt):
                Ry(params[d * self.qb_cnt + i]) * qreg[i]
                Rz(params[(d + 1) * self.qb_cnt + i]) * qreg[i]
            if self.entangle == "linear":
                for i in range(self.qb_cnt - 1):
                    CNOT * (qreg[i], qreg[i + 1])
            elif self.entangle == "full":
                for i in range(self.qb_cnt - 1):
                    for j in range(i + 1, self.qb_cnt):
                        CNOT * (qreg[i], qreg[j])
            else:
                raise QuTrunkError("supported entanglement: linear/full")

        return cir


class VQEResult:
    def __init__(self, params, eigenvalue, eigenstate, opt_time):
        """

        Args:
            params: final parameters for ansatz
            eigenvalue:  eigenvalue
            eigenstate: eigenstate
            opt_time: optimizer_time
        """
        self.optimal_params = params
        self.eigenvalue = eigenvalue
        self.eigenstate = eigenstate
        self.optimizer_time = opt_time
