import os
from enum import Enum
from typing import Optional

from qutrunk.backends.backend import Backend
from qutrunk.tools.read_qubox import get_qubox_setting
from .rpcclient import QuSproutApiServer
from qutrunk.thrift.qusproutdata import ttypes as qusproutdata
from qutrunk.backends.result import MeasureQubit, MeasureQubits, MeasureResult


class BackendQuSprout(Backend):
    """
    QuSprout: quamtum circuit simulator, provide multi-threaded OMP, multi node parallel MPI, GPU hardware acceleration.
    To use qusprout, make sure the network is connected and the service IP and Port are set correctly.

    Args:
        run_mode: cpu: calculation use single cpu; \
                  cpu_mpi: parallel calculation using multiple cpu; \ 
                  gpu: calculation use single gpu.

    Example:
        .. code-block:: python

            from qutrunk.circuit import QCircuit
            from qutrunk.backends import BackendQuSprout
            from qutrunk.circuit.gates import H, CNOT, Measure

            # use BackendQuSprout
            qc = QCircuit(backend=BackendQuSprout())
            qr = qc.allocate(2)

            # apply gate
            H * qr[0]
            CNOT * (qr[0], qr[1])
            Measure * qr[0]
            Measure * qr[1]

            # run circuit
            res = qc.run(shots=100)

            # print result
            print(res.get_counts())
    """

    def __init__(self, ip: Optional[str] = None, port: Optional[int] = None, run_mode: str = "cpu"):
        super().__init__()
        self.circuit = None
        self.run_mode = run_mode
        box_config = get_qubox_setting()

        if ip and port:
            _ip = ip
            _port = port
        elif ip is None and port is None:
            _ip = box_config.get("ip")
            _port = port=box_config.get("port")
        else:
            if ip is None:
                print("Please specify ip in BackendQuSprout()!")
            else:
                print("Please specify port in BackendQuSprout()!")
            os._exit(1)

        self._api_server = QuSproutApiServer(_ip, _port)

    def get_prob(self, index):
        """Get the probability of a state-vector at an index in the full state vector.

        Args:
            index: Index in state vector of probability amplitudes.

        Returns:
            The probability of target index.
        """
        res, elapsed = self._api_server.get_prob(index)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_probs(self, qubits):
        """Get all probabilities of circuit.

        Returns:
            An array contains all probabilities of circuit.
        """
        res, elapsed = self._api_server.get_probs(qubits)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_statevector(self):
        """Get the current state vector of probability amplitudes for a set of qubits.

        Returns:
            Array contains all amplitudes of state vector
        """
        res, elapsed = self._api_server.get_statevector()
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def send_circuit(self, circuit, final=False):
        """Send the quantum circuit to qusprout backend.

        Args:
            circuit: Quantum circuit to send.
            final: True if quantum circuit finish, default False, \
            when final==True The backend program will release the computing resources.
        """
        cmds = []
        start = circuit.cmd_cursor
        stop = len(circuit.cmds)

        for idx in range(start, stop):
            cmd = circuit.cmds[idx]

            cmdex = None
            if cmd.cmdex is not None:
                _amp = None
                _mat = None
                if cmd.cmdex.amp is not None:
                    _amp = qusproutdata.Amplitude(
                        cmd.cmdex.amp.reals,
                        cmd.cmdex.amp.imags,
                        cmd.cmdex.amp.startind,
                        cmd.cmdex.amp.numamps,
                    )
                if cmd.cmdex.mat is not None:
                    _mat = qusproutdata.Matrix(
                        cmd.cmdex.mat.reals, cmd.cmdex.mat.imags, cmd.cmdex.mat.unitary
                    )
                cmdex = qusproutdata.Cmdex(amp=_amp, mat=_mat)

            c = qusproutdata.Cmd(
                str(cmd.gate),
                cmd.targets,
                cmd.controls,
                cmd.rotation,
                cmd.qasm(),
                cmd.inverse,
                cmdex,
            )
            cmds.append(c)

        circuit.forward(stop - start)

        exectype = qusproutdata.ExecCmdType.ExecTypeDefault
        if self.run_mode == "cpu_mpi":
            exectype = qusproutdata.ExecCmdType.ExecTypeCpuMpi
        elif self.run_mode == "gpu":
            exectype = qusproutdata.ExecCmdType.ExecTypeGpuSingle
        else:
            exectype = qusproutdata.ExecCmdType.ExecTypeCpuSingle
            
        # 服务端初始化
        if start == 0:
            res, elapsed = self._api_server.init(
                circuit.num_qubits, circuit.density, exectype
            )
            if self.circuit.counter:
                self.circuit.counter.acc_run_time(elapsed)

        if len(cmds) == 0 and (not final):
            return

        # 发送至服务
        res, elapsed = self._api_server.send_circuit(qusproutdata.Circuit(cmds), final)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)

    def run(self, shots=1):
        """Run quantum circuit.

        Args:
            shots: Circuit run times, for sampling, default: 1.

        Returns:
            The Result object contain circuit running outcome.
        """
        res, elapsed = self._api_server.run(shots)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
            self.circuit.counter.finish()

        result = MeasureResult()
        for meas in res.measures:
            meas_temp = MeasureQubits()
            for mea in meas.measure:
                mea_temp = MeasureQubit(mea.idx, mea.value)
                meas_temp.measure.append(mea_temp)
            result.measures.append(meas_temp)
        """
        1 必须释放连接，不然其它连接无法连上服务端
        2 不能放在__del__中，因为对象释放不代表析构函数会及时调用
        """
        self._api_server.close()

        return result

    def get_expec_pauli_prod(self, pauli_prod_list):
        """Computes the expected value of a product of Pauli operators.

        Args:
            pauli_prod_list: A list contains the indices of the target qubits,\
            the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) to apply to the corresponding qubits.

        Returns:
            The expected value of a product of Pauli operators.
        """
        puali_list = []
        for temp in pauli_prod_list:
            puali_list.append(
                qusproutdata.PauliProdInfo(temp["oper_type"], temp["target"])
            )

        res, elapsed = self._api_server.get_expec_pauli_prod(puali_list)
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    def get_expec_pauli_sum(self, oper_type_list, term_coeff_list):
        """Computes the expected value of a sum of products of Pauli operators.

        Args:
            oper_type_list: A list of the Pauli codes (0=PAULI_I, 1=PAULI_X, 2=PAULI_Y, 3=PAULI_Z) \
                of all Paulis involved in the products of terms. A Pauli must be specified \
                for each qubit in the register, in every term of the sum.
            term_coeff_list: The coefficients of each term in the sum of Pauli products.

        Returns:
            The expected value of a sum of products of Pauli operators.
        """
        res, elapsed = self._api_server.get_expec_pauli_sum(
            oper_type_list, term_coeff_list
        )
        if self.circuit.counter:
            self.circuit.counter.acc_run_time(elapsed)
        return res

    @property
    def name(self):
        return "BackendQuSprout-" + self.run_mode
