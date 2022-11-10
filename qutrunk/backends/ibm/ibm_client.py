"""Run quantum program on IBM QE cloud platform."""
import signal
import time
import uuid

import requests
from requests import Session
from requests.compat import urljoin

# TODO: move to config file
_AUTH_API_URL = "https://auth.quantum-computing.ibm.com/api/users/loginWithToken"
_API_URL = "https://api.quantum-computing.ibm.com/api/"

# TODO: call to get the API version automatically
CLIENT_APPLICATION = "ibmqprovider/0.4.4"


# TODO: exclude in API document.
class IBMQ(Session):
    """Manage a session between QuTrunk and the IBMQ web API."""

    def __init__(self, **kwargs):
        """Initialize a session with the IBM QE's APIs."""
        super().__init__(**kwargs)
        self.backends = {}
        self.timeout = 5.0
        self.hooks["response"].append(
            lambda r, *i_args, **i_kwargs: r.raise_for_status()
        )

    def get_list_devices(self, verbose=False):
        """Get the list of available IBM backends with their properties.

        Args:
            verbose (bool): Print the returned dictionary if True

        Returns:
            Backends(dict) dictionary by name device, containing the qubit size 'nq',
            the coupling map 'coupling_map' as well as the device version 'version'
        """
        list_device_url = "Network/ibm-q/Groups/open/Projects/main/devices/v/1"
        argument = {"allow_redirects": True, "timeout": (self.timeout, None)}
        request = super().get(urljoin(_API_URL, list_device_url), **argument)
        # 反序列化
        r_json = request.json()
        self.backends = {}
        for obj in r_json:
            self.backends[obj["backend_name"]] = {
                "nq": obj["n_qubits"],
                "coupling_map": obj["coupling_map"],
                "version": obj["backend_version"],
            }

        if verbose:
            print("- List of IBMQ devices available:")
            print(self.backends)
        return self.backends

    def authenticate(self, token=None):
        """Authenticate with IBM's Web API.

        Args:
            token (str): IBM quantum experience user API token.
        """
        if token is None:
            raise Exception("please input the IBM QE token")
        # write token into headers
        self.headers.update({"X-Qx-Client-Application": CLIENT_APPLICATION})
        args = {
            "data": None,
            "json": {"apiToken": token},
            "timeout": (self.timeout, None),
        }
        request = super().post(_AUTH_API_URL, **args)
        r_json = request.json()
        self.params.update({"access_token": r_json["id"]})

    def is_online(self, device):
        """Device is available."""
        return device in self.backends

    def can_run_experiment(self, info=None, device=""):
        """Whether the resources satisfy to run quantum programs."""
        nb_qubit_max = self.backends[device]["nq"]
        nb_qubit_needed = info["nq"]
        return nb_qubit_needed <= nb_qubit_max, nb_qubit_max, nb_qubit_needed

    def run(self, info=None, device=""):
        """Run the quantum code to the IBMQ machine.

         Args:
             info (dict): Dictionary sent by the backend containing the code to run.
             device (str): Name of the ibm device to use.

        Returns:
            Execution Id.
        """
        # 1 get the device URL and job id
        json_step1 = {
            "data": None,
            "json": {
                "backend": {"name": device},
                "allowObjectStorage": True,
                "shareLevel": "none",
            },
            "timeout": (self.timeout, None),
        }
        request = super().post(
            urljoin(_API_URL, "Network/ibm-q/Groups/open/Projects/main/Jobs"),
            **json_step1,
        )
        request.raise_for_status()
        r_json = request.json()
        print("r_json==", r_json)
        upload_url = r_json["objectStorageInfo"]["uploadUrl"]
        execution_id = r_json["id"]

        # 2 send circuit
        n_classical_reg = info["nq"]
        n_qubits = n_classical_reg
        # instruction
        instructions = info["json"]
        maxcredit = info["maxCredits"]
        c_label = [["c", i] for i in range(n_classical_reg)]
        q_label = [["q", i] for i in range(n_qubits)]

        instruction_str = str(instructions).replace("'", '"')
        data = '{"qobj_id": "' + str(uuid.uuid4()) + '", '
        data += '"header": {"backend_name": "' + device + '", '
        data += '"backend_version": "' + self.backends[device]["version"] + '"}, '
        data += '"config": {"shots": ' + str(info["shots"]) + ", "
        data += '"max_credits": ' + str(maxcredit) + ', "memory": false, '
        data += '"parameter_binds": [], "memory_slots": ' + str(n_classical_reg)
        data += ', "n_qubits": ' + str(n_qubits) + '}, "schema_version": "1.2.0", '
        data += '"type": "QASM", "experiments": [{"config": '
        data += '{"n_qubits": ' + str(n_qubits) + ", "
        data += '"memory_slots": ' + str(n_classical_reg) + "}, "
        data += '"header": {"qubit_labels": ' + str(q_label).replace("'", '"') + ", "
        data += '"n_qubits": ' + str(n_classical_reg) + ", "
        data += '"qreg_sizes": [["q", ' + str(n_qubits) + "]], "
        data += '"clbit_labels": ' + str(c_label).replace("'", '"') + ", "
        data += '"memory_slots": ' + str(n_classical_reg) + ", "
        data += '"creg_sizes": [["c", ' + str(n_classical_reg) + "]], "
        data += (
            '"name": "circuit0", "global_phase": 0}, "instructions": '
            + instruction_str
            + "}]}"
        )

        json_step2 = {
            "data": data,
            "params": {"access_token": None},
            "timeout": (5.0, None),
        }
        super().put(upload_url, **json_step2)

        # 3 upload
        json_step3 = {"data": None, "json": None, "timeout": (self.timeout, None)}

        upload_data_url = urljoin(
            _API_URL,
            "Network/ibm-q/Groups/open/Projects/main/Jobs/"
            + str(execution_id)
            + "/jobDataUploaded",
        )
        super().post(upload_data_url, **json_step3)

        return execution_id

    def get_result(
        self, device, execution_id, num_retries=3000, interval=1, verbose=False
    ):
        """Get running results"""
        job_status_url = "Network/ibm-q/Groups/open/Projects/main/Jobs/" + execution_id
        if verbose:
            print(f"Waiting for results. [Job ID: {execution_id}]")
            original_sigint_handler = signal.getsignal(signal.SIGINT)
            print("signal=", original_sigint_handler)

            def _handle_sigint_during_get_result(*_):
                raise Exception(
                    "Interrupted. The ID of your submitted job is {}.".format(
                        execution_id
                    )
                )

            try:
                signal.signal(signal.SIGINT, _handle_sigint_during_get_result)

                for retries in range(num_retries):
                    # 1: wait job running
                    json_step5 = {
                        "allow_redirects": True,
                        "timeout": (self.timeout, None),
                    }
                    request = super().get(
                        urljoin(_API_URL, job_status_url), **json_step5
                    )
                    r_json = request.json()
                    acceptable_status = ["VALIDATING", "VALIDATED", "RUNNING"]

                    # 2 get running results
                    if r_json["status"] == "COMPLETED":
                        json_step6 = {
                            "allow_redirects": True,
                            "timeout": (self.timeout, None),
                        }
                        request = super().get(
                            urljoin(_API_URL, job_status_url + "/resultDownloadUrl"),
                            **json_step6,
                        )
                        r_json = request.json()

                        json_step7 = {
                            "allow_redirects": True,
                            "params": {"access_token": None},
                            "timeout": (self.timeout, None),
                        }
                        request = super().get(r_json["url"], **json_step7)
                        r_json = request.json()
                        result = r_json["results"][0]

                        json_step8 = {
                            "data": None,
                            "json": None,
                            "timeout": (5.0, None),
                        }
                        request = super().post(
                            urljoin(_API_URL, job_status_url + "/resultDownloaded"),
                            **json_step8,
                        )
                        r_json = request.json()  # {'terminated': True}

                        return result

                    if r_json["status"] not in acceptable_status:
                        raise Exception(
                            f"Error while running the code. Last status: {r_json['status']}."
                        )
                    time.sleep(interval)
                    if self.is_online(device) and retries % 60 == 0:
                        self.get_list_devices()
                        if not self.is_online(device):
                            raise Exception(
                                f"Device went offline. The ID of your submitted job is {execution_id}."
                            )

            finally:
                if original_sigint_handler is not None:
                    signal.signal(signal.SIGINT, original_sigint_handler)

            raise Exception(f"Timeout. The ID of your submitted job is {execution_id}.")


def show_devices(token=None, verbose=False):
    """Access the list of available devices.

    Args:
        token (str): IBM quantum experience user API token.
        verbose (bool): If True, additional information is printed.

    Returns:
        List (list) of available devices and their properties.
    """
    ibmq_session = IBMQ()
    ibmq_session.authenticate(token=token)
    return ibmq_session.get_list_devices(verbose=verbose)


def send(
    info,
    device="ibmq_qasm_simulator",
    token=None,
    shots=None,
    num_retries=3000,
    interval=1,
    verbose=False,
):
    """Send QASM through the IBM API and runs the quantum circuit.

    Args:
        info(dict): Contains representation of the circuit to run.
        device (str): name of the ibm device. Simulator chosen by default.
        token (str): IBM quantum experience user API token.
        shots (int): Number of runs of the same circuit to collect statistics.
        verbose (bool): If True, additional information is printed, such as measurement statistics. Otherwise, the
            backend simply registers one measurement result (same behavior as the project Simulator).

    Returns:
        Result (dict) form the IBMQ server.

    """
    try:
        ibmq_session = IBMQ()
        if shots is not None:
            info["shots"] = shots
        if verbose:
            print("- Authenticating...")
            if token is not None:
                print("user API token: " + token)
        ibmq_session.authenticate(token)

        # check if the device is online
        ibmq_session.get_list_devices(verbose)
        online = ibmq_session.is_online(device)
        if not online:
            print(
                "The device is offline. Use the simulator instead or try again later."
            )
            raise Exception("Device is offline.")

        # check if the device has enough qubit to run the code
        runnable, qmax, qneeded = ibmq_session.can_run_experiment(info, device)
        if not runnable:
            print(
                f"The device is too small ({qmax} qubits available) for the code "
                + f"requested({qneeded} qubits needed) Try to look for another "
                + "device with more qubits"
            )
            raise Exception("Device is too small.")
        if verbose:
            print(f"- Running code: {info}")
        execution_id = ibmq_session.run(info, device)
        if verbose:
            print("- Waiting for results...")
        res = ibmq_session.get_result(
            device,
            execution_id,
            num_retries=num_retries,
            interval=interval,
            verbose=verbose,
        )
        if verbose:
            print("- Done.")
        return res
    except requests.exceptions.HTTPError as err:
        print("- There was an error running your code:")
        print(err)
    except requests.exceptions.RequestException as err:
        print("- Looks like something is wrong with server:")
        print(err)
    except KeyError as err:
        print("- Failed to parse response:")
        print(err)
    return None
