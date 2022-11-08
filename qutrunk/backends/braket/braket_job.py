"""AWS Braket job."""
from typing import Optional, Union

from braket.aws import AwsQuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask

from qutrunk.backends import Backend


class MeasureResult:
    def __init__(self, id=0, value=0):
        # TODO: id是关键字，不建议使用
        self.id = id
        self.value = value


class OutcomeResult:
    def __init__(self, bit_str="", count=0):
        self.bitstr = bit_str
        self.count = count


class Result:
    def __init__(self):
        self.measureSet = []
        self.outcomeSet = []


class AWSBraketJob:
    """AWSBraketJob."""

    def __init__(
            self,
            job_id: str,
            backend: Backend,
            task: Union[LocalQuantumTask, AwsQuantumTask],
            **metadata: Optional[dict]
    ):
        """AWSBraketJob for local execution of circuits.

        Args:
            job_id: id of the job
            backend: Local simulator
            tasks: Executed tasks
            **metadata:
        """
        self._job_id = job_id
        self._backend = backend
        self._task = task
        self._metadata = metadata

    @property
    def shots(self) -> int:
        """Return the number of shots.

        Returns:
            shots: int with the number of shots.
        """
        return (
            self._metadata["shots"] if "shots" in self._metadata else 0
        )

    def result(self) -> Result:
        # todo bit str big end?
        counter = self._task.result().measurement_counts
        res = Result()
        for bs, cnt in dict(counter).items():
            res.outcomeSet.append(OutcomeResult(bs, cnt))
        return res

    def cancel(self):
        self._task.cancel()

    def status(self):
        return self._task.state()
