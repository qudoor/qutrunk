"""AWS Braket job."""
from typing import Optional, Union

from braket.aws import AwsQuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask

from qutrunk.backends import Backend
from qutrunk.backends.result import MeasureCount, MeasureResult

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
        return self._metadata["shots"] if "shots" in self._metadata else 0

    def result(self) -> MeasureResult:
        """Convert braket result to qutrunk measurement result

        Returns:
            qutrunk measurement result.

        """
        # todo bit str big end?
        counter = self._task.result().measurement_counts
        res = MeasureResult()
        for bs, cnt in dict(counter).items():
            res.measure_counts.append(MeasureCount(bs, cnt))
        return res

    def cancel(self):
        """Cancel AWS Braket task."""
        self._task.cancel()

    def status(self):
        """Task status."""
        return self._task.state()
