# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
auto_scheduler Workload Definition.
"""
import pickle
from typing import Any, Sequence

from ruamel.yaml import YAML, yaml_object
from tvm.auto_scheduler.search_task import SearchTask

from ....tune.job import Job
from ....workload import Workload
from ..util import get_canonical_tvm_target_str


@yaml_object(YAML())
class AutoSchedulerWorkload(Workload):
    """The workload for an op.
    A workload can be used to created an AutoScheduler task for tuning.
    """

    def __init__(self):
        super(AutoSchedulerWorkload, self).__init__()
        self.workload_key: str = ""
        self.task_pickle: bytes = b""
        self.dag_repr: str = ""

    @classmethod
    def from_task(cls, task: SearchTask) -> "AutoSchedulerWorkload":
        """Create a workload from an AutoScheduler task.

        Parameters
        ----------
        task: SearchTask
            The AutoScheduler task for the workload.

        Returns
        -------
        workload: Workload
            The initialized workload.
        """

        workload = cls()

        assert task.target is not None
        workload.workload_key = str(task.workload_key)
        workload.target = get_canonical_tvm_target_str(task.target, task)
        workload.task_pickle = pickle.dumps(task)
        workload.dag_repr = repr(task.compute_dag)
        return workload

    def to_task(self) -> SearchTask:
        """Create an AutoScheduler task from this workload.

        Returns
        -------
        task: SearchTask
            Return the created task, or raise RuntimeError if failed.
        """
        # Try to create task.
        try:
            task = pickle.loads(self.task_pickle)
        except Exception as err:  # pylint: disable=broad-except
            raise RuntimeError(
                "Failed to create the task for workload {0}: {1}".format(str(self), str(err))
            )

        return task

    def to_job(self) -> Job:
        """Create a job to tune this workload.

        Returns
        -------
        job: Job
            The created job.
        """
        from .job import AutoSchedulerJob  # Avoid circular import dependency.

        return AutoSchedulerJob(self)

    def get_workload_key(self) -> str:
        """Get the primary key of this workload in DB.

        Returns
        -------
        key: str
            The primary key of this workload to index the records in DB.
        """
        return self.workload_key

    def mutate(self, rules: Any) -> Sequence["Workload"]:
        """auto_scheduler task mutation is not supported yet.

        Parameters
        ----------
        workload: Workload
            The workload to be mutated.

        rules: Any
            The mutation rules that can be customized.

        Returns
        -------
        workloads: Sequence[Workload]
            The mutated workloads.
        """
        raise NotImplementedError

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, AutoSchedulerWorkload)
        for key in ["workload_key", "target"]:
            if getattr(self, key) != getattr(other, key):
                return getattr(self, key) < getattr(other, key)
        return False

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self):
        return "%s(workload_key=%r,target=%r,dag=%s)" % (
            self.__class__.__name__,
            self.workload_key,
            self.target,
            self.dag_repr,
        )
