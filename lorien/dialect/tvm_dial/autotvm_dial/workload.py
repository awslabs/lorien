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
AutoTVM Workload Definition.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ruamel.yaml import YAML, yaml_object
from tvm import autotvm
from tvm.autotvm.task.task import Task

from ....tune.job import Job
from ....util import deep_tuple_to_list, dump_to_yaml, load_from_yaml
from ....workload import Workload
from ..util import get_canonical_tvm_target_str


@yaml_object(YAML())
class AutoTVMWorkload(Workload):
    """The workload for an op.
    A workload can be used to created an AutoTVM task for tuning.
    """

    def __init__(self):
        super(AutoTVMWorkload, self).__init__()
        self.task_name = "unknown"
        self.args = []
        self._primary_key: Optional[str] = None

    @classmethod
    def from_task(cls, task: Task) -> "AutoTVMWorkload":
        """Create a workload from an AutoTVM task.

        Parameters
        ----------
        task: Task
            The AutoTVM task for the workload.

        Returns
        -------
        workload: Workload
            The initialized workload.
        """

        workload = cls()

        if task.target is None:
            raise RuntimeError(
                "Failed to generate workload from AutoTVM task %s: No target specified" % str(task)
            )
        workload.task_name = task.name
        workload.target = get_canonical_tvm_target_str(task.target, task)
        workload.args = deep_tuple_to_list(task.args)

        return workload

    def to_task(self) -> Task:
        """Create an AutoTVM task from this workload.
        Note that task may not be created if this workload violates the rules defined in
        the schedule. For example, a schedule may only work for the conv2d with 4n channel
        numbers. In this case, the task cannot be created if this workload has 18 channels.

        Returns
        -------
        task: Task
            Return the created task, or raise RuntimeError if failed.
        """

        task_args: List[Any] = [
            tuple([arg[0], tuple(arg[1]), arg[2]])
            if isinstance(arg, list) and arg[0] == "TENSOR"
            else arg
            for arg in self.args
        ]

        # Try to create task.
        try:
            task = autotvm.task.create(self.task_name, tuple(task_args), self.target)
        except Exception as err:  # pylint: disable=broad-except
            # We cannot expect the exceptions from schedules in libraries like TOPI so
            # we have to catch broad exceptions.
            raise RuntimeError(
                "Failed to create task for workload {0}: {1}".format(str(self), str(err))
            )

        return task

    def to_job(self) -> Job:
        """Create a job to tune this workload.

        Returns
        -------
        job: Job
            The created job.
        """
        from .job import AutoTVMJob  # Avoid circular import dependency.

        return AutoTVMJob(self)

    def get_workload_key(self) -> str:
        """Get the primary key of this workload in DB.

        Returns
        -------
        key: str
            The primary key of this workload to index the records in DB.
        """
        if self._primary_key is not None:
            return self._primary_key

        # Task name
        self._primary_key = self.task_name

        # Arguments
        for arg in self.args:
            serialized_arg = (
                str(arg)
                .replace("(", "_")
                .replace(")", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace(" ", "")
                .replace("'", "")
                .replace(",", "_")
            )
            self._primary_key += "#{}".format(serialized_arg)

        return self._primary_key

    def mutate(self, rules: Dict[Tuple[int, ...], str]) -> Sequence["Workload"]:
        """Mutate workload arguments with the given rules.

        Parameters
        ----------
        workload: Workload
            The workload to be mutated.

        rules: Dict[Tuple[int, ...], str]
            Mapping from argument index to a mutation rule. For example, a rule to mutate
            conv2d batch size would be (0, 1, 0): "[1, 2, 4, 8, 16]".

        Returns
        -------
        workloads: Sequence[AutoTVMWorkload]
            The mutated workloads.
        """

        def arg_getter(wkl: "AutoTVMWorkload", idxs: Tuple[int, ...]):
            """A helper function to get the value of a nested list of arguments."""
            args = wkl.args
            for idx in idxs[:-1]:
                if not isinstance(args, (list, tuple)):
                    raise RuntimeError(
                        "Index %s does not match workload argument %s" % (str(idxs), str(wkl.args))
                    )
                args = args[idx]
            return args[idxs[-1]]

        def mutator(wkl: "AutoTVMWorkload", idxs: Tuple[int, ...], cands: List[Union[int, str]]):
            wkl_yaml = dump_to_yaml(wkl)
            assert wkl_yaml is not None
            new_wkls: List["AutoTVMWorkload"] = []
            for cand in cands:
                new_wkl = load_from_yaml(wkl_yaml)
                curr_arg = new_wkl.args
                for idx in idxs[:-1]:
                    # Should have checked when getting the argument value.
                    assert isinstance(curr_arg, (list, tuple))
                    curr_arg = curr_arg[idx]

                curr_arg = list(curr_arg) if isinstance(curr_arg, tuple) else curr_arg
                curr_arg[idxs[-1]] = cand
                new_wkls.append(new_wkl)
            return new_wkls

        workloads: Sequence["AutoTVMWorkload"] = [self]
        for idxs, rule in rules.items():
            sub_workloads: List["AutoTVMWorkload"] = []
            for wkl in workloads:
                try:
                    var_map = {"v": arg_getter(wkl, idxs)}
                    cands = eval(rule, None, var_map)  # pylint: disable=eval-used
                except Exception as err:  # pylint: disable=board-except
                    raise RuntimeError("Failed to interprete rule %s: %s" % (rule, str(err)))
                sub_workloads += mutator(wkl, idxs, cands)
            workloads = sub_workloads
        return workloads

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, AutoTVMWorkload)
        for key in ["task_name", "target"]:
            if getattr(self, key) != getattr(other, key):
                return getattr(self, key) < getattr(other, key)

        if len(self.args) != len(other.args):
            return len(self.args) < len(other.args)
        return self.args < other.args

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self):
        return "%s(task_name=%r,target=%r,args=%r)" % (
            self.__class__.__name__,
            self.task_name,
            self.target,
            self.args,
        )
