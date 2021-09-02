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
Job definition module.
"""
import argparse
import datetime
from enum import Enum
from typing import Any, Dict, Optional

from ruamel.yaml import YAML, yaml_object

from .result import TuneResult
from ..util import dump_to_yaml, load_from_yaml
from ..workload import Workload


@yaml_object(YAML())
class JobState(Enum):
    """The state of a tuning job."""

    WAITING = 0
    TUNING = 1
    FINISHED = 2

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(u"!JobState", "%s" % node._value_)

    @classmethod
    def from_yaml(cls, _, node):
        return cls(int(node.value))


class JobEventLogger:
    """Log job change events to a file."""

    def __init__(self, file_name: str):
        """Initialize the logger.

        Parameters
        ----------
        file_name: str
            The logging file name.
        """
        self.filep = open(file_name, "a")

    def log(self, job: "Job"):
        self.filep.write("%s\t%s" % (str(datetime.datetime.now()), dump_to_yaml(job)))
        self.filep.flush()


@yaml_object(YAML())
class JobConfigs:
    """Maintain all configurations related to job tuning, including tuning options,
    meausre options, and commit options. The configuration is shared by all jobs in
    the same dialect.
    """

    def __init__(self, configs: argparse.Namespace):
        """Initialize a job configuration.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configuration of tuner.
        """
        self.tune_options: Dict[str, Any] = {}
        self.measure_options: Dict[str, Any] = {}
        self.commit_options: Optional[Dict[str, Any]] = {
            "db": load_from_yaml(configs.db),
            "table-name": configs.commit_table_name,
            "table-arn": None,
            "nbest": configs.commit_nbest,
            "commit-workload": configs.commit_workload,
            "commit-log": configs.commit_log_to,
        }

    def localize(self, target: str, **kwargs):
        """Localize job configs on worker for the specific target.
        This is intended for each dialect to customize.

        Parameters
        ----------
        target: str
            The target string.
        """


@yaml_object(YAML())
class Job:
    """The base class of a tuning job including a workload as well as various configurations.
    If the commit options are provided, then this function also in charge of committing
    the tuning results, or the job manager will commit the result, otherwise.
    """

    def __init__(self, workload: Workload):
        """Initialize a job.

        Parameters
        ----------
        workload: Workload
            The workload for this job.
        """
        self.workload: Workload = workload
        self.result: TuneResult = TuneResult()
        self._event_logger: Optional[JobEventLogger] = None
        self._state = JobState.WAITING
        self._metadata: Dict[str, Any] = {}

        # We cannot use the object hash because it inlcudes the job states.
        # Instead, we use stateless hash that only includes workload and tune_config,
        # so that we can use it to match jobs when recovering the job manager.
        self._stateless_hash: Optional[int] = None

    @staticmethod
    def create_job_configs(configs: argparse.Namespace) -> JobConfigs:
        """Create a JobConfigs. See `JobConfigs`.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configuration of tuner.

        Returns
        -------
        job_configs: JobConfigs
            The job configurations.
        """
        raise NotImplementedError

    def tune(
        self,
        tune_options: Dict[str, Any],
        measure_options: Dict[str, Any],
        commit_options: Optional[Dict[str, Any]] = None,
    ):
        """Tune the job with the given configurations. See `JobConfigs`."""
        raise NotImplementedError

    def is_target_compatible(self, target: str) -> bool:
        # pylint: disable=unused-argument
        """Check if the taret is compatible to this job.

        Parameters
        ----------
        target: str
            The target string

        Returns
        -------
        compatible: bool
            Whether the target is compatible to this job.
        """
        return True

    @property
    def stateless_hash(self):
        """Return the stateless hash of this job.

        Parameters
        ----------
        stateless_hash: int
            The stateless hash.
        """
        if self._stateless_hash is None:
            self._stateless_hash = self.workload.hash_sha2()
        return self._stateless_hash

    def trace(self, event_logger: JobEventLogger):
        """Start tracing the state/metadata changes of this job.

        Parameters
        ----------
        event_logger: JobEventLogger
            The job event logger.
        """
        self._event_logger = event_logger

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: JobState):
        self._state = new_state
        if self._event_logger is not None:
            self._event_logger.log(self)

    def set_metadata(self, key: str, val: Any):
        """Set a metadata entry.

        Parameters
        ----------
        key: str
            The metadata key.

        val: Any
            The metadata value.
        """
        self._metadata[key] = val
        if self._event_logger is not None:
            self._event_logger.log(self)

    def get_metadata(self, key: str) -> Any:
        """Get the value of a metadata entry.

        Parameters
        ----------
        key: str
            The metadata key.

        Returns
        -------
        entry: Any
            The entry value.
        """
        if key not in self._metadata:
            raise RuntimeError("Metadata key %s not found in %s" % (key, str(self)))
        return self._metadata[key]

    def __getstate__(self):
        """Customize the serialization method to reset tracer attributes."""
        state = self.__dict__.copy()
        state["_event_logger"] = None
        return state

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self):
        return "%s(state=%r,workload=%r,result=%r,metadata=%r)" % (
            self.__class__.__name__,
            self._state,
            self.workload,
            self.result,
            self._metadata,
        )
