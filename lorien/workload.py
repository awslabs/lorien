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
Workload Definition Module.
"""
import hashlib
import uuid
from typing import Any, Sequence

from ruamel.yaml import YAML, yaml_object


@yaml_object(YAML())
class Workload:
    """The workload base class that can be used to create a tuning task."""

    def __init__(self):
        self.target = "unknown"

    @classmethod
    def from_task(cls, task: Any) -> "Workload":
        """Create a workload from a tuning task.

        Parameters
        ----------
        task: Any
            The tuning task for the workload.

        Returns
        -------
        workload: Workload
            The initialized workload.
        """
        raise NotImplementedError

    def to_task(self) -> Any:
        """Create a tuning task from this workload.

        Returns
        -------
        task: Any
            Return the created task, or raise RuntimeError if failed.
        """
        raise NotImplementedError

    def to_job(self):
        """Create a job to tune this workload.

        Returns
        -------
        job: Job
            The created job.
        """
        raise NotImplementedError

    def mutate(self, rules: Any) -> Sequence["Workload"]:
        """Mutate workload arguments with the given rules.

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
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def hash_sha2(self) -> str:
        """Hash this workload with SHA256 algorithm to be a unique 64-byte string.

        Returns
        -------
        code: str
            A 64-byte string.
        """
        sha_obj = hashlib.sha256(str(self).encode("utf-8"))
        return sha_obj.hexdigest()

    def get_log_file_name(self) -> str:
        """Log file name is encoded as <workload SHA2 code>-<random code>.json

        Parameters
        ----------
        workload: Workload
            The target workload.

        Returns
        -------
        log_file_name: str
            The generated log file name.
        """
        return "{0}-{1}.json".format(self.hash_sha2(), str(uuid.uuid4())[:5])

    def get_workload_key(self) -> str:
        """Get the primary key of this workload in DB.

        Returns
        -------
        key: str
            The primary key of this workload to index the records in DB.
        """
        return self.hash_sha2()

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and str(self) == str(other)

    def __str__(self) -> str:
        return repr(self)
