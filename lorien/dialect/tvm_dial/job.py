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
TVM Job definition module.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple

from ...configs import append_config_parser
from ...logger import get_logger
from ...tune.job import Job, JobConfigs
from .util import get_canonical_tvm_target_str, get_tvm_build_config, is_cover

log = get_logger("TVMTuneJob")


class TuneMetadata:
    """Metadata for a tuning process."""

    def __init__(self):
        self.max_thrpt = 0  # Maximum throughput (GFLOP/s).
        self.trial_count = 0
        self.failed_count = 0  # The number of trails with error number != 0.


class TVMJobConfigs(JobConfigs):
    """AutoTVM job configurations."""

    def __init__(self, configs: argparse.Namespace):
        """Initialize a job configuration.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configuration of tuner.
        """
        super(TVMJobConfigs, self).__init__(configs)
        self.tvm_build_config: Dict[str, str] = get_tvm_build_config()

    def check_tvm_build_config(self) -> bool:
        """Check if the TVM build config on this machine matches the expected one.

        Returns
        -------
        match: bool
            Return True if matcheing.
        """
        if not self.tvm_build_config:
            # Always match if job configs do not specify the expectation.
            return True

        this_config = get_tvm_build_config()
        for key, val in this_config.items():
            if key not in self.tvm_build_config or self.tvm_build_config[key] != val:
                log.warning(
                    "TVM build config mismatch: expected %s, but here is %s",
                    str(self.tvm_build_config),
                    str(this_config),
                )
                return False
        return True

    def localize(self, target: str, **kwargs):
        """Localize options on worker.

        Parameters
        ----------
        target: str
            The target string.

        **kwargs
            The kwargs of job configuration for updating.
        """
        raise NotImplementedError


class TVMJob(Job):
    """A tuning job including a workload as well as tuning related configurations."""

    def is_target_compatible(self, target: str) -> bool:
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
        this_target = get_canonical_tvm_target_str(self.workload.target)
        that_target = get_canonical_tvm_target_str(target)
        return is_cover(this_target, that_target)

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
        """Tune the job with the given configuration and update the result.
        If the commit options are provided, then this function also in charge of
        committing the tuning results, or the job manager will commit the result, otherwise.
        """
        raise NotImplementedError


@append_config_parser("top.tune", "TVM tuning options")
def append_tune_config() -> List[Tuple[List[str], Dict[str, Any]]]:
    """Define the command line interface for TVM tuning.

    Returns
    -------
    actions: List[Tuple[List[str], Dict[str, Any]]]
        The AutoTVM tuning configs.
    """
    return [
        (
            ["-n", "--ntrial"],
            {"default": 3000, "type": int, "help": "Number of tuning trials for each workload"},
        ),
        (["--test"], {"default": 5, "type": int, "help": "Number of tests in one measurement"}),
        (
            ["--repeat"],
            {"default": 1, "type": int, "help": "Number of measurements for one config"},
        ),
        (["--min"], {"default": 1000, "type": int, "help": "Minimum repeat time (ms)"}),
    ]


@append_config_parser("top.rpc-client", "TVM options for RPC")
def append_rpc_config() -> List[Tuple[List[str], Dict[str, Any]]]:
    """Define the command line interface for TVM RPC.

    Returns
    -------
    actions: List[Tuple[List[str], Dict[str, Any]]]
        The AutoTVM tuning configs.
    """
    return [
        (["--device"], {"type": str, "help": "Device name owned by this host"}),
        (["--runner-port"], {"type": int, "help": "The port to the TVM runner"}),
    ]
