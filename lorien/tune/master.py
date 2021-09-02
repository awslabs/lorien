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
The tuning master.
"""
import argparse
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type

from ..configs import create_config_parser, register_config_parser
from ..logger import get_logger
from ..util import download_s3_file, load_from_yaml
from ..workload import Workload
from .job_manager import JOB_MANAGER_TABLE, Job, JobManagerBase
from .result import TuneResult

log = get_logger("Master")


def run_job_manager(
    manager_cls: Type[JobManagerBase], packed_args: Dict[str, Any]
) -> List[TuneResult]:
    """Tune workloads on a worker.

    Parameters
    ----------
    manager_cls: Type[JobManagerBase]
        The job manager class definition.

    packed_args: Dict[str, Any]
        The packed job manager arguments.

    Returns
    -------
    results: List[TuneResult]
        The result can be either the performance or error message.
    """

    try:
        manager = manager_cls(**packed_args)  # type: ignore
    except RuntimeError as err:
        log.warning(str(err))
        return []

    return manager.tune()


def get_manager_configs(
    configs: argparse.Namespace,
) -> Tuple[Optional[Type[JobManagerBase]], Optional[Any]]:
    """Validate the job manager configurations and fetch them.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configurations.

    Returns
    -------
    manager_cls, data: Tuple[Optional[Type[JobManagerBase]], Optional[Any]]
        The job manager class and configuration.
    """

    manager_cls: Optional[Type[JobManagerBase]] = None
    data: Optional[Any] = None
    for mname, mcls in JOB_MANAGER_TABLE.items():
        assert hasattr(
            configs, mname
        ), "Job manager {} has not registered the same name config".format(mname)

        manager_details = getattr(configs, mname)
        if manager_details:
            if data is not None:
                raise RuntimeError("Only one type of managers is allowed for a tuning proces")

            # Process manager arguments
            manager_cls = mcls
            data = load_from_yaml(manager_details)

    return (manager_cls, data)


def get_target(configs: argparse.Namespace) -> Optional[str]:
    """Parse worker info and get the target for creating table and S3 buckets.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configurations.

    Returns
    -------
    target: Optional[str]
        The target string, or None if the required configuration is missing.
    """
    _, data = get_manager_configs(configs)
    if data is None:
        return None

    target: Optional[str] = None
    if isinstance(data, dict):
        target = data["target"]
    elif isinstance(data, str):
        target = data

    assert target is not None, "Worker config is expected in dict or str, but got %s" % type(data)
    return target


def parse_manager_info(
    target: str,
    configs: argparse.Namespace,
    jobs: List[Job],
) -> Tuple[Type[JobManagerBase], Dict[str, Any]]:
    """Parse and formulate worker info.

    Parameters
    ----------
    target: str
        The target string.

    configs: argparse.Namespace
        The system configurations.

    jobs: List[Job]
        A complete list of jobs to be tuned.

    Returns
    -------
    worker_info: List[Tuple[Type[JobManagerBase], Dict[str, Any]]]
        A list of (manager class, manager info).
    """
    manager_cls, _ = get_manager_configs(configs)
    assert manager_cls is not None

    return (
        manager_cls,
        {
            "target": target,
            "jobs": jobs,
            "configs": configs,
        },
    )


def run(configs: argparse.Namespace):
    """Distribute workloads to job managers and track the progress.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configuration of tuner.
    """
    # Load workloads.
    workloads: List[Workload] = []
    if configs.workload:
        workloads = list({load_from_yaml(data, Workload) for data in configs.workload})
    log.info("Loaded %d workloads", len(workloads))

    # Generate jobs from workloads.
    jobs: List[Job] = [wkl.to_job() for wkl in workloads]

    # Load jobs.
    if configs.job:
        for job_str_or_path in configs.job:
            if job_str_or_path.startswith("s3://"):
                # The job file is in a S3 bucket.
                temp_job_file = "/tmp/lorien_download_job_{}.yaml".format(
                    next(tempfile._get_candidate_names())  # type: ignore
                )
                err_msg = download_s3_file(job_str_or_path, temp_job_file, delete=True)
                if err_msg:
                    raise RuntimeError(err_msg)
                with open(temp_job_file, "r") as filep:
                    job_str = filep.read()
                os.remove(temp_job_file)
            else:
                job_str = job_str_or_path
            jobs.append(load_from_yaml(job_str, Job))
        log.info("Loaded %d jobs", len(jobs))

    target = get_target(configs)
    if target is None:
        log.warning("Skip tuning due to no tuning config specified")
        return []
    log.info("Target: %s", target)

    # Process job managers.
    manager_info: Tuple[Type[JobManagerBase], Dict[str, Any]] = parse_manager_info(
        target, configs, jobs
    )

    # Run job managers.
    results = run_job_manager(*manager_info)
    log.info("Finished tuning")
    return results


@register_config_parser("top.tune")
def define_config() -> argparse.ArgumentParser:
    """Define the command line interface for tuning.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("Distributed Tuning")
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        required=False,
        help="A workload in YAML format. It is recommanded to list workload in "
        "a YAML file and use @workloads.yaml to specify them",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        required=False,
        help="A string of job in YAML format, or a S3 file path to the job file."
        "This argument is used internally.",
    )
    parser.add_argument(
        "--job-configs",
        default=None,
        required=False,
        help="The job config in YAML format. This argument is used internally.",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        required=False,
        help="The job state trace file. If presents, the job manager will replay trace logs to "
        "resume job states. Otherwise the tuning starts over and a new trace file will be created.",
    )
    commit_options = parser.add_argument_group("Result committing options")
    commit_options.add_argument(
        "--db", default="{ }", help="DynamoDB client options in YAML format"
    )
    commit_options.add_argument(
        "--commit-log-to",
        default=None,
        help="Commit full tuning log to the given S3 bucket. "
        "If folder is specified (e.g., bucket-name/folder-name), "
        "then tuning logs of all targets will be uploaded to the same folder. "
        "When unset, full tuning logs will not be stoed.",
    )
    commit_options.add_argument(
        "--commit-table-name",
        default="lorien",
        help="The table name in DynamoDB for the best records to commit",
    )
    commit_options.add_argument(
        "--commit-nbest",
        default=20,
        type=int,
        help="The best config of a number of data layouts we will commit. "
        " Only applicable to the workload with inferrable layout such as "
        "conv2d and depthwise conv2d on x86 and Intel graphics",
    )
    commit_options.add_argument(
        "--commit-workload",
        default=False,
        action="store_true",
        help="Commit the tuned workload to workload table in the DB",
    )
    parser.set_defaults(entry=run)
    return parser
