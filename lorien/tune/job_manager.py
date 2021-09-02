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
The worker module.
"""
import argparse
import os
import tempfile
import time
from abc import abstractmethod
from copy import deepcopy
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import boto3
from tqdm import tqdm

from ..configs import append_config_parser
from ..database.table import create_table
from ..logger import get_logger
from ..util import dump_to_yaml, load_from_yaml, upload_s3_file
from .job import Job, JobConfigs, JobEventLogger, JobState
from .result import TuneErrorCode, TuneResult
from .rpc import RPCClient, launch_server

log = get_logger("JobManager")
JOB_MANAGER_TABLE: Dict[str, Type["JobManagerBase"]] = {}


def register_job_manager(name: str, format_help: str) -> Callable:
    """Register job manager.

    Parameters
    ----------
    name: str
        The job manager name.

    format_help: str
        The job manager specific config format description.

    Returns
    -------
    reg: Callable
        A callable function for registration.
    """

    def _do_reg(manager: Type[JobManagerBase]):
        if name in JOB_MANAGER_TABLE:
            raise RuntimeError("%s has been registered" % name)

        # Register job manager
        JOB_MANAGER_TABLE[name] = manager

        # Register job manager config
        cfg_func = lambda: [
            (
                ["--{}".format(name)],
                {
                    "help": "Job manager description in YAML format. "
                    'Format: "<TVM target string>: {}"'.format(format_help),
                },
            )
        ]
        append_config_parser("top.tune", "{} job manager options".format(name))(cfg_func)
        return manager

    return _do_reg


class JobManagerBase:
    """The base class of job manager."""

    def __init__(
        self,
        target: str,
        jobs: List[Job],
        configs: argparse.Namespace,
    ):
        """Initialize a job manager.

        Parameters
        ----------
        target: str
            The target string.

        jobs: List[Job]
            The job list.

        configs: argparse.Namespace
            The system configuration.
        """
        self.target = target

        job_cls: Optional[Type[Job]] = None

        # Map from job.stateless_hash to job object.
        self.job_set: Dict[int, Job] = {}

        # Select jobs for this target.
        for job in jobs:
            # Filter out other targets.
            if not job.is_target_compatible(target):
                continue

            # Update the target.
            clone_job = deepcopy(job)
            clone_job.workload.target = target

            self.job_set[clone_job.stateless_hash] = clone_job

            if job_cls is None:
                job_cls = type(clone_job)

        if not self.job_set:
            raise RuntimeError(
                "Terminate one %s due to no jobs for %s" % (type(self).__name__, target)
            )

        # Create the job configs. If the serialized job config is specified,
        # then load it directly to override other CLI configs.
        assert job_cls is not None
        self.job_configs: JobConfigs = (
            load_from_yaml(configs.job_configs)
            if configs.job_configs is not None
            else job_cls.create_job_configs(configs)
        )

        # Create a table in DB if needed and get its ARN.
        assert self.job_configs.commit_options is not None
        self.job_configs.commit_options["table-arn"] = create_table(
            self.job_configs.commit_options["table-name"], **self.job_configs.commit_options["db"]
        )

        # Resume job states from the trace file.
        if configs.trace_file is not None:
            self.trace_file = configs.trace_file
        else:
            self.trace_file = "lorien-tune-%s.trace" % str(time.time()).replace(".", "")
        log.info("Tuning state is maintained in %s", self.trace_file)
        self.replay_trace()
        self.resume_job_states()

        # Resume job states and put WAITING jobs to the waiting queue.
        self.waiting_jobs = [job for job in self.job_set.values() if job.state == JobState.WAITING]

        # Start tracing job changes.
        self.job_event_logger = JobEventLogger(self.trace_file)
        for job in self.job_set.values():
            job.trace(self.job_event_logger)

    def num_jobs(self) -> int:
        """Return the total number of jobs to be tuned by this manager."""
        return len(self.job_set)

    @abstractmethod
    def desc(self) -> str:
        """Return a description shown at the beginning of progress bar while tuning."""
        raise NotImplementedError

    @abstractmethod
    def tune_impl(self, progress: tqdm):
        """Workload tuning implementation. The tuning results are directly stored in
        self.job_n_results.

        Parameters
        ----------
        progress: tqdm
            The formulated progress bar to be updated progressively.
        """
        raise NotImplementedError

    @abstractmethod
    def resume_job_states(self):
        """Resume the jobs that were being tuned when the state was dumped."""
        raise NotImplementedError

    def tune(self) -> List[TuneResult]:
        """Tune workloads on the servers via RPC.

        Returns
        -------
        tune_results: List[TuneResult]
            The result can be either the absolute performance,
            the speedup over the last performance, or the error message.
        """
        log.info("Tuning target %s", self.target)

        progress = tqdm(
            total=self.num_jobs(),
            desc=self.desc(),
            bar_format="{desc}{percentage:3.0f}%|{bar:50}{r_bar}",
        )

        self.tune_impl(progress)
        print("\n")  # Sometimes tqdm miss the last newline.

        # Aggregate results
        results = []
        for job in self.job_set.values():
            results.append(job.result)

        return results

    def replay_trace(self):
        """Update the current state from the trace file."""
        if not os.path.exists(self.trace_file):
            return

        with open(self.trace_file, "r") as filep:
            for trace in filep:
                try:
                    _, job_str = trace.replace("\n", "").split("\t")
                except ValueError:
                    log.warning("Invalid trace: %s", trace)
                    continue

                try:
                    loaded_job = load_from_yaml(job_str, Job)
                except RuntimeError as err:
                    log.warning("Invalid job format: %s", str(err))
                    continue

                if loaded_job.stateless_hash in self.job_set:
                    # Update state of an existing job.
                    curr_job = self.job_set[loaded_job.stateless_hash]
                    curr_job.state = loaded_job.state
                    curr_job._metadata = dict(loaded_job._metadata)
                    curr_job.result = deepcopy(loaded_job.result)
                else:
                    # Recover missing jobs.
                    self.job_set[loaded_job.stateless_hash] = loaded_job

        log.info("Successfully resumed job manager state from %s", self.trace_file)


@register_job_manager("local", "no additional config is required")
class LocalJobManager(JobManagerBase):
    """Local job manager class."""

    def desc(self) -> str:
        return "Local"

    def resume_job_states(self):
        """Resume the jobs that were being tuned when the state was dumped."""
        for job in self.job_set.values():
            job.state = JobState.WAITING if job.state == JobState.TUNING else job.state

    def tune_impl(self, progress):
        """Tune workloads with locally.

        .. note::
            Local tuner will not update the progress bar in order to keep the console concise.

        Parameters
        ----------
        progress: tqdm
            The formulated progress bar to be updated progressively.
        """
        self.job_configs.localize(self.target)
        while self.waiting_jobs:
            curr_job = self.waiting_jobs.pop()
            assert curr_job.stateless_hash in self.job_set
            curr_job.state = JobState.TUNING
            curr_job.tune(
                self.job_configs.tune_options,
                self.job_configs.measure_options,
                self.job_configs.commit_options,
            )
            log.info(curr_job.result)
            curr_job.state = JobState.FINISHED


@register_job_manager(
    "batch",
    "{ job_queue: <Job queue>, job_def: <Job definition>, "
    "job_bucket: <s3://bucket/folder(optional)>}",
)
class AWSBatchJobManager(JobManagerBase):
    """AWS batch job manager class."""

    # Timeout for checking if a job is halted or failed, unit is in seconds.
    RELAUNCH_TIMEOUT = 1800

    def __init__(
        self,
        target: str,
        jobs: List[Job],
        configs: argparse.Namespace,
    ):
        """Initialize an AWS batch job manager."""
        super(AWSBatchJobManager, self).__init__(target, jobs, configs)

        # Parse batch environment.
        batch_info: Dict[str, Any] = load_from_yaml(configs.batch)

        for field in ["target", "job_queue", "job_def", "job_bucket"]:
            if field not in batch_info:
                raise RuntimeError("%s is missing in AWS batch config" % field)
        self.job_queue = batch_info["job_queue"]
        self.job_def = batch_info["job_def"]
        self.job_bucket = batch_info["job_bucket"]
        self.container_env: Dict[str, str] = {}

        # Parse AWS credentials.
        session = boto3.session.Session()
        if session.region_name is None:
            raise RuntimeError('AWS region is unset. Please use "aws configure" to setup a region')
        self.container_env["AWS_REGION"] = session.region_name

        self.terminated_jobs: List[str] = []
        self.last_check_time = time.time()

        credential = session.get_credentials()
        if credential is None:
            raise RuntimeError(
                'AWS credential is required for AWS batch. Please use "aws configure" to setup'
            )
        self.container_env["AWS_ACCESS_KEY_ID"] = credential.access_key
        self.container_env["AWS_SECRET_ACCESS_KEY"] = credential.secret_key

        self.job_configs_str = dump_to_yaml(self.job_configs)

        # Map from AWS batch job ID to job.
        self.jobid_2_job: Dict[str, Job] = {}

    def desc(self) -> str:
        return "AWS batch job queue {0}".format(self.job_queue)

    def resume_job_states(self):
        """Check if the tuning jobs have correct metadata."""
        invalid_jobs = []
        for job in self.job_set.values():
            if job.state != JobState.TUNING:
                continue

            try:
                job.get_metadata("AWSBatchJobID")
                job.get_metadata("AWSBatchJobDetail")
            except RuntimeError:
                log.warning("AWSBatchJobID or AWSBatchJobDetail is missing in %s", job)
                invalid_jobs.append(job.stateless_hash)
                continue

        # Remove invalid jobs from this tuning.
        for job_hash in invalid_jobs:
            del self.job_set[job_hash]

    def tune_impl(self, progress):
        """Tune workloads with AWS batch.

        Parameters
        ----------
        progress: tqdm
            The formulated progress bar to be updated progressively.
        """
        # Add resumed tuning jobs.
        for job in self.job_set.values():
            if job.state == JobState.TUNING:
                self.jobid_2_job[job.get_metadata("AWSBatchJobID")] = job

        # Connect to AWS batch.
        batch_client = boto3.client("batch")

        # Make the command.
        command = ["python3", "-m", "lorien", "tune", "--local", self.target]
        command += ["--job-configs", self.job_configs_str]

        # Submit for tuning.
        while self.waiting_jobs:
            curr_job = self.waiting_jobs.pop()
            job_str = dump_to_yaml(curr_job)
            if len(job_str) > 7000:
                # AWS batch limits the job payload to 30 KiB and the container overrride
                # length to 8192, so we cannot directly submit the serialized job
                # if it is too large.
                with tempfile.NamedTemporaryFile(
                    mode="w", prefix="lorien_upload_job_", suffix=".yaml"
                ) as filep:
                    filep.write(job_str)
                    filep.flush()
                    s3_path = "s3://{0}/{1}".format(self.job_bucket, os.path.basename(filep.name))
                    err_msg = upload_s3_file(filep.name, s3_path)
                if err_msg:
                    raise RuntimeError(err_msg)
                job_str = s3_path

            job_detail = {
                "jobName": "lorien-tuning-job",
                "jobQueue": self.job_queue,
                "jobDefinition": self.job_def,
                "containerOverrides": {
                    "command": command + ["--job", job_str],
                    "environment": [
                        {"name": name, "value": val} for name, val in self.container_env.items()
                    ],
                },
            }
            try:
                res = batch_client.submit_job(**job_detail)
                self.jobid_2_job[res["jobId"]] = curr_job
                curr_job.set_metadata("AWSBatchJobID", res["jobId"])
                curr_job.set_metadata("AWSBatchJobDetail", job_detail)
                curr_job.state = JobState.TUNING
            except Exception as err:  # pylint: disable=broad-except
                log.warning("Failed to submit %s to AWS batch: %s", str(curr_job), str(err))
                result = TuneResult()
                result.error_code = TuneErrorCode.FAIL_TO_SUBMIT
                result.error_msgs.append("Failed to submit to AWS batch")
                assert curr_job.stateless_hash in self.job_set
                curr_job.result = result
                curr_job.state = JobState.FINISHED
                continue

        # Calculate finished jobs.
        done_count = sum(
            [1 if job.state == JobState.FINISHED else 0 for job in self.job_set.values()]
        )
        progress.update(done_count)

        # Check progress.
        while done_count < self.num_jobs():
            # Divide job IDs to chunks with maximum size 100, because describe_jobs API only allows
            # at most 100 jobs in each query.
            jobs_desc = []
            job_ids = list(self.jobid_2_job.keys())
            chunks = [
                job_ids[i * 100 : (i + 1) * 100] for i in range((len(job_ids) + 100 - 1) // 100)
            ]
            for chunk in chunks:
                jobs_desc += batch_client.describe_jobs(jobs=chunk)["jobs"]

            # Filter finished jobs
            success_ids = [desc["jobId"] for desc in jobs_desc if desc["status"] == "SUCCEEDED"]
            fail_ids = [desc["jobId"] for desc in jobs_desc if desc["status"] == "FAILED"]
            new_done_count = len(success_ids) + len(fail_ids)

            # Remove finished jobs and update results.
            for jobid in fail_ids:
                curr_job = self.jobid_2_job[jobid]
                result = TuneResult()
                result.error_code = TuneErrorCode.FAIL_TO_GET_RESULT
                result.error_msgs.append("Failed")
                assert curr_job.stateless_hash in self.job_set
                curr_job.result = result
                curr_job.state = JobState.FINISHED
                del self.jobid_2_job[jobid]
            for jobid in success_ids:
                curr_job = self.jobid_2_job[jobid]
                result = TuneResult()
                result.error_code = TuneErrorCode.FAIL_TO_GET_RESULT
                result.error_msgs.append("Success")
                assert curr_job.stateless_hash in self.job_set
                curr_job.result = result
                curr_job.state = JobState.FINISHED
                del self.jobid_2_job[jobid]

            # Update status.
            progress.update(new_done_count)
            done_count += new_done_count
            time.sleep(1)

            self.relaunch_hanging_jobs(batch_client, jobs_desc)

    def relaunch_hanging_jobs(
        self,
        batch_client,
        jobs_desc: List[Dict[str, Any]],
    ):
        """Timed monitoring check for halted/failed jobs and relaunch them

        Parameters
        ----------
        batch_client: botocore.client.Batch
            boto3 client for AWS batch. Note that we do not annotate boto3 types because
            it requires an additional package.
        jobs_desc: List[Dict[str, Any]]
            Job descriptions that extracted from each job.
        """

        def get_log_events(log_group: str, stream_name: List[str], filter_pattern: str = ""):
            try:
                client = boto3.client("logs")
                resp = client.filter_log_events(
                    logGroupName=log_group,
                    logStreamNames=stream_name,
                    filterPattern=filter_pattern,
                    limit=10000,
                )
                return resp["events"]
            except Exception as err:  # pylint: disable=broad-except
                log.warning(
                    "Failed to obtain log from AWS Cloudwatch for: %s %s %s",
                    log,
                    stream_name,
                    str(err),
                )
                return []

        self.curr_time = time.time()
        if self.curr_time - self.last_check_time >= self.RELAUNCH_TIMEOUT:
            # Check all running jobs when relaunch timeout is reached
            running_jobs = [
                [desc["jobId"], desc["container"]["logStreamName"]]
                for desc in jobs_desc
                if desc["status"] == "RUNNING"
            ]

            for running_job in running_jobs:
                job_id, log_stream = running_job[0], running_job[1]
                # Prevent redundantly killing terminated jobs
                if job_id in self.terminated_jobs:
                    continue

                job = self.jobid_2_job[job_id]
                job_detail = job.get_metadata("AWSBatchJobDetail")
                events = get_log_events("/aws/batch/job", [log_stream], "Too many errors")
                is_in_debug_mode = bool(events)

                relaunch = False
                if is_in_debug_mode:
                    events = get_log_events("/aws/batch/job", [log_stream], "LOG_COUNT")
                    assert len(events) >= 2  # Should not have less than 2 event logs.

                    first_time_interval = events[1]["timestamp"] - events[0]["timestamp"]
                    curr_timestamp = events[1]["timestamp"]

                    # Check the runtime of searching a number of trials and terminate the job
                    # if it now needs 10x longer time to finish the same number of trials.
                    for event in events[2:]:
                        time_interval = event["timestamp"] - curr_timestamp
                        if time_interval > first_time_interval * 10:
                            relaunch = True
                            break
                        curr_timestamp = event["timestamp"]

                # Relaunch the same job and terminate the hanging one.
                if relaunch:
                    try:
                        res = batch_client.submit_job(**job_detail)
                    except Exception as err:  # pylint: disable=broad-except
                        log.warning("Failed to submit the relaunched job : %s %s", job_id, str(err))
                        break

                    try:
                        batch_client.terminate_job(
                            jobId=job_id,
                            reason="In DEBUG mode for a long time. "
                            "Relaunch a job with ID {}".format(res["jobId"]),
                        )
                        self.terminated_jobs.append(job_id)
                    except Exception as err:  # pylint: disable=broad-except
                        log.warning("Failed to terminate job : %s %s", job_id, str(err))
                        break

                    job.set_metadata("AWSBatchJobID", res["jobId"])
                    job.set_metadata("AWSBatchJobDetail", job_detail)
                    self.jobid_2_job[res["jobId"]] = job
                    break

            self.last_check_time = self.curr_time


@register_job_manager("rpc", "port")
class RPCJobManager(JobManagerBase):
    """RPC job manager class."""

    def __init__(
        self,
        target: str,
        jobs: List[Job],
        configs: argparse.Namespace,
    ):
        """Initialize a RPC job manager."""
        super(RPCJobManager, self).__init__(target, jobs, configs)

        # Parse server info.
        server_info: Dict[str, Any] = load_from_yaml(configs.rpc)

        if "port" not in server_info:
            raise RuntimeError("port is missing in RPC server config")
        server_port = server_info["port"]

        # Launch a RPC server with daemon so that it will be terminated with the main thread.
        self.server = Thread(target=launch_server, args=(server_port, target), daemon=True)
        try:
            self.server.start()
        except Exception as err:  # pylint: disable=broad-except
            raise RuntimeError("Failed to launch RPC server: %s" % str(err))

        configs.server = "localhost:{}".format(server_port)
        configs.target = target

        # Launch a RPC client to initialize the server.
        log.info("Initializing RPC server")
        while True:
            try:
                self.client = RPCClient(configs, silent=True)
                break
            except Exception as err:  # pylint: disable=broad-except
                time.sleep(1)
                continue

        self.client.init_server(self.job_configs)
        if not self.client.is_server_init():
            raise RuntimeError("Failed to initialize RPC server")

    def desc(self) -> str:
        return "{0} RPC workers".format(self.client.num_workers())

    def resume_job_states(self):
        """Resume the jobs that were being tuned when the state was dumped."""
        for job in self.job_set.values():
            job.state = JobState.WAITING if job.state == JobState.TUNING else job.state

    def tune_impl(self, progress):
        """Tune workloads with RPC hosts.

        Parameters
        ----------
        progress: tqdm
            The formulated progress bar to be updated progressively.
        """

        # Submit for tuning.
        done_count = 0
        while self.waiting_jobs or done_count < self.num_jobs():
            # Fetch (job, tune result) pairs in YAML string format.
            new_results: List[Tuple[str, str]] = self.client.fetch_results()
            if new_results:
                progress.update(len(new_results))
                done_count += len(new_results)

                # Process results.
                for job_str, result_str in new_results:
                    # Note that the new created job is a different object as in the job set,
                    # and we must use the original object to maintain the trace.
                    job_hash = load_from_yaml(job_str, Job).stateless_hash
                    result: TuneResult = load_from_yaml(result_str, TuneResult)
                    assert job_hash in self.job_set
                    curr_job = self.job_set[job_hash]
                    curr_job.state = JobState.FINISHED
                    curr_job.result = result

                    # Commit results if workers did not do so.
                    if "tune_logs" in result.metadata:
                        log_file = os.path.join("/tmp", curr_job.workload.get_log_file_name())
                        with open(log_file, "w") as filep:
                            filep.write(result.metadata["tune_logs"])
                            filep.flush()
                        result.log_file = log_file
                        result.commit(
                            self.job_configs.commit_options, workload=curr_job.workload, silent=True
                        )
                        os.remove(log_file)
                        del result.metadata["tune_logs"]

            # Submit new jobs.
            while self.waiting_jobs:
                curr_job = self.waiting_jobs.pop()
                # Keep submitting until server refuses to accept.
                if not self.client.submit(dump_to_yaml(curr_job)):
                    self.waiting_jobs.append(curr_job)
                    break
                curr_job.state = JobState.TUNING

            # Update status.
            progress.set_description(self.desc())
            time.sleep(1)
