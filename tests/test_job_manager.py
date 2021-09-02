"""
The unit tests for job managers.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import
import argparse
import os
import subprocess
import tempfile
import threading
import time
import uuid

import mock
import pytest
from moto import mock_dynamodb2

from lorien import main  # pylint: disable=unused-import
from lorien.configs import make_config_parser
from lorien.logger import get_logger
from lorien.tune.job import JobState
from lorien.tune.job_manager import AWSBatchJobManager, LocalJobManager, RPCJobManager
from lorien.tune.result import TuneResult
from lorien.tune.rpc.client import RPCClient
from lorien.tune.rpc.launch import ClientThread
from lorien.util import dump_to_yaml, load_from_yaml

from .common import (
    LorienTestJob,
    LorienTestJobConfig,
    find_first_available_port,
    gen_jobs,
    gen_workloads,
    mock_s3_client,
)

log = get_logger("Unit-Test")


@mock_dynamodb2
def test_local_manager():
    with tempfile.TemporaryDirectory(prefix="lorien_test_local_mgr_") as temp_dir:
        trace_file_name = "{}/local-manager-test.trace".format(temp_dir)
        configs = make_config_parser(
            [
                "tune",
                "--trace-file",
                trace_file_name,
                "--local",
                "llvm",
                "--db",
                "{ region_name: us-west-2 }",
            ]
        )
        jobs = gen_jobs(0, 5, "cuda")

        # No target jobs
        with pytest.raises(RuntimeError):
            LocalJobManager("llvm", jobs, configs)

        jobs += gen_jobs(5, 10, "llvm")
        manager = LocalJobManager("llvm", jobs, configs)
        assert manager.desc() == "Local"
        assert manager.target == "llvm"
        assert manager.num_jobs() == 5
        assert len(manager.waiting_jobs) == 5

        manager.tune()

        # Add some noise to the trace file
        with open(trace_file_name, "a") as filep:
            filep.write("aaa\n")  # Invalid trace
            filep.write("timestamp\tinvalid_job\n")  # Invalid job format
            job = gen_jobs(10, 11, "llvm")[0]
            filep.write("timestamp\t%s" % dump_to_yaml(job))  # A never seen job

        # Replay trace
        manager = LocalJobManager("llvm", jobs, configs)
        assert len(manager.waiting_jobs) == 1


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@mock_dynamodb2
def test_aws_batch_manager(aws_credentials, mock_s3_client):
    with tempfile.TemporaryDirectory(prefix="lorien_test_aws_batch_mgr_") as temp_dir:
        trace_file_name = "{}/aws-batch-manager-test.trace".format(temp_dir)
        batch_cfg = {"target": "llvm", "job_queue": "a", "job_def": "b"}
        configs = make_config_parser(
            [
                "tune",
                "--trace-file",
                trace_file_name,
                "--batch",
                dump_to_yaml(batch_cfg),
                "--db",
                "{ region_name: us-west-2 }",
            ]
        )
        jobs = gen_jobs(0, 5, "llvm")

        # Make this job larger than the limited size so that it will be uploaded to S3.
        for i in range(1000, 7000):
            jobs[4].workload.dummy_data += str(i)

        # Missing required field
        with pytest.raises(RuntimeError):
            AWSBatchJobManager("llvm", jobs, configs)

        batch_cfg["job_bucket"] = "unit-test-bucket"
        configs = make_config_parser(
            [
                "tune",
                "--trace-file",
                trace_file_name,
                "--batch",
                dump_to_yaml(batch_cfg),
                "--db",
                "{ region_name: us-west-2 }",
            ]
        )

        manager = AWSBatchJobManager("llvm", jobs, configs)
        assert manager.container_env["AWS_ACCESS_KEY_ID"] == "foobar_key"
        assert manager.container_env["AWS_SECRET_ACCESS_KEY"] == "foobar_secret"
        assert manager.container_env["AWS_REGION"] == "us-east-1"

        def mock_submit_job_func(**kwargs):
            """A mock function to make the job with workload ID 3n failed to be submitted."""
            job = load_from_yaml(kwargs["containerOverrides"]["command"][-1])
            if job.workload.idx > 0 and job.workload.idx % 3 == 0:
                raise RuntimeError("Expected failure")
            return {"jobId": str(uuid.uuid4())}

        def mock_desc_job_func(jobs):
            """A mock function to make a half jobs successed and another half failed."""
            return {
                "jobs": [
                    {
                        "jobId": job_id,
                        "status": "SUCCEEDED" if idx % 2 == 0 else "FAILED",
                        "container": {"logStreamName": "/aws/batch"},
                    }
                    for idx, job_id in enumerate(jobs)
                ]
            }

        def mock_filter_log_func(**kwargs):
            return {
                "events": [
                    {
                        "timestamp": 0,
                        "message": "LOG_COUNT 4 trials. Failed count 0. Best thrpt 20.12 GFlop/s",
                    },
                    {
                        "timestamp": 10,
                        "message": "LOG_COUNT 8 trials. Failed count 0. Best thrpt 20.12 GFlop/s",
                    },
                    {
                        "timestamp": 1000,
                        "message": "LOG_COUNT 12 trials. Failed count 0. Best thrpt 20.12 GFlop/s",
                    },
                    {
                        "timestamp": 1002,
                        "message": "WARNING:autotvm:Too many errors happen in the tuning. "
                        "Now is in debug mode",
                    },
                ]
            }

        def mock_terminate_job_func(jobId, reason):
            if jobId == "hanging-fail-to-terminate":
                raise RuntimeError("Expected failure")

        with mock.patch("lorien.tune.job_manager.boto3.client") as mock_batch:
            mock_batch_client = mock.MagicMock()
            mock_batch.return_value = mock_batch_client
            mock_batch_client.submit_job = mock.MagicMock()
            mock_batch_client.submit_job.side_effect = mock_submit_job_func
            mock_batch_client.describe_jobs = mock.MagicMock()
            mock_batch_client.describe_jobs.side_effect = mock_desc_job_func
            mock_batch_client.terminate_job = mock.MagicMock()
            mock_batch_client.terminate_job.side_effect = mock_terminate_job_func
            mock_batch_client.filter_log_events = mock.MagicMock()
            mock_batch_client.filter_log_events.side_effect = mock_filter_log_func
            manager.last_check_time = manager.last_check_time - 20000

            results = manager.tune()
            assert len(results) == 5
            assert not manager.waiting_jobs

            # Test relaunching
            hanging_jobs = gen_jobs(7, 10, "llvm")
            job_descs = []
            for job_id, job in zip(
                ["hanging", "hanging-2", "hanging-fail-to-terminate"], hanging_jobs
            ):
                job.state = JobState.TUNING
                fake_job_detail = {
                    "jobId": job_id,
                    "status": "RUNNING",
                    "container": {"logStreamName": "valid_stream"},
                    "containerOverrides": {"command": [dump_to_yaml(job)]},
                }
                job_descs.append(fake_job_detail)
                job._metadata["AWSBatchJobID"] = job_id
                job._metadata["AWSBatchJobDetail"] = fake_job_detail
                manager.jobid_2_job[job_id] = job

            manager.last_check_time -= manager.RELAUNCH_TIMEOUT
            manager.relaunch_hanging_jobs(mock_batch_client, job_descs)

            # Put new jobs to test resuming
            new_jobs = gen_jobs(5, 7, "llvm")
            new_jobs[0].state = JobState.TUNING
            manager.job_set[new_jobs[0].stateless_hash] = new_jobs[0]

            new_jobs[1].state = JobState.TUNING
            new_jobs[1]._metadata["AWSBatchJobID"] = "resume_job"
            new_jobs[1]._metadata["AWSBatchJobDetail"] = {}
            manager.job_set[new_jobs[1].stateless_hash] = new_jobs[1]

            manager.resume_job_states()
            assert new_jobs[0].stateless_hash not in manager.job_set
            assert new_jobs[1].stateless_hash in manager.job_set


@mock_dynamodb2
def test_rpc_manager():
    with tempfile.TemporaryDirectory(prefix="lorien_test_rpc_mgr_") as temp_dir:
        jobs = gen_jobs(0, 5, "llvm")
        trace_file_name = "{}/rpc-manager-test.trace".format(temp_dir)
        rpc_cfg = {"target": "llvm"}
        configs = make_config_parser(
            [
                "tune",
                "--trace-file",
                trace_file_name,
                "--rpc",
                dump_to_yaml(rpc_cfg),
                "--db",
                "{ region_name: us-west-2 }",
            ]
        )

        # Missing required field (i.e., port)
        with pytest.raises(RuntimeError):
            RPCJobManager("llvm", jobs, configs)

        rpc_cfg["port"] = find_first_available_port()
        configs = make_config_parser(
            [
                "tune",
                "--trace-file",
                trace_file_name,
                "--rpc",
                dump_to_yaml(rpc_cfg),
                "--db",
                "{ region_name: us-west-2 }",
            ]
        )
        manager = RPCJobManager("llvm", jobs, configs)
        assert manager.desc().find("RPC workers") != -1
        assert manager.target == "llvm"
        assert manager.num_jobs() == 5
        assert len(manager.waiting_jobs) == 5

        client_configs = make_config_parser(
            [
                "rpc-client",
                "--server",
                "0.0.0.0:{}".format(rpc_cfg["port"]),
                "--target",
                "llvm",
            ]
        )
        client_thread = ClientThread(client_configs)
        client_thread.start()

        results = manager.tune()
        assert len(results) == 5
        assert all(["tune_logs" not in r.metadata for r in results])
        client_thread.join(1)


def test_rpc_client(mocker):
    def mock_get_job_configs_str(token):
        if not token:
            return ""

        configs = argparse.Namespace(
            db="{ region_name: us-west-2 }",
            commit_table_name="table",
            commit_nbest=1,
            commit_workload=False,
            commit_log_to=None,
        )
        job_config = LorienTestJobConfig(configs)
        job_config.commit_options["table-arn"] = "arn"
        return dump_to_yaml(job_config)

    def mock_request_job(token):
        if not token:
            return ""
        job = gen_jobs(0, 1, "llvm")[0]
        return dump_to_yaml(job)

    def mock_send_result(token, job_n_result):
        if not token:
            return "not register"
        if job_n_result[0] == "random_job":
            return "not allocated to tune"
        return ""

    mock_client_conn = mock.MagicMock()
    mock_client_conn._channel.stream.sock.getsockname.return_value = ("fake-socket", 100)
    mock_server = mock.MagicMock()
    mock_server.is_init.return_value = True
    mock_server.get_job_configs_str.side_effect = mock_get_job_configs_str
    mock_server.request_job.side_effect = mock_request_job
    mock_server.send_result.side_effect = mock_send_result
    mock_server.register_as_worker.return_value = (True, "fake-token")
    mock_server.fetch_results.side_effect = RuntimeError("Expected failure")
    mock_client_conn.root = mock_server

    mocker.patch("lorien.tune.rpc.client.rpyc.connect").return_value = mock_client_conn
    mocker.patch("lorien.tune.rpc.client.check_table").return_value = False

    # Test client
    # Invalid port
    client_configs = make_config_parser(
        ["rpc-client", "--server", "0.0.0.0:aa", "--target", "cuda"]
    )
    with pytest.raises(RuntimeError):
        RPCClient(client_configs)

    # Missing port
    client_configs = make_config_parser(["rpc-client", "--server", "0.0.0.0", "--target", "cuda"])
    with pytest.raises(RuntimeError):
        RPCClient(client_configs)

    client_configs = make_config_parser(
        ["rpc-client", "--server", "0.0.0.0:18872", "--target", "llvm"]
    )

    client = RPCClient(client_configs)
    assert client.is_server_init()

    # Fail to initialize the worker due to no register
    with pytest.raises(RuntimeError):
        client.init_worker(client_configs)

    # Cannot request a job before registration
    assert not client.request_job()
    assert client.send_result(TuneResult())

    ret = client.register_as_worker()
    assert ret[0], ret[1]

    client.init_worker(client_configs)
    assert client.job_configs is not None
    assert client.job_configs.commit_options is None

    # Fail due to not root
    with pytest.raises(RuntimeError):
        client.fetch_results()

    assert client.send_result(("random_job", TuneResult())).find("not allocated to tune") != -1

    job_str = client.request_job()
    assert job_str is not None
    assert isinstance(load_from_yaml(job_str), LorienTestJob)

    # Make a result with tuning log to let the master commit
    result = TuneResult()
    assert not client.send_result((job_str, result))
