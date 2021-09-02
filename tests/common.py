"""
The common utilities for unit tests.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
import socket

import boto3
import pytest
from moto import mock_dynamodb2, mock_s3

from lorien.database.table import create_table
from lorien.tune.job import Job, JobConfigs
from lorien.tune.result import TuneResult
from lorien.workload import Workload


class LorienTestJobConfig(JobConfigs):
    pass


class LorienTestJob(Job):
    # pylint: disable=abstract-method

    @staticmethod
    def create_job_configs(configs):
        return LorienTestJobConfig(configs)

    def is_target_compatible(self, target):
        return self.workload.target == target

    def tune(self, tune_options, measure_options, commit_options=None):
        result = TuneResult()
        result.metadata["tune_logs"] = "tuning logs"
        result.commit = lambda options, workload, silent: None


class LorienTestWorkload(Workload):
    # pylint: disable=abstract-method
    def __init__(self, target, idx):
        super(LorienTestWorkload, self).__init__()
        self.target = target
        self.idx = idx
        self.dummy_data = ""

    def to_job(self):
        return LorienTestJob(self)

    def __repr__(self):
        return "LorienTestWorkload(%s, %d)" % (self.target, self.idx)


def gen_workloads(lower_idx, upper_idx, target="llvm"):
    """Generate a number of dummy workloads."""
    return [LorienTestWorkload(target, idx) for idx in range(lower_idx, upper_idx)]


def gen_jobs(lower_idx, upper_idx, target="llvm"):
    """Generate a number of dummy jobs."""
    return [LorienTestWorkload(target, idx).to_job() for idx in range(lower_idx, upper_idx)]


def find_first_available_port():
    """Find an available port to perform RPC tests."""
    skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    skt.bind(("0.0.0.0", 0))
    _, port = skt.getsockname()
    skt.close()
    return port


@pytest.fixture
def mock_s3_client():
    with mock_s3():
        client = boto3.client("s3")
        client.create_bucket(
            Bucket="unit-test-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
        yield client


@pytest.fixture
def mock_db_table_arn():
    table_name = "unit-test-lorien"
    with mock_dynamodb2():
        arn = create_table(table_name)
        yield (table_name, arn)
