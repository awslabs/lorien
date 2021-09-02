"""
The unit test for result.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import
import os
import tempfile

import pytest

from lorien.tune.result import Records, TuneErrorCode, TuneResult
from lorien.util import dump_to_yaml, load_from_yaml

from .common import mock_db_table_arn, mock_s3_client


def test_result(mocker, mock_s3_client, mock_db_table_arn):
    table_name, _ = mock_db_table_arn

    class TestRecords(Records[int]):
        def __init__(self, build_config, target_key, alter_key=None, workload_key=None):
            super(TestRecords, self).__init__(target_key, alter_key, workload_key)
            self._data = []
            self._build_config = build_config

        def get_framework_build_config(self):
            return self._build_config

        @staticmethod
        def encode(record):
            return str(record)

        @staticmethod
        def decode(record_str):
            return int(record_str)

        def gen_task_item(self):
            return {}

        @staticmethod
        def gen_record_item(record):
            return {"latency": record}

        def push(self, record):
            self._data.append(record)

        def pop(self):
            self._data, val = self._data[:-1], self._data[-1]
            return val

        def peak(self):
            return self._data[0]

        def to_list(self, nbest=-1):
            return self._data

        def __len__(self) -> int:
            return len(self._data)

    class TestResult1(TuneResult):
        @staticmethod
        def create_records_by_workloads(log_file_path, nbest, workload=None):
            return []

    class TestResult2(TuneResult):
        @staticmethod
        def create_records_by_workloads(log_file_path, nbest, workload=None):
            records = TestRecords({"commit": "abc"}, "workload_key1", "target_ley", "alter_key")
            for idx in range(10):
                records.push(idx)
            return [
                records,
                TestRecords({"commit": "abc"}, "workload_key2", "target_ley", "alter_key"),
            ]

    class TestResult3(TuneResult):
        @staticmethod
        def create_records_by_workloads(log_file_path, nbest, workload=None):
            records = TestRecords({"commit": "pqr"}, "workload_key1", "target_ley", "alter_key")
            for idx in range(10):
                records.push(idx)
            return [records]

    with tempfile.TemporaryDirectory(prefix="lorien_test_result_") as temp_dir:
        log_file = os.path.join(temp_dir, "tuning_log.json")
        with open(log_file, "w") as filep:
            filep.write("aaa\n")

        commit_options = {
            "commit-log": "s3://unit-test-bucket/tuning_log.json",
            "table-name": table_name,
            "nbest": 1,
            "db": {},
        }

        result = TestResult1()
        assert str(result).find("error_code") != -1

        # Failed due to no log file.
        with pytest.raises(RuntimeError):
            result.commit(commit_options)

        # Failed due to no valid record.
        result.log_file = log_file
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.STORAGE_ERROR

        # Success committed twice.
        result = TestResult2()
        result.log_file = log_file
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.NORMAL
        assert not result.error_msgs
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.NORMAL
        assert not result.error_msgs

        # Success committed the same log with different build config.
        result = TestResult3()
        result.log_file = log_file
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.NORMAL
        assert not result.error_msgs

        # Failed to upload to S3
        commit_options["commit-log"] = "s3://invalid-bucket/tuning_log.json"
        result = TestResult3()
        result.log_file = log_file
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.STORAGE_ERROR
        assert result.error_msgs

        # Failed to commit to DB
        mocker.patch.object(TestResult3, "commit_tuning_log").side_effect = RuntimeError
        commit_options["commit-log"] = None
        result = TestResult3()
        result.log_file = log_file
        result.commit(commit_options)
        assert result.error_code == TuneErrorCode.STORAGE_ERROR
        assert any([msg.find("Failed to commit result") != -1 for msg in result.error_msgs])
