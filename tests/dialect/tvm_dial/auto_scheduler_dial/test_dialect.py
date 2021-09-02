"""
The unit test module for AutoScheduler dialect.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import, wrong-import-position, ungrouped-imports
import argparse
import os
import re
import tempfile

import mock
import pytest
from moto import mock_dynamodb2

from lorien.util import is_dialect_enabled

if not is_dialect_enabled("auto_scheduler"):
    pytest.skip("AutoScheduler dialect is not available", allow_module_level=True)

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.measure import MeasureInput, MeasureResult
from tvm.auto_scheduler.measure_record import dump_record_to_string, load_record_from_string
from tvm.auto_scheduler.search_task import SearchTask

from lorien.database.table import create_table
from lorien.dialect.tvm_dial.auto_scheduler_dial.extract import extract_from_models
from lorien.dialect.tvm_dial.auto_scheduler_dial.workload import AutoSchedulerWorkload
from lorien.dialect.tvm_dial.auto_scheduler_dial.job import (
    AutoSchedulerJob,
    AutoSchedulerJobConfigs,
    AutoSchedulerTuneResult,
    RecordToMetadata,
)
from lorien.dialect.tvm_dial.auto_scheduler_dial.result import AutoSchedulerRecords
from lorien.dialect.tvm_dial.job import TuneMetadata
from lorien.tune.result import TuneErrorCode


def gen_conv2d_task(ishape, wshape):
    dtype = "float32"

    data = relay.var("data", shape=(ishape), dtype=dtype)
    weight1 = relay.var("weight1", shape=(wshape), dtype=dtype)

    conv2d = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), padding=(1, 1))
    out = relay.nn.relu(conv2d)
    func = relay.Function([data, weight1], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    tasks, _ = auto_scheduler.extract_tasks(mod, None, "llvm")
    return tasks[0]


def test_workload(mocker):
    task = gen_conv2d_task((1, 3, 224, 224), (32, 3, 3, 3))
    workload = AutoSchedulerWorkload.from_task(task)
    assert str(workload.to_task().compute_dag) == str(task.compute_dag)
    assert str(workload)

    task2 = gen_conv2d_task((1, 32, 112, 112), (32, 32, 3, 3))
    workload2 = AutoSchedulerWorkload.from_task(task2)
    assert workload != workload2
    assert (workload < workload2) == (workload.workload_key < workload2.workload_key)

    job = workload.to_job()
    assert isinstance(job, AutoSchedulerJob)

    mocker.patch(
        "lorien.dialect.tvm_dial.auto_scheduler_dial.workload.pickle.loads",
        side_effect=RuntimeError,
    )
    with pytest.raises(RuntimeError):
        workload.to_task()


def test_job_n_config():
    task = gen_conv2d_task((1, 3, 224, 224), (32, 3, 3, 3))
    workload = AutoSchedulerWorkload.from_task(task)
    job = workload.to_job()

    configs = argparse.Namespace(
        ntrial=4,
        test=1,
        repeat=1,
        min=400,
        db="{ region_name: us-west-2 }",
        commit_table_name="random-table",
        commit_nbest=1,
        commit_workload=False,
        commit_log_to=None,
    )

    job_configs = job.create_job_configs(configs)
    job_configs.commit_options["table-arn"] = "random-arn"
    assert isinstance(job_configs, AutoSchedulerJobConfigs)
    assert job_configs.tune_options
    assert job_configs.measure_options
    assert job_configs.check_tvm_build_config()

    # Localize with RPC runner
    rpc_config = argparse.Namespace(device="test-device", runner_port=188875)
    job_configs.localize("llvm", configs=rpc_config)
    assert "runner" in job_configs.measure_options
    del job_configs.measure_options["runner"]

    # Failed to localize
    mock_check_tvm_build_config = mock.MagicMock()
    mock_check_tvm_build_config.return_value = False
    job_configs.check_tvm_build_config = mock_check_tvm_build_config
    with pytest.raises(RuntimeError):
        job_configs.localize("llvm")

    # Localize with local runner
    job_configs = job.create_job_configs(configs)
    job_configs.commit_options["table-arn"] = "random-arn"
    job_configs.tvm_build_config = {}
    job_configs.localize("llvm")
    assert "measure_ctx" in job_configs.measure_options
    del job_configs.measure_options["measure_ctx"]

    # Test callback separately since coverage cannot reach to TVM PackedFunc
    metadata = TuneMetadata()
    recorder = RecordToMetadata(metadata)

    res = mock.MagicMock()
    res.error_no = 2
    recorder.callback(None, [None], [res])
    assert metadata.trial_count == 1
    assert metadata.failed_count == 1

    inp = mock.MagicMock()
    inp.task = mock.MagicMock()
    inp.task.compute_dag = mock.MagicMock()
    inp.task.compute_dag.flop_ct = 1e9
    res = mock.MagicMock()
    res.error_no = 0
    value = mock.MagicMock()
    value.value = 1
    res.costs = [value]
    recorder.callback(None, [inp], [res])
    assert metadata.trial_count == 2
    assert metadata.failed_count == 1
    assert metadata.max_thrpt == 1


@pytest.fixture
def tune_config_fixture():
    with mock_dynamodb2():
        with tempfile.TemporaryDirectory(prefix="lorien-test-auto_scheduler-commit-") as temp_dir:
            table_name = "lorien-test"
            arn = create_table(table_name, region_name="us-west-2")

            task = gen_conv2d_task((1, 3, 224, 224), (32, 3, 3, 3))
            workload = AutoSchedulerWorkload.from_task(task)
            job = workload.to_job()

            configs = argparse.Namespace(
                ntrial=2,
                test=1,
                repeat=1,
                min=100,
                db="{ region_name: us-west-2 }",
                commit_table_name=table_name,
                commit_nbest=1,
                commit_workload=False,
                commit_log_to=None,
            )

            job_configs = job.create_job_configs(configs)
            job_configs.tune_options["tune_dir"] = temp_dir
            job_configs.commit_options["table-arn"] = arn

            job_configs.localize("llvm")
            yield table_name, task, job, job_configs

            del job_configs.measure_options["measure_ctx"]


def test_tune_n_commit_n_query(tune_config_fixture):
    table_name, task, job, job_configs = tune_config_fixture

    with mock.patch.object(AutoSchedulerWorkload, "to_task", side_effect=RuntimeError):
        job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
        assert job.result.error_code == TuneErrorCode.FAIL_TO_CREATE_TASK

    with mock.patch.object(SearchTask, "tune", return_value=None):
        job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
        assert job.result.error_code == TuneErrorCode.NO_VALID_RESULT

    # Do not commit
    job.tune(job_configs.tune_options, job_configs.measure_options, None)
    assert "tune_logs" in job.result.metadata

    # Success committed
    job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
    assert "tune_logs" not in job.result.metadata

    workload_key = AutoSchedulerWorkload.from_task(task).get_workload_key()
    records = AutoSchedulerRecords(task.target, workload_key)
    records.query(table_name, region_name="us-west-2")
    assert len(records) == 1

    records = AutoSchedulerRecords(task.target, workload_key)
    records.query(table_name, use_alter_key=True, region_name="us-west-2")
    assert len(records) == 1

    # Do not provide workload key to query all records with the same target
    records = AutoSchedulerRecords("llvm", workload_key=None)
    records.query(table_name, region_name="us-west-2")
    assert len(records) == 1

    def mock_commit(self, commit_options, workload, silent=False):
        self.error_code = TuneErrorCode.STORAGE_ERROR

    with mock.patch.object(
        AutoSchedulerTuneResult, "commit", side_effect=mock_commit, autospec=True
    ):
        job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
        assert "tune_logs" in job.result.metadata

    # Test log file filering
    log_file = job.result.log_file
    with tempfile.NamedTemporaryFile(mode="w", prefix="lorien-test-auto-sch-") as temp_file:
        with open(log_file, "r") as filep:
            inp, _ = load_record_from_string(filep.readline())
            # Commented out record
            temp_file.write("\n")
            res = MeasureResult([0.1], 0, "", 0.2, 1)
            temp_file.write("#{}".format(dump_record_to_string(inp, res)))

            # Record with error
            res = MeasureResult([0.2], 2, "", 0.3, 2)
            temp_file.write(dump_record_to_string(inp, res))

            # Record for different workload
            res = MeasureResult([0.3], 0, "", 0.5, 4)
            record_str = dump_record_to_string(inp, res)
            workload_key = re.search(r"\[\"(.+)\", .+\]", str(task.workload_key)).group(1)
            record_str = record_str.replace(workload_key, "aaa")
            temp_file.write(record_str)
            temp_file.flush()

        records = AutoSchedulerTuneResult.create_records_by_workloads(
            temp_file.name, 1, job.workload
        )
        assert len(records) == 1
        assert records[0].target_key == "llvm -keys=cpu -link-params=0"
        assert records[0].alter_key == "llvm_cpu"
        assert records[0].workload_key.find("aaa") == -1


def test_extract_from_model():
    configs = argparse.Namespace(
        gcv=["alexnet", "alexnet: { data: [1, 3, 224, 224]}"],
        target=["llvm -libs=cblas"],
        include_simple_tasks=False,
        tf=[],
        tflite=[],
        onnx=[],
        keras=[],
        torch=[],
        mxnet=[],
    )
    workloads = extract_from_models(configs)
    assert len(workloads) == 8

    # Test failed to create task.
    with mock.patch.object(AutoSchedulerWorkload, "from_task", side_effect=RuntimeError):
        workloads = extract_from_models(configs)
        assert len(workloads) == 0

    # Test failure.
    configs = argparse.Namespace(
        gcv=["alexnet_wrong_name"],
        target=["llvm"],
        include_simple_tasks=False,
        tf=[],
        tflite=[],
        onnx=[],
        keras=[],
        torch=[],
        mxnet=[],
    )
    workloads = extract_from_models(configs)
    assert len(workloads) == 0
