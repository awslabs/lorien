"""
The unit test module for AutoTVM dialect.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import, wrong-import-position, ungrouped-imports
import argparse
import glob
import os
import tempfile
from copy import deepcopy

import json
import mock
import numpy as np
import pytest
from moto import mock_dynamodb2

from lorien.util import is_dialect_enabled

if not is_dialect_enabled("autotvm"):
    pytest.skip("AutoTVM dialect is not available", allow_module_level=True)

from tvm import autotvm
from tvm.autotvm.measure import MeasureInput, MeasureResult
from tvm.autotvm.task.space import ConfigEntity

from lorien.database.table import create_table
from lorien.dialect.tvm_dial.autotvm_dial.job import (
    AutoTVMJob,
    AutoTVMJobConfigs,
    create_autotvm_tuner,
)
from lorien.dialect.tvm_dial.autotvm_dial.extract_from_model import extract_from_models
from lorien.dialect.tvm_dial.autotvm_dial.extract_from_record import extract_from_records
from lorien.dialect.tvm_dial.autotvm_dial.result import AutoTVMRecords, AutoTVMTuneResult
from lorien.dialect.tvm_dial.autotvm_dial.workload import AutoTVMWorkload
from lorien.tune.result import TuneErrorCode
from lorien.util import dump_to_yaml, load_from_yaml


@pytest.fixture
def fixture_autotvm_workload():
    # This fixture workload has 429 configs.
    workload = AutoTVMWorkload()
    workload.task_name = "dense_nopack.x86"
    workload.args = [
        ["TENSOR", [1, 9216], "float32"],
        ["TENSOR", [4096, 9216], "float32"],
        None,
        "float32",
    ]
    workload.lib = "topi"
    workload.target = "llvm"
    return workload


def gen_x86_conv2d_log_record(target, n_config, data, weight, stride, padding, dilation):
    records = []

    # Generate configs for 3 different data layouts to test the commit mechanism.
    layouts = [(1, 8), (2, 16), (4, 4)]
    assert n_config % len(layouts) == 0

    # Generate records to mimic tuning logs.
    inp = [
        target,
        "conv2d_NCHWc.x86",
        [
            ["TENSOR", data, "float32"],
            ["TENSOR", weight, "float32"],
            (stride, stride),
            (padding, padding, padding, padding),
            (dilation, dilation),
            "NCHW",
            "NCHW",
            "float32",
        ],
        {},
    ]
    latencies = np.random.uniform(1e-6, 1e-5, n_config)
    for idx in range(0, n_config, len(layouts)):
        for sid, layout in enumerate(layouts):
            entity = [
                ["tile_ic", "sp", [-1, layout[0]]],
                ["tile_oc", "sp", [-1, layout[1]]],
                ["tile_ow", "sp", [-1, 1]],
                ["unroll_kw", "ot", False],
            ]
            records.append(
                {
                    "input": inp,
                    "config": {"index": idx, "code_hash": "", "entity": entity},
                    "result": [[latencies[idx + sid]], 0, 0, idx + sid],
                    "version": 0.2,
                    "tvm_version": "0.7.dev1",
                }
            )
    return records


def gen_cuda_conv2d_log_record(n_config, data, weight, stride, padding, dilation):
    records = []

    # Generate records to mimic tuning logs.
    inp = [
        "cuda",
        "conv2d_nchw.cuda",
        [
            ["TENSOR", data, "float32"],
            ["TENSOR", weight, "float32"],
            (stride, stride),
            (padding, padding, padding, padding),
            (dilation, dilation),
            "float32",
        ],
        {},
    ]
    latencies = np.random.uniform(1e-6, 1e-5, n_config)
    for idx in range(n_config):
        entity = [
            ["tile_f", "sp", [-1, 1, 1, idx + 1]],
            ["tile_y", "sp", [-1, 1, 1, 1]],
            ["tile_x", "sp", [-1, 1, 1, 1]],
            ["tile_rc", "sp", [-1, 1]],
            ["tile_ry", "sp", [-1, 1]],
            ["tile_rx", "sp", [-1, 1]],
            ["auto_unroll_max_step", "ot", 0],
            ["unroll_explicit", "ot", 0],
        ]
        records.append(
            {
                "input": inp,
                "config": {"index": idx, "code_hash": "", "entity": entity},
                "result": [[latencies[idx]], 0, 0, idx],
                "version": 0.2,
                "tvm_version": "0.7.dev1",
            }
        )
    return records


def gen_dense_log_record_w_cblas(target, n_config, shape_a, shape_b):
    records = []

    # Generate records to mimic tuning logs.
    assert n_config > 1, "Must have at least one non-vendor library record"
    n_config -= 1
    inp = [
        target,
        "dense_pack.x86",
        [["TENSOR", shape_a, "float32"], ["TENSOR", shape_b, "float32"], None, "float32"],
        {},
    ]
    latencies = np.random.uniform(1e-6, 1e-5, n_config)
    for idx in range(n_config):
        entity = [
            ["tile_y", "sp", [-1, 1, idx + 1]],
            ["tile_x", "sp", [-1, 1, 1]],
            ["tile_k", "sp", [-1, 1]],
        ]
        records.append(
            {
                "input": inp,
                "config": {"index": idx, "code_hash": "", "entity": entity},
                "result": [[latencies[idx]], 0, 0, idx],
                "version": 0.2,
                "tvm_version": "0.7.dev1",
            }
        )

    # Add one vendor library record.
    inp = [
        target,
        "dense_cblas.x86",
        [["TENSOR", shape_a, "float32"], ["TENSOR", shape_b, "float32"], None, "float32"],
        {},
    ]
    records.append(
        {
            "input": inp,
            "config": {"index": 0, "code_hash": "", "entity": []},
            "result": [[5e-7], 0, 0, 0],
            "version": 0.2,
            "tvm_version": "0.7.dev1",
        }
    )
    return records


def test_workload():
    # pylint:disable=missing-docstring, redefined-outer-name

    workload = AutoTVMWorkload()
    workload.target = "cuda -model=v100 -libs=cublas"

    # Test invalid arguments caused task creation failure
    workload.args = [[1, 3, 224, 224], [32, 3, 3, 3]]
    with pytest.raises(RuntimeError):
        workload.to_task()

    workload.args = [
        ["TENSOR", [1, 3, 224, 224], "float32"],
        ["TENSOR", [32, 3, 3, 3], "float32"],
    ]

    # Test missing task definition
    with pytest.raises(RuntimeError):
        workload.to_task()

    workload.task_name = "conv2d_nchw_winograd.cuda"

    # Test invalid workload for the TOPI schedule. conv2d winograd on CUDA only accepts stide 1.
    workload.args += [[2, 2], [1, 1, 1, 1], [1, 1], "float32"]
    with pytest.raises(RuntimeError):
        workload.to_task()

    workload.args[-4] = [1, 1]
    task = workload.to_task()
    assert isinstance(workload.to_job(), AutoTVMJob)

    # Test load from task. -libs should be removed from target since conv2d_nchw_winograd.cuda
    # does not depend on it.
    workload_from_task = AutoTVMWorkload.from_task(task)
    assert (
        workload_from_task.target
        == "cuda -keys=cuda,gpu -max_num_threads=1024 -model=v100 -thread_warp_size=32"
    )

    # Other than that should be identical.
    workload_from_task.target = workload.target
    assert workload == workload_from_task
    task.target = None
    with pytest.raises(RuntimeError):
        AutoTVMWorkload.from_task(task)

    # Test dump and load from YAML
    workload_str = dump_to_yaml(workload)
    assert workload == load_from_yaml(workload_str, AutoTVMWorkload)

    workload2 = deepcopy(workload)

    # Different argument values.
    workload2.args[-2] = [0, 0]
    assert workload > workload2

    # Different argument numbers.
    workload2.args = workload2.args[:-1]
    assert workload > workload2

    # Different target.
    workload2.target = "cuda -model=zz"
    assert workload < workload2

    # Test loading invalid workload
    with pytest.raises(RuntimeError):
        load_from_yaml(workload_str.replace("TENSOR", ""), AutoTVMWorkload)

    # Test mutation
    workload = AutoTVMWorkload()
    workload.task_name = "conv2d_NCHWc.x86"
    workload.target = "llvm"
    workload.args = [
        ["TENSOR", [1, 3, 224, 224], "float32"],
        ["TENSOR", [32, 3, 3, 3], "float32"],
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        "NCHW",
        "NCHW",
        "float32",
    ]

    # A rule to mutate batch size and channel
    rules = {(0, 1, 0): "[1, 2, 3, 4]", (0, 1, 1): "[v, v * 2, v * 4]"}

    mutated = workload.mutate(rules)
    assert len(mutated) == 12

    # Wrong index
    rules = {(0, 1, 0, 0): "[1, 2, 3, 4]"}
    with pytest.raises(RuntimeError):
        workload.mutate(rules)

    # Wrong description
    rules = {(0, 1, 0): "[a, a * 2]"}
    with pytest.raises(RuntimeError):
        workload.mutate(rules)


def test_create_autotvm_tuner(fixture_autotvm_workload):
    task = fixture_autotvm_workload.to_task()

    create_autotvm_tuner("xgb", task)
    create_autotvm_tuner("ga", task)
    create_autotvm_tuner("random", task)
    create_autotvm_tuner("gridsearch", task)

    with pytest.raises(RuntimeError):
        create_autotvm_tuner("wrong-tuner", task)


@mock_dynamodb2
def test_job_n_configs_n_commit_n_query(mocker, fixture_autotvm_workload):
    table_name = "lorien-test"
    arn = create_table(table_name, region_name="us-west-2")

    workload = fixture_autotvm_workload
    job = workload.to_job()
    assert isinstance(job, AutoTVMJob)
    assert not job.is_target_compatible("cuda")
    task = workload.to_task()

    configs = argparse.Namespace(
        tuner="random",
        ntrial=4,
        test=1,
        repeat=1,
        min=400,
        db="{ region_name: us-west-2 }",
        commit_table_name=table_name,
        commit_nbest=1,
        commit_workload=False,
        commit_log_to=None,
    )

    job_configs = job.create_job_configs(configs)
    job_configs.commit_options["table-arn"] = arn
    assert isinstance(job_configs, AutoTVMJobConfigs)
    assert job_configs.tune_options
    assert job_configs.measure_options
    assert job_configs.check_tvm_build_config()

    # Localize with RPC runner
    rpc_config = argparse.Namespace(device="test-device", runner_port=188875)
    job_configs.localize("llvm", configs=rpc_config)

    with tempfile.TemporaryDirectory(prefix="lorien-test-autotvm-commit-") as temp_dir:
        # Localize with local runner
        job_configs = job.create_job_configs(configs)
        job_configs.tune_options["tune_dir"] = temp_dir
        job_configs.commit_options["table-arn"] = arn
        job_configs.tvm_build_config = {}
        job_configs.localize("llvm")

        def mock_tuner_no_valid(_, task):
            class MockTuner:
                def tune(self, n_trial, early_stopping, measure_option, callbacks):
                    for _ in range(2):
                        res = mock.MagicMock()
                        res.error_no = 2
                        callbacks[1](None, [None], [res])

            return MockTuner()

        mocker.patch(
            "lorien.dialect.tvm_dial.autotvm_dial.job.create_autotvm_tuner"
        ).side_effect = mock_tuner_no_valid
        job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
        assert job.result.error_code == TuneErrorCode.NO_VALID_RESULT

        def mock_tuner(_, task):
            class MockTuner:
                def __init__(self, task):
                    self.task = task

                def tune(self, n_trial, early_stopping, measure_option, callbacks):
                    # Write to log file to test commit
                    inp = MeasureInput("llvm", self.task, ConfigEntity(0, "", {}, []))
                    ret = MeasureResult([10], 0, 20, 0)
                    callbacks[0](None, [inp], [ret])

                    inp = MeasureInput("llvm", self.task, ConfigEntity(1, "", {}, []))
                    ret = MeasureResult([1e8], 2, 20, 0)
                    callbacks[0](None, [inp], [ret])

                    # Update metadata
                    res = mock.MagicMock()
                    res.error_no = 0
                    res.costs = [1, 1, 1]

                    inp = mock.MagicMock()
                    inp.task = mock.MagicMock()
                    inp.task.flop = 1e9
                    callbacks[1](None, [inp], [res])

            return MockTuner(task)

        mocker.patch(
            "lorien.dialect.tvm_dial.autotvm_dial.job.create_autotvm_tuner"
        ).side_effect = mock_tuner

        # Do not commit
        job.tune(job_configs.tune_options, job_configs.measure_options, commit_options=None)
        assert job.result.error_code == TuneErrorCode.NORMAL
        assert "tune_logs" in job.result.metadata

        # Success
        job.tune(job_configs.tune_options, job_configs.measure_options, job_configs.commit_options)
        assert job.result.error_code == TuneErrorCode.NORMAL
        assert "tune_logs" not in job.result.metadata

        # Test failed to localize
        mock_check_tvm_build_config = mock.MagicMock()
        mock_check_tvm_build_config.return_value = False
        job_configs.check_tvm_build_config = mock_check_tvm_build_config
        with pytest.raises(RuntimeError):
            job_configs.localize("llvm")

        log_file = os.path.join(temp_dir, "tune.log")
        inps = [
            MeasureInput("llvm", task, ConfigEntity(1, "", {}, [])),
            MeasureInput("llvm", task, ConfigEntity(2, "", {}, [])),
        ]
        ress = [MeasureResult([1e8], 2, 20, 0), MeasureResult([1e2], 0, 20, 0)]
        autotvm.callback.log_to_file(log_file)(None, inps, ress)

        # Add other records to test the filter.
        with open(log_file, "a") as filep:
            records = gen_dense_log_record_w_cblas(
                "llvm -mcpu=core-avx2 -libs=cblas", 5, [100, 1024], [256, 1024]
            )
            for record in records:
                filep.write("{}\n".format(json.dumps(record)))

        records = AutoTVMTuneResult.create_records_by_workloads(log_file, 1, workload)
        assert len(records) == 1
        assert records[0].target_key == "llvm -keys=cpu -link-params=0", records[0].target_key
        assert records[0].alter_key == "llvm_cpu", records[0].alter_key
        assert (  # pylint: disable=line-too-long
            records[0].workload_key
            == "dense_nopack.x86#_TENSOR__1_9216__float32_#_TENSOR__4096_9216__float32_#None#float32"
        ), records[0].workload_key

        job.result.commit_tuning_log(
            workload, log_file, table_name, nbest=1, region_name="us-west-2"
        )
        job.result.commit_tuning_log(None, log_file, table_name, nbest=1, region_name="us-west-2")

        records = AutoTVMRecords(task.target, workload.get_workload_key())
        records.query(table_name, region_name="us-west-2")
        assert len(records) == 1

        records = AutoTVMRecords(task.target, workload.get_workload_key())
        records.query(table_name, use_alter_key=True, region_name="us-west-2")
        assert len(records) == 1

        # Do not provide workload key to query all records with the same target
        records = AutoTVMRecords("llvm", workload_key=None)
        records.query(table_name, region_name="us-west-2")
        assert len(records) == 1


def test_extract_from_model():
    configs = argparse.Namespace(
        gcv=["alexnet", "alexnet: { data: [1, 3, 224, 224]}"],
        target=["llvm -libs=cblas"],
        tf=[],
        tflite=[],
        onnx=[],
        keras=[],
        torch=[],
        mxnet=[],
    )
    workloads = extract_from_models(configs)
    assert len(workloads) == 14, "\nWorkloads:\n%s" % "\n".join([str(wkl) for wkl in workloads])

    # Test failure.
    configs = argparse.Namespace(
        gcv=["alexnet_wrong_name"],
        target=["llvm"],
        tf=[],
        tflite=[],
        onnx=[],
        keras=[],
        torch=[],
        mxnet=[],
    )
    workloads = extract_from_models(configs)
    assert len(workloads) == 0


@mock_dynamodb2
def test_extract_from_record(mocker):
    # Mock a table.
    records = gen_x86_conv2d_log_record(
        "llvm -mcpu=core-avx2 -libs=cblas", 6, [1, 1024, 32, 32], [16, 1024, 3, 3], 1, 1, 1
    )
    records += gen_dense_log_record_w_cblas(
        "llvm -mcpu=core-avx2 -libs=cblas", 5, [100, 1024], [256, 1024]
    )
    table_name = "lorien-test"
    with tempfile.TemporaryDirectory(prefix="lorien-test-autotvm-layout-") as temp_dir:
        create_table(table_name, region_name="us-west-2")
        log_file = "{}/fake.log".format(temp_dir)
        with open(log_file, "w") as filep:
            for record in records:
                filep.write("{}\n".format(json.dumps(record)))
        AutoTVMTuneResult().commit_tuning_log(None, log_file, table_name, region_name="us-west-2")

    # Test layout transform workload generation.
    configs = argparse.Namespace(
        table_name=table_name,
        db='{ "region_name": "us-west-2" }',
        target=["llvm"],
        ignore_target_attrs=False,
    )

    # The target "llvm" does not match "llvm -mcpu=core-avx2" so it should get nothing
    # unless we enable ignore-target-attrs.
    assert len(extract_from_records(configs)) == 0

    # "gen_x86_conv2d_log_record" generates 3 layouts, but one of them has the same
    # input and output layout so it should be ignored when generting layout transform workloads.
    # In addition, all records from "gen_dense_log_record_w_cblas" should be ignored because layout
    # transform does not support dense.
    configs.ignore_target_attrs = True
    assert len(extract_from_records(configs)) == 2

    # Intend to fail all task creations.
    mocker.patch(
        "lorien.dialect.tvm_dial.autotvm_dial.extract_from_record.autotvm.task.create"
    ).side_effect = Exception()
    assert not extract_from_records(configs)


def test_gen_feature():
    with tempfile.TemporaryDirectory(prefix="lorien-test-autotvm-feature-") as temp_dir:
        log_dir = os.path.join(temp_dir, "logs")
        os.mkdir(log_dir)

        # Generate the first log file, which includes conv2d_NCHWc.x86
        log_file = os.path.join(log_dir, "fake1.log")
        with open(log_file, "w") as filep:
            records = gen_x86_conv2d_log_record(
                "llvm -mcpu=core-avx2", 6, [1, 1024, 32, 32], [16, 1024, 3, 3], 1, 1, 1
            )
            failed_record = deepcopy(records[0])
            failed_record["result"][1] = 1  # let error code be non-zero.
            records.append(failed_record)
            for record in records:
                filep.write("{}\n".format(json.dumps(record)))

        # Generate the second log file, which includes dense_cblas.x86 and dense_pack.x86
        log_file = os.path.join(log_dir, "fake2.log")
        with open(log_file, "w") as filep:
            records = gen_dense_log_record_w_cblas(
                "llvm -mcpu=core-avx2", 5, [100, 1024], [256, 1024]
            )
            for record in records:
                filep.write("{}\n".format(json.dumps(record)))

        feature_dir = os.path.join(temp_dir, "features")
        AutoTVMTuneResult.gen_features(log_dir, feature_dir)

        # The lock files should be removed.
        assert not glob.glob("{}/**/*.lock".format(feature_dir), recursive=True)

        def check_helper(name, n_data, n_numeric_features, n_category_features):
            """Check dumped feature files."""
            csv_file = os.path.join(feature_dir, "{}.csv".format(name))
            meta_file = os.path.join(feature_dir, "{}.meta".format(name))
            assert os.path.exists(csv_file), "Missing %s" % csv_file
            assert os.path.exists(meta_file), "Missing %s" % meta_file

            with open(csv_file, "r") as filep:
                features = filep.readline().replace("\n", "").split(",")
                assert len(features) == n_numeric_features + n_category_features + 1
                assert len(filep.read().split("\n")) == n_data + 1

            with open(meta_file, "r") as filep:
                n_numeric = 0
                n_category = 0
                for line in filep:
                    tokens = line.split(",")
                    if tokens[1] == "numeric":
                        n_numeric += 1
                    elif tokens[1] == "category":
                        n_category += 1

                assert n_numeric == n_numeric_features
                assert n_category == n_category_features

        check_helper("conv2d_NCHWc.x86", 7, 22, 6)
        check_helper("dense_cblas.x86", 1, 4, 4)
        check_helper("dense_pack.x86", 4, 12, 4)


def test_extract_feature(fixture_autotvm_workload):
    task = fixture_autotvm_workload.to_task()
    config_dict = {
        "index": 7,
        "code_hash": "some_hash",
        "entity": [
            ("tile", "sp", [16, 4]),
            ("reorder", "re", [0, 2, 1]),
            ("annotate", "an", "unroll"),
            ("other", "ot", "auto"),
        ],
    }
    config = ConfigEntity.from_json_dict(config_dict)
    inp = MeasureInput("llvm", task, config)

    features = AutoTVMTuneResult.extract_feature(inp)
    expected_features = {
        "in_0": 1,
        "in_1": 9216,
        "in_2": "float32",
        "in_3": 4096,
        "in_4": 9216,
        "in_5": "float32",
        "attr_0": None,
        "attr_1": "float32",
        "sp_tile_0": 16,
        "sp_tile_1": 4,
        "re_reorder": "0;2;1",
        "an_annotate": "unroll",
        "ot_other": "auto",
    }
    assert features == expected_features
