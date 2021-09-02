"""
The unit test module for TVM dialect utilities.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import, wrong-import-position, ungrouped-imports
import datetime
import pytz

import mock
import boto3
import pytest
from moto import mock_dynamodb2

from lorien.util import is_dialect_enabled

if not is_dialect_enabled("tvm"):
    pytest.skip("TVM dialect is not available", allow_module_level=True)

import tvm
from lorien.database.util import convert_to_db_dict, convert_to_db_list
from lorien.database.table import create_table, scan_table
from lorien.dialect.tvm_dial.util import (
    check_tvm_version,
    gen_target_id_keys,
    get_canonical_tvm_target_str,
    get_tvm_build_config,
    is_cover,
    prune_table,
)


def test_check_tvm_version():
    # Invalid version.
    with pytest.raises(RuntimeError):
        check_tvm_version("a", "b")

    # Same version.
    assert check_tvm_version("0.6.0", "0.6.0")
    assert check_tvm_version("0.7.dev1", "0.7.dev1")
    assert check_tvm_version("0.8.dev94+g0d07a329e", "0.8.dev94+g0d07a329e")

    # Older versions.
    assert not check_tvm_version("0.6.0", "1.6.dev1")
    assert not check_tvm_version("0.6.0", "0.7.dev1")
    assert not check_tvm_version("0.6.1", "0.7.dev1")
    assert not check_tvm_version("0.7.dev0", "0.7.dev1")
    assert not check_tvm_version("0.7.dev1", "0.7.0")
    assert not check_tvm_version("0.8.dev0", "0.8.dev94+g0d07a329e")
    assert not check_tvm_version("0.8.dev94+g0d07a329e", "0.8.dev95+aaaaaaaaaa")

    # Newer versions.
    assert check_tvm_version("1.7.dev1", "0.8.0")
    assert check_tvm_version("0.7.0", "0.7.dev1")
    assert check_tvm_version("0.7.1", "0.7.dev1")
    assert check_tvm_version("0.8.dev0", "0.7.dev1")
    assert check_tvm_version("0.8.dev95+aaaaaaaaaa", "0.8.dev94+g0d07a329e")


def test_tvm_build_config():
    with mock.patch("tvm.support.libinfo") as mock_libinfo:
        # Test missing fields
        mock_libinfo.return_value = {"GIT_COMMIT_HASH": "aaa"}
        tvm_config = get_tvm_build_config()
        assert tvm_config == {
            "commit": "aaa",
            "commit_time": None,
            "llvm_version": None,
            "cuda_version": None,
        }

        mock_libinfo.return_value = {
            "GIT_COMMIT_HASH": "bbb",
            "GIT_COMMIT_TIME": "2021-07-07 13:28:23 +0900",
            "LLVM_VERSION": "10.0.1",
            "CUDA_VERSION": "10.2",
        }
        tvm_config = get_tvm_build_config()
        assert tvm_config == {
            "commit": "bbb",
            "commit_time": "2021-07-07 13:28:23 +0900",
            "llvm_version": "10.0.1",
            "cuda_version": "10.2",
        }


def test_get_id_keys():
    # Invalid target string
    with pytest.raises(RuntimeError):
        gen_target_id_keys("abc")

    assert gen_target_id_keys("llvm") == "llvm_cpu"
    assert gen_target_id_keys("llvm -device=arm_cpu") == "llvm_arm_cpu-cpu"
    assert gen_target_id_keys("cuda -model=v100") == "cuda_cuda-gpu"


def test_get_canonical_tvm_target_str():
    target = tvm.target.Target("llvm")
    assert get_canonical_tvm_target_str(target) == "llvm -keys=cpu -link-params=0"

    with pytest.raises(RuntimeError):
        get_canonical_tvm_target_str("abc")

    assert (
        get_canonical_tvm_target_str("llvm -device=arm_cpu")
        == "llvm -keys=arm_cpu,cpu -device=arm_cpu -link-params=0"
    )
    assert (
        get_canonical_tvm_target_str("llvm -libs=cblas")
        == "llvm -keys=cpu -libs=cblas -link-params=0"
    )
    assert (
        get_canonical_tvm_target_str("llvm -libs=cblas", remove_libs=True)
        == "llvm -keys=cpu -link-params=0"
    )

    dense_args = [["TENSOR", (1, 10), "float32"], ["TENSOR", (1, 10), "float32"], None, "float32"]
    task = tvm.autotvm.task.create("dense_pack.x86", dense_args, "llvm")

    # The task does not depend on cblas so we should remove it.
    assert get_canonical_tvm_target_str("llvm -libs=cblas", task) == "llvm -keys=cpu -link-params=0"

    task = tvm.autotvm.task.create("dense_cblas.x86", dense_args, "llvm")
    # The task depends on cblas so we should keep it.
    assert (
        get_canonical_tvm_target_str("llvm -libs=cblas", task)
        == "llvm -keys=cpu -libs=cblas -link-params=0"
    )

    # If removed_libs=True then we remove libs even the task depends on it.
    assert (
        get_canonical_tvm_target_str("llvm -libs=cblas", task, remove_libs=True)
        == "llvm -keys=cpu -link-params=0"
    )


def test_is_cover():
    with pytest.raises(RuntimeError):
        is_cover("a", "b")

    with pytest.raises(RuntimeError):
        is_cover("llvm", "b")

    assert not is_cover("llvm", "cuda")

    # General covers specific.
    assert is_cover("llvm", "llvm -mcpu=skylake-avx512")

    # Inconsist attributes.
    assert not is_cover(
        "llvm -device=arm_cpu -mattr=+neon,fp-armv8,thumb-mode", "llvm -device=arm_cpu -mattr=+neon"
    )
    assert not is_cover("llvm -mcpu=skylake-avx512", "llvm -mcpu=core-avx2")

    # Required libs have to be specified.
    assert not is_cover("llvm -libs=cblas", "llvm -mcpu=skylake-avx512")
    assert is_cover("llvm -libs=cblas", "llvm -mcpu=skylake-avx512 -libs=cblas")
    assert not is_cover("cuda -libs=cublas", "cuda -libs=cudnn -model=v100")
    assert is_cover("cuda -libs=cublas", "cuda -libs=cublas,cudnn -model=v100")

    assert not is_cover("llvm -device=arm_cpu", "llvm -mcpu=skylake-avx512")
    assert is_cover(
        "llvm -device=arm_cpu",
        "llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon",
    )


@mock_dynamodb2
def test_prune_table():
    table_name = "lorien-test"

    # Create a table and commit some dummy items
    create_table(table_name, region_name="us-west-2")
    client = boto3.client("dynamodb", region_name="us-west-2")
    for idx in range(3):
        time1 = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        time2 = (datetime.datetime.now(pytz.utc) + datetime.timedelta(0, 3)).strftime(
            "%Y-%m-%d %H:%M:%S %z"
        )
        configs = [
            {
                "config": "cfg0",
                "tvm_build_config": {"commit": 1, "commit_time": time1},
                "latency": 1,
            },
            {
                "config": "cfg1",
                "tvm_build_config": {"commit": 1, "commit_time": time1},
                "latency": 2,
            },
            {
                "config": "cfg1",
                "tvm_build_config": {"commit": 2, "commit_time": time2},
                "latency": 1,
            },
            {
                "config": "cfg2",
                "tvm_build_config": {"commit": 2, "commit_time": time2},
                "latency": 3,
            },
        ]

        item = {
            "Target": {"S": "target1"},
            "TargetIDKeys": {"S": "target1_key"},
            "PrimaryRangeKey": {"S": "key{}".format(idx)},
            "BestConfigs": convert_to_db_list(configs),
        }
        client.put_item(TableName=table_name, Item=item)

    # Test prune table.
    prune_table(table_name, nbest=2, region_name="us-west-2")

    scanner = scan_table(table_name, limit=1, region_name="us-west-2")
    while True:
        try:
            resp = next(scanner)
            item = resp["Items"][0]
            assert len(item["BestConfigs"]["L"]) == 2
        except StopIteration:
            break
