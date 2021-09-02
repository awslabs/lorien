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
Utility functions for TVM dialects
"""
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import boto3

import tvm
from tvm.autotvm.task import Task
from tvm.ir.container import Array
from tvm.target import Target

from ...database.table import scan_table
from ...database.util import convert_to_db_list, convert_to_list
from ...util import serialize_framework_build_config

TASK_TO_REQUIRED_LIBS = {
    "dense_mkl.x86": "mkl",
    "dense_mkldnn.x86": "mkldnn",
    "dense_cblas.x86": "cblas",
    "batch_matmul_cblas.x86": "cblas",
    "dense_cublas.cuda": "cublas",
    "batch_matmul_cublas.cuda": "cublas",
    "softmax.cudnn": "cudnn",
    "conv2d_cudnn.cuda": "cudnn",
    "conv3d_cudnn.cuda": "cudnn",
}


def check_tvm_version(curr: str, min_req: str) -> bool:
    """Check if the current TVM version satisfies the minimum requirement.

    Parameters
    ----------
    curr: str
        The current version.

    min_req: str
        The minimum requirement version.

    Returns
    -------
    check: bool
        Return true if the current version satisfies the minimum requirement.
    """

    def parse_version(ver_str: str) -> Tuple[Tuple[int, int, int], bool]:
        """Parse TVM version to a tuple-3.

        Parameters
        ----------
        ver_str: str
            The TVM version string.

        Returns
        -------
        ver: Tuple[Tuple[int, int, int], bool]
            (3-way version number, is a release version)
        """

        # The release version such as 0.8.0.
        tokens = re.search(r"(\d|)\.(\d+)\.(\d+)", ver_str)
        if tokens is not None:
            return ((int(tokens.group(1)), int(tokens.group(2)), int(tokens.group(3))), True)

        # The dev version such as 0.8.dev0 or 0.8.dev94+g0d07a329e.
        tokens = re.search(r"(\d|)\.(\d+)\.dev(\d+)\+*.*", ver_str)
        if tokens is not None:
            return ((int(tokens.group(1)), int(tokens.group(2)), int(tokens.group(3))), False)

        raise RuntimeError("Unrecognized TVM version: %s" % ver_str)

    curr_ver, curr_rel = parse_version(curr)
    req_ver, req_rel = parse_version(min_req)

    # Major version.
    if curr_ver[0] < req_ver[0]:
        return False
    if curr_ver[0] > req_ver[0]:
        return True

    # Miner version.
    if curr_ver[1] < req_ver[1]:
        return False
    if curr_ver[1] > req_ver[1]:
        return True

    if curr_rel and not req_rel:
        # Current version is "release" but the target version is "dev".
        return True
    if not curr_rel and req_rel:
        # Current version is "dev" but the target version is "release".
        return False

    # Both are "dev" versions.
    return curr_ver[2] >= req_ver[2]


def get_tvm_build_config() -> Dict[str, str]:
    """Get the TVM build config such as commit, LLVM and CUDA version.

    Returns
    -------
    tvm_config: Dict[str, str]
        A dict of build config.
    """
    libinfo = tvm.support.libinfo()
    config = {}

    def getter_helper(name):
        if name not in libinfo or libinfo[name] == "NOT-FOUND":
            return None
        return libinfo[name]

    # Extract commit hash.
    config["commit"] = getter_helper("GIT_COMMIT_HASH")
    config["commit_time"] = getter_helper("GIT_COMMIT_TIME")

    # Extract LLVM and CUDA version.
    config["llvm_version"] = getter_helper("LLVM_VERSION")
    config["cuda_version"] = getter_helper("CUDA_VERSION")
    return config


TVM_BUILD_CONFIG = get_tvm_build_config()


def gen_target_id_keys(target: str) -> str:
    """Generate a string that includes TVM target ID and keys.

    Parameters
    ----------
    target: str
        TVM target string.

    Returns
    -------
    target_id_keys: str
        A string including target ID and keys.
    """
    try:
        tvm_target = tvm.target.Target(target)
    except ValueError as err:
        raise RuntimeError("Invalid TVM target string %s: %s" % (target, str(err)))

    return "{0}_{1}".format(str(tvm_target.kind.name), "-".join(tvm_target.keys))


def get_tvm_container_value_str(val: Any):
    """Get the primitive value in string from a TVM container.

    Parameters
    ----------
    val: Any
        The TVM container.

    Returns
    -------
    ret: str
        The container value in string.
    """
    return str(val.value) if hasattr(val, "value") else str(val)


def get_canonical_tvm_target_str(
    target: Union[str, Target], task: Optional[Task] = None, remove_libs: bool = False
) -> str:
    """Generate a canonical TVM target string. Note that we do not use the builtin
    string function in TVM target object because we need to sort all attributes
    and their values to guarantee the target strings from the functional equivalent
    targets are identical.

    Parameters
    ----------
    target: Union[str, Target]
        TVM target or target string.

    task: Optional[Task]
        The task of the given target. The canonical target depends on the required libraries
        of the task. If None, then no additional process will be applied to the target.

    remove_libs: bool
        Whether to remove the -libs in the target string.

    Returns
    -------
    ret: str
        Canonical TVM target string.
    """
    if isinstance(target, str):
        try:
            tvm_target = tvm.target.Target(target)
        except ValueError as err:
            raise RuntimeError("Invalid TVM target string %s: %s" % (target, str(err)))
    else:
        tvm_target = target

    ret = str(tvm_target.kind.name)
    if tvm_target.keys:  # Do not sort keys as they are ordered.
        ret += " -keys={}".format(",".join(tvm_target.keys))
    if tvm_target.attrs:
        # Process attributes.
        attrs: Dict[str, Any] = {}
        for key, val in tvm_target.attrs.items():
            if key == "libs":
                if task is not None:
                    if (
                        hasattr(task, "name") and task.name in TASK_TO_REQUIRED_LIBS
                    ):  # Override with task required libs.
                        val = TASK_TO_REQUIRED_LIBS[task.name]
                    else:  # Remove libs if the task does not require.
                        continue

                if remove_libs:
                    continue
            if isinstance(val, Array):  # Sort attribute value array.
                attrs[key] = sorted(list(val))
            else:
                attrs[key] = val

        # Sort attributes by names.
        for key, ori_val in sorted(attrs.items(), key=lambda p: p[0]):
            vals = []
            if not isinstance(ori_val, list):
                vals = [ori_val]
            else:
                vals = ori_val
            ret += " -{0}={1}".format(key, ",".join([get_tvm_container_value_str(v) for v in vals]))
    return ret


def is_cover(target1: str, target2: str) -> bool:
    """Check if target 1 covers target 2. We define target 1 covers target 2
    if all specified attributes in target 1 are same as target 2.

    Parameters
    ----------
    target1: str
        The TVM taret string.

    target2: str
        The TVM taret string.

    Returns
    -------
    cover: bool
        True if target 1 covers target 2.
    """

    try:
        tvm_target1 = tvm.target.Target(target1)
    except ValueError as err:
        raise RuntimeError("Invalid TVM target string %s: %s" % (target1, str(err)))

    try:
        tvm_target2 = tvm.target.Target(target2)
    except ValueError as err:
        raise RuntimeError("Invalid TVM target string %s: %s" % (target2, str(err)))

    # Check kind and keys.
    if tvm_target1.kind != tvm_target2.kind:
        return False
    if str(tvm_target1.keys) != str(tvm_target2.keys):
        return False

    # Only check target 1 attributes.
    for key, val in tvm_target1.attrs.items():
        if key not in tvm_target2.attrs:
            return False
        if isinstance(val, Array):
            if key == "libs":
                if any([v not in tvm_target2.attrs[key] for v in val]):
                    return False
            else:
                val_str1 = ",".join([get_tvm_container_value_str(v) for v in sorted(val)])
                val_str2 = ",".join(
                    [get_tvm_container_value_str(v) for v in sorted(tvm_target2.attrs[key])]
                )
                if val_str1 != val_str2:
                    return False
        elif get_tvm_container_value_str(val) != get_tvm_container_value_str(
            tvm_target2.attrs[key]
        ):
            return False
    return True


def prune_table_item(item: Dict[str, Any], nbest: int = 3) -> Dict[str, Any]:
    """Prune the best configs in a DynamoDB item by the following steps:
    1. Group by TVM build config and keep the best one for each group.
    2. Group by config and keep the most recent one for each group.
    3. Prune the rest configs based on user-strategy.

    Parameters
    ----------
    item: Dict[str, Any]
        The item to be pruned.

    nbest: int
        Keep the number of the most recent unique configs. Default 3.

    Returns
    -------
    ret: Dict[str, Any]
        A pruned item.
    """
    assert "BestConfigs" in item

    # Group by TVM build config and keep the best one for each group.
    tvm_commit_to_config: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for config_info in convert_to_list(item["BestConfigs"]):
        tvm_build_hash = serialize_framework_build_config(config_info["tvm_build_config"])
        if (
            tvm_build_hash not in tvm_commit_to_config
            or config_info["latency"] < tvm_commit_to_config[tvm_build_hash]["latency"]
        ):
            tvm_commit_to_config[tvm_build_hash] = config_info

    # Group by the best config and keep the most recent one for each group.
    config_hash_to_config: Dict[str, Dict[str, Any]] = {}
    for config_info in tvm_commit_to_config.values():
        config_hash = config_info["config"]
        new_info_time = datetime.strptime(
            config_info["tvm_build_config"]["commit_time"], "%Y-%m-%d %H:%M:%S %z"
        )
        exist_info_time = None
        if config_hash in config_hash_to_config:
            exist_info_time = datetime.strptime(
                config_hash_to_config[config_hash]["tvm_build_config"]["commit_time"],
                "%Y-%m-%d %H:%M:%S %z",
            )

        if exist_info_time is None or new_info_time > exist_info_time:
            config_hash_to_config[config_hash] = config_info

    new_item = deepcopy(item)
    new_configs = sorted(
        config_hash_to_config.values(),
        key=lambda x: datetime.strptime(
            x["tvm_build_config"]["commit_time"], "%Y-%m-%d %H:%M:%S %z"
        ),
        reverse=True,
    )
    if len(new_configs) > nbest:
        new_configs = new_configs[:nbest]
    new_item["BestConfigs"] = convert_to_db_list(new_configs)
    return new_item


def prune_table(table_name: str, nbest: int = 3, **db_kwargs) -> None:
    """Prune the best configs in a given table. Specifically, it scans a table, prune
    each item in the table and puts items back. See ``prune_items()``.

    Parameters
    ----------
    table_name: str
        The target table name to be pruned.

    nbest: int
        Keep the number of the most recent unique configs. Default 3.

    **db_kwargs
        The kwargs of boto3 client. For example, use "endpoint_url=http://localhost:8000"
        for local DynamoDB.
    """

    client = boto3.client("dynamodb", **db_kwargs)
    scanner = scan_table(table_name, **db_kwargs)
    while True:
        # Get a batch of items.
        try:
            resp = next(scanner)
        except StopIteration:
            break

        # Pruning.
        assert resp["Count"] > 0
        with ProcessPoolExecutor(max_workers=resp["Count"]) as pool:
            futures = [
                pool.submit(prune_table_item, item=item, nbest=nbest) for item in resp["Items"]
            ]
            for future in as_completed(futures):
                item = future.result()

                # Write pruned items.
                try:
                    client.update_item(
                        TableName=table_name,
                        Key={"Target": item["Target"], "PrimaryRangeKey": item["PrimaryRangeKey"]},
                        UpdateExpression="SET BestConfigs = :config",
                        ExpressionAttributeValues={":config": item["BestConfigs"]},
                    )
                except Exception as err:  # pylint:disable=broad-except
                    raise RuntimeError("Failed to write the pruned item: %s" % str(err))
