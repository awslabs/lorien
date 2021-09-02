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
The module of AutoTVM workload extraction from tuning records. The graph-level optimization
workloads such as data layout transform fit in this scenario.
"""
import argparse
from typing import Optional, Tuple

from tvm import autotvm, te
from tvm.autotvm import graph_tuner  # pylint: disable=unused-import
from tvm.autotvm.record import MeasureInput, MeasureResult
from tvm.autotvm.task import Task

from ....configs import create_config_parser, register_config_parser
from ....generate import gen
from ....logger import get_logger
from ....util import load_from_yaml
from .result import AutoTVMRecords
from .util import infer_task_layout
from .workload import AutoTVMWorkload

log = get_logger("Extract")


def create_layout_transform_task(record: Tuple[MeasureInput, MeasureResult]) -> Optional[Task]:
    """Create an AutoTVM task of layout transform.

    Parameters
    ----------
    record: Tuple[MeasureInput, MeasureResult]
        The AutoTVM record pair to create the layout transform task.

    Returns
    -------
    task: Optional[Task]
        The created layout_transform task, or None if the record has no layout transform support.
    """
    layout = infer_task_layout(record)
    if layout is None:
        return None

    try:
        in_shape, in_layout = layout[0][0]
        _, out_layout = layout[1][0]
        if in_layout == out_layout:  # No need to transform layout.
            return None
        data = te.placeholder(in_shape, name="data", dtype=record[0].task.args[0][2])
        args = (data, in_layout, out_layout)
        task = autotvm.task.create("layout_transform", args=args, target=record[0].target)
        return task
    except Exception as err:  # pylint:disable=broad-except
        log.warning("Failed to create layout transfrom task from %s: %s", args, str(err))
    return None


def extract_from_records(configs: argparse.Namespace):
    """Extract graph optimization workloads from a given DB table.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract-from-record.

    Returns
    -------
    workloads: List[AutoTVMWorkload]
        A list of collected workloads.
    """
    db_options = load_from_yaml(configs.db)

    tasks = []
    for target in configs.target:
        records = AutoTVMRecords(target)
        records.query(configs.table_name, configs.ignore_target_attrs, **db_options)
        for record in records.to_list():
            task = create_layout_transform_task(record)
            if task is not None:
                tasks.append(task)

        workloads = list({AutoTVMWorkload.from_task(t) for t in tasks})
        log.info("%d layout transform workloads for %s have been generated", len(workloads), target)

    return workloads


@register_config_parser("top.generate.autotvm.extract-from-record")
def define_config_extract() -> argparse.ArgumentParser:
    """Define the command line interface for workload generation by record extraction.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("AutoTVMWorkload Generation by Tuning Record Extraction")
    parser.add_argument("--table-name", required=True, help="The DynamoDB table name")
    parser.add_argument("--db", default="{ }", help="DynamoDB client options in YAML format")
    parser.add_argument(
        "--ignore-target-attrs",
        default=False,
        action="store_true",
        help="Only use target ID (e.g., llvm) and keys (e.g., cpu) "
        "instead of the full target string when querying records.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        required=True,
        help="A TVM target (e.g., llvm, cuda, etc). "
        "Note that the device tag (e.g., -model=v100) is not required.",
    )
    parser.add_argument(
        "-o", "--output", default="autotvm_workloads.yaml", help="The output file path"
    )
    parser.set_defaults(entry=gen(extract_from_records), validate_task=False)
    return parser
