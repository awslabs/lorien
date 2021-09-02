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
The module of AutoTVM workload extraction from DNN models.
"""
import argparse
from typing import Set

import tqdm
from tvm import autotvm

from ....configs import create_config_parser, register_config_parser
from ....generate import gen
from ....logger import get_logger
from ..frontend_parser import EXTRACTOR_FUNC_TABLE
from .workload import AutoTVMWorkload

log = get_logger("Extract")


def extract_from_models(configs: argparse.Namespace):
    """Extract op workloads from a given model.

    Parameters
    ----------
    configs: argparse.Namespace
        The system configure of generate.extract-from-model.

    Returns
    -------
    workloads: List[Workload]
        A list of collected workloads.
    """

    # Extract operator worklaods from models.
    workloads: Set[AutoTVMWorkload] = set()
    for framework in ["gcv", "keras", "onnx", "torch", "tf", "tflite", "mxnet"]:
        if not getattr(configs, framework):
            continue
        mod_n_params = EXTRACTOR_FUNC_TABLE[framework](configs)

        # Extract workloads from models.
        progress = tqdm.tqdm(
            total=len(mod_n_params), desc="", bar_format="{desc} {percentage:3.0f}%|{bar:50}{r_bar}"
        )
        for name, (mod, params) in mod_n_params:
            for target in configs.target:
                progress.set_description_str(name, refresh=True)
                tasks = autotvm.task.extract_from_program(mod, target=target, params=params)

                # Task to workload
                for task in tasks:
                    try:
                        workloads.add(AutoTVMWorkload.from_task(task))
                    except RuntimeError as err:
                        log.warning(
                            "Failed to create workload from task %s: %s", str(task), str(err)
                        )
                        continue
                progress.update(1)

    log.info("%d operator workloads have been generated", len(workloads))
    return list(workloads)


@register_config_parser("top.generate.autotvm.extract-from-model")
def define_config_extract() -> argparse.ArgumentParser:
    """Define the command line interface for workload generation by model extraction.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("Workload Generation by Model Extraction")

    common_desc = (
        "A {0} with input shape in YAML format: "
        '"<model-name>: {{<input-name>: [<input-shape>]}}". When shape is ignored, '
        'the default input {{"{1}": ({2})}} will be applied'
    )

    parser.add_argument(
        "--gcv",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("Gluon CV model name", "data", "1, 3, 224, 224"),
    )
    parser.add_argument(
        "--keras",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("Keras model file path", "input_1", "1, 3, 224, 224"),
    )
    parser.add_argument(
        "--onnx",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("ONNX model file path", "input", "1, 3, 224, 224"),
    )
    parser.add_argument(
        "--torch",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("PyTorch model file path", "input", "1, 3, 224, 224"),
    )
    parser.add_argument(
        "--tf",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("TensorFlow model file path", "Placeholder", "1, 224, 224, 3"),
    )
    parser.add_argument(
        "--tflite",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("TFLite model file path", "Placeholder", "1, 224, 224, 3"),
    )
    parser.add_argument(
        "--mxnet",
        action="append",
        default=[],
        required=False,
        help=common_desc.format("MXNet model file path", "data", "1, 3, 224, 224"),
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
    parser.set_defaults(entry=gen(extract_from_models), validate_task=False)
    return parser
