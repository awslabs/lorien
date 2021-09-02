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
AutoTVM dialects.
"""
import argparse

from ....configs import create_config_parser, register_config_parser

from . import extract_from_model, extract_from_record
from .job import AutoTVMJob
from .result import AutoTVMTuneResult
from .workload import AutoTVMWorkload


@register_config_parser("top.generate.autotvm")
def define_config() -> argparse.ArgumentParser:
    """Define the command line interface for AutoTVM workload generation.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("AutoTVM Workload Generation")

    # Define generators
    subparsers = parser.add_subparsers(
        dest="mode", description="The mode to generate AutoTVM workloads"
    )
    subparsers.required = True
    return parser
