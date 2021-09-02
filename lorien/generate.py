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
Workload generator.
"""
import argparse
from typing import Callable, List

from .configs import create_config_parser, register_config_parser
from .logger import get_logger
from .util import dump_to_yaml
from .workload import Workload

log = get_logger("Generator")


def gen(gen_func: Callable[[argparse.Namespace], List[Workload]]) -> Callable:
    """Generate workloads based on configs.

    Parameters
    ----------
    gen_func: Callable[[argparse.Namespace], List[WorkloadBase]]
        The generator function that accepts generator specific configs and returns
        a list of generated workloads.

    Returns
    -------
    ret: Callable
        The entry function that uses the given generation function to generate workloads.
    """

    def _do_gen(configs: argparse.Namespace):
        """Invoke the generator function to get the workload list, and validate if the workload
        is a valid for AutoTVM.

        Parameters
        ----------
        configs: argparse.Namespace
            The configuration of the generator.
        """
        # Collect workloads
        log.info("Generating workloads...")
        workloads: List[Workload] = gen_func(configs)

        # Dump each workload to a string with the last "\n" removed, and
        # aggregate all dumped workloads to a single dict to match tuning config.
        with open(configs.output, "w") as workload_file:
            dumped = dump_to_yaml(
                {"workload": [dump_to_yaml(w)[:-1] for w in workloads]}, single_line=False
            )
            assert dumped is not None
            workload_file.write(dumped)

    return _do_gen


@register_config_parser("top.generate")
def define_config() -> argparse.ArgumentParser:
    """Define the command line interface for workload generation.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """
    parser = create_config_parser("Workload Generation")

    # Define generators
    subparsers = parser.add_subparsers(dest="dialect", description="The generator dialect")
    subparsers.required = True
    return parser
