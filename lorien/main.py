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
The main flow that integrates all modules
"""
import sys
from .configs import create_config_parser, make_config_parser, register_config_parser
from .logger import enable_log_file


@register_config_parser("top")
def define_config():
    """Define the command line interface for the main entry.

    Returns
    -------
    parser: argparse.ArgumentParser
        The defined argument parser.
    """

    parser = create_config_parser("Lorien: TVM Optimized Schedule Database", prog="lorien")
    parser.add_argument(
        "--log-run", action="store_true", default=False, help="Log execution logs to a file"
    )
    subparsers = parser.add_subparsers(dest="command", help="The command being executed")
    subparsers.required = True
    return parser


class Main:
    """The main entry."""

    def __init__(self):
        args = make_config_parser(sys.argv[1:])
        if args.log_run:
            enable_log_file()
        args.entry(args)
