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
The system config module.
"""
import argparse
from typing import Any, Callable, Dict, List, Tuple

from .logger import get_logger
from .util import dump_to_yaml, load_from_yaml

log = get_logger("Configs")

CONFIG_GROUP: Dict[str, argparse.ArgumentParser] = {}
APPEND_GROUP: Dict[Tuple[str, str], List[Tuple[List[str], Dict[str, Any]]]] = {}
CONFIG_BUILT: bool = False


def read_args_from_files(astrs: List[str]) -> List[str]:
    """Convert command line arguments to a list. If the argument starts with the defined
    "fromfile_prefix_chars" (@), then it indicates a config file in YAML format.
    This function will parse the config file and expand the content to argument list.

    For example, a CLI is defined as follows. The `model` command can be specified multiple
    times to form a list of models.

    .. code-block:: bash

        exec --model alexnet --model resnet50_v2

    This is equivalent to the following:

    .. code-block:: bash

        exec @models.yaml

    where the config file models.yaml has:

    .. code-block:: yaml

        model:
          - alexnet
          - resnet50_v2

    Parameters
    ----------
    astrs: List[str]
        The full list of command line arguments.

    Returns
    -------
    arg_list: List[str]
        The expanded argument list.
    """

    arg_list: List[str] = []
    for astr in astrs:
        if not astr.startswith("@"):
            arg_list.append(astr)
        else:
            with open(astr[1:], "r") as filep:
                data = load_from_yaml(filep.read())
            if not isinstance(data, dict):
                raise RuntimeError("The top level in YAML config is not dict: %s" % astr)

            # Expand the list. If the values is not a simple string then we serialize it back
            # and let the module handle.
            for key, vals in data.items():
                if isinstance(vals, list):
                    for val in vals:
                        arg_list += [
                            "--{}".format(key),
                            val if isinstance(val, str) else dump_to_yaml(val),
                        ]
                else:
                    arg_list += ["--{}".format(key), str(vals)]
    return arg_list


def create_config_parser(desc: str, **kwargs) -> argparse.ArgumentParser:
    """Create an argument parser to parse the CLIs.

    Parameters
    ----------
    desc: str
        The descriptions to the paresr being created.

    Returns
    -------
    parser: argparse.ArgumentParser
        The created parser.
    """
    parser = argparse.ArgumentParser(
        description=desc,
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        **kwargs
    )
    parser._read_args_from_files = read_args_from_files  # type: ignore
    return parser


def register_config_parser(full_name: str):
    """Register config parser for parsing command line interface.

    Parameters
    ----------
    full_name: str
        The full command name. If the command should be register under another command,
        using "." to represent the complete hierarchy (e.g., gen-workload.modelzoo).

    Returns
    -------
    reg: Callable
        A callable function for registration.
    """

    def _do_reg(func: Callable):
        if full_name in CONFIG_GROUP:
            raise RuntimeError("Config parser of %s has been registered" % full_name)
        if full_name.find(".") == -1 and full_name != "top":
            raise RuntimeError('The main entry must be named "top"')

        CONFIG_GROUP[full_name] = func()

    return _do_reg


def append_config_parser(full_name: str, desc: str):
    """Append configs to another config parser as a config group for parsing command line interface.

    Parameters
    ----------
    full_name: str
        The full command name to be appended. Use "." to represent the complete hierarchy of
        the config parser to be appended.

    desc: str
        The description of the config gruop.

    Returns
    -------
    reg: Callable
        A callable function for appending.
    """

    def _do_append(func: Callable):
        key = (full_name, desc)
        if key in APPEND_GROUP:
            raise RuntimeError("Config group %s has been registered" % str(key))
        if full_name.find(".") == -1:
            raise RuntimeError("Config group appends to nothing")

        APPEND_GROUP[key] = func()

    return _do_append


def make_config_parser(sys_args: List[str]) -> argparse.Namespace:
    """Generate CLI from the registered configs.

    Parameters
    ----------
    sys_args: List[str]
        The list of system arguments (e.g., sys.argv).

    Returns
    -------
    configs: argparse.Namespace
        The parsed CLI argument namespace.
    """
    global CONFIG_BUILT

    if CONFIG_BUILT:
        return CONFIG_GROUP["top"].parse_args(sys_args)
    CONFIG_BUILT = True

    if "top" not in CONFIG_GROUP:
        raise RuntimeError('Main entry config "top" is not registered')

    # Register sub-configs to their parents
    for full_command, parser in CONFIG_GROUP.items():
        if full_command == "top":  # Main entry has no parent.
            continue
        parent = full_command[: full_command.rfind(".")]
        command_name = full_command[full_command.rfind(".") + 1 :]

        if parent not in CONFIG_GROUP:
            raise RuntimeError("Parent config %s is not registered" % parent)

        parent_parser = CONFIG_GROUP[parent]
        subparsers = [
            p for p in parent_parser._actions if isinstance(p, argparse._SubParsersAction)
        ]
        if not subparsers:
            raise RuntimeError("Config %s has no sub-config for registration" % parent)
        subparser = subparsers[0]
        subparser.choices[command_name] = parser

    # Append config groups
    for (full_target_name, desc), config_group in APPEND_GROUP.items():
        if full_target_name not in CONFIG_GROUP:
            raise RuntimeError(
                "Target config to be append is not registered: %s" % full_target_name
            )

        target_parser = CONFIG_GROUP[full_target_name]
        group = target_parser.add_argument_group(desc)
        for arg, kwargs in config_group:
            group.add_argument(*arg, **kwargs)

    # Broadcast top level arguments
    top_actions = []
    for action in CONFIG_GROUP["top"]._get_optional_actions():
        if isinstance(action, argparse._HelpAction):
            continue
        top_actions.append(action)

    for full_command, parser in CONFIG_GROUP.items():
        if full_command == "top":
            continue
        for action in top_actions:
            parser._add_action(action)

    return CONFIG_GROUP["top"].parse_args(sys_args)
