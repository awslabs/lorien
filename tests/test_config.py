"""
The unit test module for config.
"""
# pylint:disable=unused-import, missing-docstring, redefined-outer-name
import argparse

import pytest
from testfixtures import TempDirectory

from lorien import main
from lorien.configs import CONFIG_GROUP, make_config_parser, read_args_from_files
from lorien.logger import get_logger
from lorien.util import dump_to_yaml

log = get_logger("Unit-Test")


def test_definition():
    # Check main entry.
    assert "top" in CONFIG_GROUP

    # Check subparsers and sub-module entries.
    for parser in CONFIG_GROUP.values():
        subparsers = [p for p in parser._actions if isinstance(p, argparse._SubParsersAction)]
        if len(subparsers) == 1:
            # If the parser has a subparser, it has to be required.
            assert subparsers[0].required
        elif not subparsers:
            # If no sub-commands, then this parser has to define "entry" as the default function.
            assert parser.get_default("entry") is not None
        else:
            # Not allowed to have more than one subparsers.
            assert False


def test_config():
    with pytest.raises(SystemExit):
        make_config_parser([])

    # Test config cache
    with pytest.raises(SystemExit):
        make_config_parser([])


def test_read_args_from_files():
    args = read_args_from_files(["a", "b", "c"])
    assert len(args) == 3

    with TempDirectory() as temp_dir:
        config_file = "{}/cfg.yaml".format(temp_dir.path)
        with open(config_file, "w") as filep:
            filep.write(dump_to_yaml({"model": ["a", "b", "c"]}))
        args = read_args_from_files(["p", "@{}".format(config_file)])
        assert args == ["p", "--model", "a", "--model", "b", "--model", "c"]
