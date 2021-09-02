"""
The unit test for generator.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name, unused-argument
import argparse
import os
import tempfile

from lorien.generate import gen
from lorien.util import load_from_yaml

from .common import gen_workloads


def test_gen():
    def test_generator(configs):
        return gen_workloads(0, 5, configs.target)

    with tempfile.TemporaryDirectory(prefix="lorien_test_gen_") as temp_dir:
        wkl_file = os.path.join(temp_dir, "test_workloads.yaml")
        configs = argparse.Namespace(target="llvm", output=wkl_file)

        gen(test_generator)(configs)
        assert os.path.exists(wkl_file)
        with open(wkl_file, "r") as filep:
            workloads = load_from_yaml(filep.read())
            assert len(workloads["workload"]) == 5
            assert all([load_from_yaml(wkl).target == "llvm" for wkl in workloads["workload"]])
