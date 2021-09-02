"""
The unit test module for tuning master.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import
import argparse
import tempfile

import pytest
from lorien import main  # pylint: disable=unused-import
from lorien.configs import make_config_parser
from lorien.tune.master import run, run_job_manager
from lorien.util import dump_to_yaml, upload_s3_file

from .common import gen_jobs, gen_workloads, mock_s3_client


def test_run(mocker, mock_s3_client):
    # Failed to load workloads
    with pytest.raises(RuntimeError):
        configs = argparse.Namespace(workload=["not/exist/path/a.json"])
        run(configs)

    argv = ["tune"]
    for wkl in gen_workloads(0, 5):
        argv += ["--workload", dump_to_yaml(wkl)]

    for job in gen_jobs(5, 10):
        argv += ["--job", dump_to_yaml(job)]

    job = gen_jobs(10, 11)[0]
    with tempfile.NamedTemporaryFile(mode="w", prefix="lorien-test-tune-") as temp_file:
        temp_file.write(dump_to_yaml(job))
        temp_file.flush()
        upload_s3_file(temp_file.name, "s3://unit-test-bucket/job.yaml")

        argv += ["--job", "s3://unit-test-bucket/job.yaml"]

        def mock_run_job_manager(_, packed_args):
            return packed_args

        mocker.patch("lorien.tune.master.run_job_manager").side_effect = mock_run_job_manager

        # No job manager config
        configs = make_config_parser(argv)
        assert not run(configs)

        batch_cfg_str = dump_to_yaml({"target": "llvm"})
        for mgr_cfg in [["--batch", batch_cfg_str], ["--local", "llvm"]]:
            upload_s3_file(temp_file.name, "s3://unit-test-bucket/job.yaml")
            configs = make_config_parser(argv + mgr_cfg)
            packed_args = run(configs)

            assert packed_args["target"] == "llvm"
            assert len(packed_args["jobs"]) == 11

        # Conflict job managers
        with pytest.raises(RuntimeError):
            upload_s3_file(temp_file.name, "s3://unit-test-bucket/job.yaml")
            configs = make_config_parser(argv + ["--batch", batch_cfg_str, "--local", "llvm"])
            run(configs)


def test_run_job_manager():
    class FakeManager1:
        def __init__(self, **args):
            raise RuntimeError

    class FakeManager2:
        def __init__(self, **args):
            pass

        def tune(self):
            return 1

    assert not run_job_manager(FakeManager1, {})
    assert run_job_manager(FakeManager2, {}) == 1
