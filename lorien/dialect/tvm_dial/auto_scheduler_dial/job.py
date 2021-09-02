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
TVM auto_scheduler Job definition module.
"""
import argparse
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import PythonBasedMeasureCallback

from ....logger import get_logger
from ....tune.job import JobConfigs
from ....tune.result import TuneErrorCode
from ..job import TuneMetadata, TVMJob, TVMJobConfigs
from .result import AutoSchedulerTuneResult

log = get_logger("TVMAutoSchedulerJob")


class RecordToMetadata(PythonBasedMeasureCallback):
    """A Lorien customized callback to update the tuning metadata."""

    def __init__(self, metadata: TuneMetadata):
        """An AutoScheduler callback function to update the tuning metadata.

        Parameters
        ----------
        metadata: TuneMetadata
            The statistic information for the tuning process.
        """
        self.metadata = metadata
        super(RecordToMetadata, self).__init__()

    def callback(self, _, inputs, results):
        """A callback to log the so far best status."""
        for inp, res in zip(inputs, results):
            self.metadata.trial_count += 1
            if res.error_no == 0:
                flop_ct = inp.task.compute_dag.flop_ct
                latency = np.mean([v.value for v in res.costs])
                self.metadata.max_thrpt = max(self.metadata.max_thrpt, flop_ct / latency / 1e9)
            elif res.error_no != 1:
                # Error code 1 (task instantiation error) is not considered a failure
                # because it is caused by improper schedule configs for GPU kernels.
                self.metadata.failed_count += 1

        log.info(self.metadata.trial_count)
        print(
            "LOG_COUNT %d trials. Failed count %d. Best thrpt %.2f GFlop/s"
            % (self.metadata.trial_count, self.metadata.failed_count, self.metadata.max_thrpt)
        )


class AutoSchedulerJobConfigs(TVMJobConfigs):
    """AutoScheduler job configurations."""

    def localize(self, target: str, **kwargs):
        """Localize options on worker.

        Parameters
        ----------
        target: str
            The target string.

        **kwargs
            The kwargs of AutoTVM job configuration for updating.

            ``configs``:
                System configuration that may include RPC information (`argparse.Namespace`).
        """
        # Check TVM version and disconnect the worker if the requirement doesn't meet.
        if not self.check_tvm_build_config():
            raise RuntimeError("TVM build config between client and server are mismatching")

        tvm_target = tvm.target.Target(target)
        is_x86_target = str(tvm_target.kind.name) == "llvm" and tvm_target.keys[0] == "cpu"

        if (
            "configs" in kwargs
            and kwargs["configs"].device is not None
            and kwargs["configs"].runner_port is not None
        ):
            # TVM RPC runner. Required attributes should be checked by the client.
            device_name = kwargs["configs"].device
            runner_port = kwargs["configs"].runner_port
            runner = auto_scheduler.RPCRunner(
                device_name,
                host="0.0.0.0",  # Assume our RPC server and TVM RPC tracker are the same machine.
                port=runner_port,
                number=self.measure_options["test"],
                repeat=self.measure_options["repeat"],
                min_repeat_ms=self.measure_options["min_repeat_ms"],
                timeout=50,
                enable_cpu_cache_flush=is_x86_target,
            )
            log.info("RPCRunner: 0.0.0.0:%d - %s", runner_port, device_name)
            self.measure_options = {"runner": runner}
        else:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                number=self.measure_options["test"],
                repeat=self.measure_options["repeat"],
                min_repeat_ms=self.measure_options["min_repeat_ms"],
                enable_cpu_cache_flush=is_x86_target,
            )
            log.info("LocalRunner")
            self.measure_options = {"measure_ctx": measure_ctx}

    def __del__(self):
        """Manually delete local RPC measure context."""
        if "measure_ctx" in self.measure_options:
            del self.measure_options["measure_ctx"]


class AutoSchedulerJob(TVMJob):
    """A tuning job including a workload as well as tuning related configurations."""

    @staticmethod
    def create_job_configs(configs: argparse.Namespace) -> JobConfigs:
        """Create a JobConfigs. See `JobConfigs`.

        Parameters
        ----------
        configs: argparse.Namespace
            The system configuration of tuner.

        Returns
        -------
        job_configs: JobConfigs
            The job configurations.
        """
        job_configs = AutoSchedulerJobConfigs(configs)
        job_configs.tune_options = {"ntrial": configs.ntrial}
        job_configs.measure_options = {
            "test": configs.test,
            "repeat": configs.repeat,
            "min_repeat_ms": configs.min,
        }
        return job_configs

    def tune(
        self,
        tune_options: Dict[str, Any],
        measure_options: Dict[str, Any],
        commit_options: Optional[Dict[str, Any]] = None,
    ):
        """Tune the job with the given configuration and update the result.
        If the commit options are provided, then this function also in charge of
        committing the tuning results, or the job manager will commit the result, otherwise.
        """

        workload = self.workload
        self.result = AutoSchedulerTuneResult()

        try:
            task = self.workload.to_task()
        except RuntimeError as err:
            self.result.error_code = TuneErrorCode.FAIL_TO_CREATE_TASK
            self.result.error_msgs.append(str(err))
            return

        log_file_name = workload.get_log_file_name()

        # Create a local folder to temporary keep tuning logs.
        work_dir = tempfile.mkdtemp(
            prefix="lorien-tune-log-",
            dir=tune_options["tune_dir"] if "tune_dir" in tune_options else ".",
        )
        log_file_path = os.path.join(work_dir, log_file_name)

        log.info("Tuning workload %s and save log to %s", str(workload), log_file_name)

        metadata = TuneMetadata()

        runner = (
            measure_options["runner"]
            if "runner" in measure_options
            else measure_options["measure_ctx"].runner
        )

        auto_scheduler_options = auto_scheduler.TuningOptions(
            num_measure_trials=int(tune_options["ntrial"]),
            runner=runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(log_file_path),
                RecordToMetadata(metadata),
            ],
        )
        task.tune(auto_scheduler_options)
        self.result.log_file = log_file_path

        # No valid results.
        if metadata.trial_count == metadata.failed_count:
            self.result.error_code = TuneErrorCode.NO_VALID_RESULT
            return

        # Skip result committing. Send all tuning logs back.
        if commit_options is None:
            log.info("No commit options specified. Results should be committed by master")
            # Load tuning log
            with open(self.result.log_file, "r") as filep:
                self.result.metadata["tune_logs"] = filep.read()
            return

        self.result.commit(commit_options, workload=workload)

        # Failed to commit results so send all back to the manager.
        if self.result.error_code == TuneErrorCode.STORAGE_ERROR:
            with open(log_file_path, "r") as filep:
                self.result.metadata["tune_logs"] = filep.read()
        return
