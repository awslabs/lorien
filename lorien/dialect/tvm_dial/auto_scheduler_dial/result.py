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
Tuning result of auto_scheduler dialect.
"""
import heapq
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureInput, MeasureResult
from tvm.auto_scheduler.measure_record import dump_record_to_string, load_record_from_string

from ....logger import get_logger
from ....tune.result import TuneResult
from ....workload import Workload
from ..result import TVMRecords
from ..util import gen_target_id_keys, get_canonical_tvm_target_str
from .workload import AutoSchedulerWorkload

log = get_logger("AutoScheduler-Result")


class AutoSchedulerRecords(TVMRecords[Tuple[MeasureInput, MeasureResult]]):
    """The container to maintain the records of an auto_scheduler task."""

    def __init__(self, target_key: str, workload_key: Optional[str] = None):
        """Initialize records.

        Parameters
        ----------
        target_key: str
            The target key (partition key) of the workload.

        workload_key: Optional[str]
            The workload key (sort key) of the workload. If not presneted,
            then all records with the same target_key will be fetched when querying.
        """
        target_key = get_canonical_tvm_target_str(target_key, remove_libs=True)
        alter_key = gen_target_id_keys(target_key)
        super(AutoSchedulerRecords, self).__init__(target_key, alter_key, workload_key)
        self._data: List[Tuple[float, float, Tuple[MeasureInput, MeasureResult]]] = []

    @staticmethod
    def encode(record: Tuple[MeasureInput, MeasureResult]) -> str:
        """Encode a record to a string."""
        return dump_record_to_string(*record)

    @staticmethod
    def decode(record_str: str) -> Tuple[MeasureInput, MeasureResult]:
        """Decode a string to a record."""
        return load_record_from_string(record_str)

    def gen_task_item(self) -> Dict[str, Any]:
        """No additional attribute is required for auto_scheduler."""
        return {}

    @staticmethod
    def gen_record_item(record: Tuple[MeasureInput, MeasureResult]):
        """Generate an item for a record that can be appended to the task item."""
        return {"latency": np.mean([v.value for v in record[1].costs])}

    def push(self, record: Tuple[MeasureInput, MeasureResult]):
        """Push a new record.

        Parameters
        ----------
        record: Any
            The record to be pushed.
        """
        # Push with -cost as heapq is min-heap as we want the worst record on the top.
        heapq.heappush(
            self._data, (-np.mean([v.value for v in record[1].costs]), record[1].timestamp, record)
        )

    def pop(self) -> Tuple[MeasureInput, MeasureResult]:
        """Pop the worst record in the container and remove the cost."""
        return heapq.heappop(self._data)[2]

    def peak(self) -> Tuple[MeasureInput, MeasureResult]:
        """Peak the first record."""
        assert self._data
        return self._data[0][2]

    def to_list(self, nbest: int = -1) -> List[Tuple[MeasureInput, MeasureResult]]:
        """Sort the record (of any layout) to be a list and return the best N.

        Parameters
        ----------
        nbest: int
            The best N records to be returned. Default to return all.

        Returns
        -------
        records: List[Tuple[MeasureInput, MeasureResult]]
            The sorted list of records.
        """
        nbest = nbest if nbest != -1 else len(self._data)
        return [item[2] for item in heapq.nsmallest(nbest, self._data)]

    def __len__(self) -> int:
        return len(self._data)


class AutoSchedulerTuneResult(TuneResult):
    """The result of a tuning job."""

    @staticmethod
    def create_records_by_workloads(
        log_file_path: str, nbest: int, workload: Optional[Workload] = None
    ) -> Sequence[AutoSchedulerRecords]:
        """Parse records from the tuning log and group them by workloads.

        Parameters
        ----------
        log_file_path: str
            The log file path.

        nbest: int
            The maximum number of best records to be kept.

        workload: Optional[Workload]
            The target workload. If presented, the returnede map will only have one entry.

        Returns
        -------
        records: Sequence[AutoSchedulerRecords]
            A list of created records.
        """
        target_workload_key = workload.get_workload_key() if workload is not None else None

        best_records: Dict[str, AutoSchedulerRecords] = {}
        with open(log_file_path, "r") as filep:
            for line in filep:
                if line[0] == "#" or line[0] == " " or line[0] == "\n":
                    continue

                inp, res = load_record_from_string(line)
                workload_key = AutoSchedulerWorkload.from_task(inp.task).get_workload_key()
                if target_workload_key is not None and workload_key != target_workload_key:
                    continue

                if workload_key not in best_records:
                    best_records[workload_key] = AutoSchedulerRecords(inp.task.target, workload_key)
                curr_records = best_records[workload_key]

                if res.error_no != MeasureErrorNo.NO_ERROR:
                    continue

                curr_records.push((inp, res))
                if len(curr_records) > nbest:
                    curr_records.pop()
        return list(best_records.values())

    @staticmethod
    def gen_features(log_file_path: str, out_path: str):
        """Featurize tuning logs to be input features of the performance cost model.

        Parameters
        ----------
        log_file_path: str
            The log file path. It can be a file path for a single file, or a directory of
            several log files.

        out_path: str
            The path to write generated features and artifacts.
        """
        raise RuntimeError("Feature extraction is not supported yet in AutoScheduler dialect")
