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
Tuning result of AutoTVM dialect.
"""
import fnmatch
import glob
import json
import heapq
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Set, Sequence, Tuple

import numpy as np
import tqdm
from filelock import FileLock

from tvm.autotvm.measure import MeasureInput, MeasureResult
from tvm.autotvm.record import decode as autotvm_decode
from tvm.autotvm.record import encode as autotvm_encode
from tvm.autotvm.record import load_from_file
from tvm.autotvm.task import create, serialize_args, Task
from tvm.autotvm.task.space import AnnotateEntity, OtherOptionEntity, ReorderEntity, SplitEntity

from ....database.util import convert_to_db_list
from ....logger import get_logger
from ....tune.result import TuneResult
from ....workload import Workload
from ..result import TVMRecords
from ..util import TASK_TO_REQUIRED_LIBS, gen_target_id_keys, get_canonical_tvm_target_str
from .util import infer_task_layout
from .workload import AutoTVMWorkload

log = get_logger("AutoTVM-Result")


class AutoTVMRecords(TVMRecords[Tuple[MeasureInput, MeasureResult]]):
    """The container to maintain the records of a tuning task by data layout."""

    def __init__(
        self, target_key: str, workload_key: Optional[str] = None, group_by_layout: bool = False
    ):
        """Initialize records.

        Parameters
        ----------
        target_key: str
            The target key (partition key) of the workload.

        workload_key: Optional[str]
            The workload key (sort key) of the workload. If not presneted,
            then all records with the same target_key will be fetched when querying.

        group_by_layout: bool
            Whether to group records by layout. If enabled, we only maintain the
            best record of each layout.
        """
        self.group_by_layout = group_by_layout
        target_key = get_canonical_tvm_target_str(target_key, remove_libs=True)
        alter_key = gen_target_id_keys(target_key)
        super(AutoTVMRecords, self).__init__(target_key, alter_key, workload_key)

        # Map from layout to records.
        self._data: Dict[Any, List[Tuple[float, float, Tuple[MeasureInput, MeasureResult]]]] = {}
        self._task_str: str = ""

    @staticmethod
    def encode(record: Tuple[MeasureInput, MeasureResult]) -> str:
        """Encode a record to a string."""
        return autotvm_encode(*record)

    @staticmethod
    def decode(record_str: str) -> Tuple[MeasureInput, MeasureResult]:
        """Decode a string to a record."""
        return autotvm_decode(record_str)

    def gen_task_item(self) -> Dict[str, Any]:
        """Generate an item that can be committed to the database. Note that since all records
        in this container should be for the same task, they should be in the same task item.
        """
        task = self.peak()[0].task
        task_name = task.name
        item = {
            "OpName": {"S": task.workload[0]},
            "TaskName": {"S": task_name},
            "Args": convert_to_db_list(task.args),
        }
        if task_name in TASK_TO_REQUIRED_LIBS:
            item["RequiredLib"] = {"S": TASK_TO_REQUIRED_LIBS[task_name]}
        return item

    @staticmethod
    def gen_record_item(record: Tuple[MeasureInput, MeasureResult]):
        """Generate an item for a record that can be appended to the task item."""
        return {"latency": np.mean(record[1].costs)}

    def push(self, record: Tuple[MeasureInput, MeasureResult]):
        """Push a new record to the bucket based on its layout.

        Parameters
        ----------
        record: Tuple[MeasureInput, MeasureResult]
            The record to be pushed.
        """
        if not self._task_str:
            self._task_str = str(record[0].task)

        layout = infer_task_layout(record) if self.group_by_layout else None
        if layout not in self._data:
            self._data[layout] = []

        # Push with -cost as heapq is min-heap as we want the worst record on the top.
        heapq.heappush(
            self._data[layout],
            (-np.mean(record[1].costs), record[1].timestamp, record),
        )

        if self.group_by_layout:
            # Only keep the best record for each layout.
            if len(self._data[layout]) > 1:
                heapq.heappop(self._data[layout])

    def pop(self) -> Tuple[MeasureInput, MeasureResult]:
        """Pop the worst record in the container. We do not need this API for AutoTVM."""
        raise NotImplementedError

    def peak(self) -> Tuple[MeasureInput, MeasureResult]:
        """Peak the first record (of any layout)."""
        assert self._data
        return list(self._data.values())[0][0][2]

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
        all_records: List[Tuple[float, float, Tuple[MeasureInput, MeasureResult]]] = []
        for records in self._data.values():
            all_records += records

        nbest = len(all_records) if nbest == -1 else nbest
        best_records = sorted(all_records, key=lambda r: r[0])[:nbest]
        return [r[2] for r in best_records]

    def __len__(self) -> int:
        return sum([len(v) for v in self._data.values()])


class AutoTVMTuneResult(TuneResult):
    """The result of a tuning job."""

    @staticmethod
    def create_records_by_workloads(
        log_file_path: str, nbest: int, workload: Optional[Workload] = None
    ) -> Sequence[TVMRecords]:
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
        records: Sequence[AutoTVMRecords]
            A list of created records.
        """
        target_workload_key = workload.get_workload_key() if workload is not None else None

        # Parse the records and group by tasks.
        best_records: Dict[str, AutoTVMRecords] = {}
        for record in load_from_file(log_file_path):
            if record[1].error_no != 0:  # Ignore invalid records.
                continue

            task = record[0].task
            task.target = record[0].target
            workload_key = AutoTVMWorkload.from_task(record[0].task).get_workload_key()
            if target_workload_key is not None and workload_key != target_workload_key:
                continue

            if workload_key not in best_records:
                best_records[workload_key] = AutoTVMRecords(
                    task.target, workload_key, group_by_layout=True
                )
            best_records[workload_key].push(record)

        return list(best_records.values())

    @staticmethod
    def extract_feature(inp):
        """Extract features from a measure input.

        Parameters
        ----------
        inp: MeasureInput
            AutoTVM measure input.

        Returns
        -------
        features: Dict[str, Any]
            A map from feature name to value.
        """
        features = {}

        name_n_idxs = [["attr", 0], ["in", 0]]

        def parse_arg(arg, attr_or_in):
            """A helper function to parse an argument."""
            arg_features = {}
            if isinstance(arg, (list, tuple)):
                for elt in arg:
                    arg_features.update(parse_arg(elt, attr_or_in))
            else:
                arg_features["{0}_{1}".format(*name_n_idxs[attr_or_in])] = arg
                name_n_idxs[attr_or_in][1] += 1
            return arg_features

        # Feature arguments
        for arg in inp.task.args:
            if isinstance(arg, (list, tuple)) and arg[0] == "TENSOR":
                features.update(parse_arg(arg[1:], 1))
            else:
                features.update(parse_arg(arg, 0))

        # Feature configs
        for key, val in inp.config._entity_map.items():  # pylint: disable=protected-access
            if isinstance(val, SplitEntity):
                for idx, elt in enumerate(val.size):
                    features["sp_{0}_{1}".format(key, idx)] = elt
            elif isinstance(val, AnnotateEntity):
                pval = None
                if isinstance(val.anns, (int, float)):
                    pval = str(val.anns)
                elif isinstance(val.anns, str):
                    pval = val.anns
                elif isinstance(val.anns, (list, tuple)):
                    pval = ";".join([str(e) for e in val.anns])
                assert pval is not None, "Unrecognized annotate type: %s" % type(val.anns)
                features["an_{0}".format(key)] = pval
            elif isinstance(val, OtherOptionEntity):
                features["ot_{0}".format(key)] = val.val
            elif isinstance(val, ReorderEntity):
                features["re_{0}".format(key)] = ";".join([str(a) for a in val.perm])
            else:
                raise RuntimeError("Unsupported config entity: %s" % val)

        return features

    @staticmethod
    def extract_feature_from_file(log_file_path: str, out_path: str):
        """Featurize tuning results in a log file and write to a file.

        Parameters
        ----------
        log_file_path: str
            The log file path.

        out_path: str
            The path to write generated features.
        """

        def gen_key_from_measure_input(inp):
            inp_key = "{0}-{1}".format(
                str(inp.task), str(inp.target).replace(" ", "").replace("=", "-")
            )
            target_key = str(inp.target).replace(" ", "").replace("=", "-")
            file_key = inp.task.name
            return (inp_key, target_key, file_key)

        data: Dict[Tuple[str, str, str], List[str]] = {}
        cached_tasks: Dict[Tuple[str, Tuple[Any], str], Task] = {}
        for inp, res in load_from_file(log_file_path):
            key = gen_key_from_measure_input(inp)
            if key not in data:
                data[key] = []

            try:
                features = AutoTVMTuneResult.extract_feature(inp)
            except Exception as err:  # pylint: disable=broad-except
                log.warning("Failed to extract features from %s: %s", str(inp), str(err))
                continue

            # Use the FLOPS in AutoTVM task to compute throughput. Note that we use a cache
            # to avoid time-consuming AutoTVM task creation.
            task_cache_key = (inp.task.name, serialize_args(inp.task.args), str(inp.target))
            if task_cache_key not in cached_tasks:
                cached_tasks[task_cache_key] = create(inp.task.name, inp.task.args, inp.target)

            task = cached_tasks[task_cache_key]
            if res.error_no == 0:
                features["thrpt"] = np.around(task.flop / 1e9 / np.mean(res.costs), 2).tolist()
            else:
                features["thrpt"] = 0

            data[key].append(json.dumps(features))

        for (_, target_key, file_key), feats in data.items():
            if not os.path.exists(os.path.join(out_path, target_key)):
                os.mkdir(os.path.join(out_path, target_key))
            out_file = "{0}/{1}/{2}.json".format(out_path, target_key, file_key)
            lock_file = "{0}.lock".format(out_file)
            with FileLock(lock_file):
                with open(out_file, "a") as filep:
                    for record in feats:
                        filep.write(record)
                        filep.write("\n")

    @staticmethod
    def gen_features(log_file_path: str, out_path: str):
        """Featurize tuning logs to be input features of the performance cost model.
        The process includes two phases. The first phase aggregates the tuning logs
        by AutoTVM templates (e.g., conv2d_nchw, conv2d_winograd, etc) and extracts
        features from the workload (e.g., shapes, attribtues, schedule configurations).
        The output of phase 1 is few JSON files. Since the number of AutoTVM templates
        is not many, phase 1 usually only outputs ~10 files. Then, phase 2 reads
        the JSON file one-by-one and does 3 tasks: 1) normalize numeric features,
        2) enumerate string features, 3) output processed features to .csv files
        along with the metadata (e.g., mean and std of numeric features and
        the mapping of string features). Although writing the phase 1 output to files
        may slow down the process due to disk I/O, it avoids out-of-memory issue
        when processing a large number of tuning logs.

        Parameters
        ----------
        log_file_path: str
            The log file path. It can be a file path for a single file, or a directory of
            several log files.

        out_path: str
            The path to write generated features and artifacts.
        """
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # Phase 1: Extract features for each log file and generate JSON files.
        phase1_path = os.path.join(out_path, "_feature_temp")
        if not os.path.exists(phase1_path):
            os.mkdir(phase1_path)

        n_workers = cpu_count() // 2
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            path = "{0}/*".format(log_file_path) if os.path.isdir(log_file_path) else log_file_path
            file_list = glob.glob(path)
            log.info(
                "Extracting features from %d tuning logs using %d workers",
                len(file_list),
                n_workers,
            )
            for start in tqdm.tqdm(range(0, len(file_list), n_workers)):
                futures = [
                    pool.submit(AutoTVMTuneResult.extract_feature_from_file, log_file, phase1_path)
                    for log_file in file_list[start : min(start + n_workers, len(file_list))]
                ]
                for _ in as_completed(futures):
                    continue

        # Remove lock files
        for root_dir, _, file_names in os.walk(phase1_path):
            for file_name in fnmatch.filter(file_names, "*.lock"):
                os.remove(os.path.join(root_dir, file_name))

        # Phase 2: Load and process the JSON files (e.g., normalization) to csv files.
        for json_file in glob.glob("{0}/*/*.json".format(phase1_path)):
            timer = time.time()
            feature_name_set: Set[str] = set()

            with open(json_file, "r") as filep:
                data = [json.loads(r) for r in filep]

            # The first pass to get a super set of features.
            for json_record in data:
                feature_name_set.update(json_record.keys())

            # Turn to a list and enforce thrpt (result) to be the last.
            feature_name_set.remove("thrpt")
            feature_names = list(feature_name_set) + ["thrpt"]

            # The second pass fills out the data to a data frame.
            feature_data = []
            for json_record in data:
                csv_record = []
                for feature_name in feature_names:
                    val = (
                        str(None)
                        if feature_name not in json_record
                        else str(json_record[feature_name])
                    )
                    csv_record.append(val)
                feature_data.append(csv_record)

            # Metadata for mean, std, and category mapping.
            meta_file = os.path.join(
                out_path, os.path.basename(json_file).replace(".json", ".meta")
            )
            with open(meta_file, "w") as filep:
                std_data = []

                # Transpose the feature data so that each row is the data of a feature.
                # pylint: disable=unsubscriptable-object
                tran_feature_data = np.array(feature_data).T
                for feature_name, row in zip(feature_names[:-1], tran_feature_data[:-1]):
                    val_type = "numeric"
                    try:  # Standardize floating values.
                        float_row = row.astype("float")
                        meta = [float_row.mean(), float_row.std()]
                        meta[1] = 1 if meta[1] == 0 else meta[1]  # Workaround for std=0
                        std_data.append((float_row - meta[0]) / meta[1])
                    except ValueError:  # String to index transformation.
                        meta = np.unique(row)
                        cate_map = {c: i for i, c in enumerate(meta)}
                        std_data.append([cate_map[c] for c in row])
                        val_type = "category"

                    filep.write("{},".format(feature_name))  # Feature name
                    filep.write("{},".format(val_type))  # Feature value type (numeric or category)
                    filep.write(",".join([str(e) for e in meta]))
                    filep.write("\n")

                std_data.append(tran_feature_data[-1].astype("float32"))

            # pylint: disable=unsubscriptable-object
            out_feature_data: np.ndarray = np.array(std_data).T

            # Write to file
            file_name = os.path.basename(json_file).replace(".json", ".csv")
            with open(os.path.join(out_path, file_name), "w") as filep:
                # Write titles
                filep.write(",".join(feature_names))
                filep.write("\n")

                # Write features and results
                # False positive pylint error: https://github.com/PyCQA/pylint/issues/3387
                for record in out_feature_data:  # pylint: disable=not-an-iterable
                    filep.write(",".join([str(r) for r in record]))
                    filep.write("\n")

            log.info("Processing %s ...done, %d sec", json_file, int(time.time() - timer))
