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
A module of tuning results.
"""
import os
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union

import boto3
from ruamel.yaml import YAML, yaml_object

from ..database.util import convert_to_db_list, convert_to_list
from ..logger import get_logger
from ..util import serialize_framework_build_config, split_s3_path, upload_s3_file
from ..workload import Workload

log = get_logger("Result")

# The maximum number of configs for an item in the database.
MAX_CONFIG_PER_ITEM = 100

RecordType = TypeVar("RecordType")


@yaml_object(YAML())
class TuneErrorCode(Enum):
    """Error code for the tuning."""

    NORMAL = 0
    NO_VALID_RESULT = 1  # Do not find any valid schedule config.
    FAIL_TO_SUBMIT = 2  # Fail to submit the workload for tuning.
    FAIL_TO_LOAD_WORKLOAD = 3  # Fail to load the workload from a JSON string.
    FAIL_TO_CREATE_TASK = 4  # Fail to create an AutoTVM task from a workload.
    FAIL_TO_GET_RESULT = 5  # Fail to retrieve results from worker.
    STORAGE_ERROR = 6  # Database or S3 bucket related errors.

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar("!TuneErrorCode", "%s" % node._value_)

    @classmethod
    def from_yaml(cls, _, node):
        return cls(int(node.value))


class Records(Generic[RecordType]):
    """The container to maintain the records of a workload."""

    def __init__(
        self, target_key: str, alter_key: Optional[str] = None, workload_key: Optional[str] = None
    ):
        """Initialize records.

        Parameters
        ----------
        target_key: str
            The target key (partition key) of the workload.

        alter_key: Optional[str]
            The alternative target key (global secondary key) of the workload.
            If not presented, querying with alternative target key is unavailable.

        workload_key: Optional[str]
            The workload key (sort key) of the workload. If not presneted,
            then all records with the same target_key will be fetched when querying.
        """
        self._target_key = target_key
        self._alter_key = alter_key
        self._workload_key = workload_key

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def alter_key(self) -> Optional[str]:
        return self._alter_key

    @property
    def workload_key(self) -> Optional[str]:
        return self._workload_key

    def get_framework_build_config(self) -> Optional[Dict[str, str]]:
        """Get the framework build configurations that generate these records.
        If None, then the committed records will not have this information.
        """
        raise NotImplementedError

    @staticmethod
    def encode(record: RecordType) -> str:
        """Encode a record to a string."""
        raise NotImplementedError

    @staticmethod
    def decode(record_str: str) -> RecordType:
        """Decode a string to a record."""
        raise NotImplementedError

    def gen_task_item(self) -> Dict[str, Any]:
        """Generate an item that can be committed to the database. Note that since all records
        in this container should be for the same task, they should be in the same task item.
        """
        raise NotImplementedError

    @staticmethod
    def gen_record_item(record: RecordType):
        """Generate an item for a record that can be appended to the task item."""
        raise NotImplementedError

    def push(self, record: RecordType):
        """Push a new record.

        Parameters
        ----------
        record: Any
            The record to be pushed.
        """
        raise NotImplementedError

    def pop(self) -> RecordType:
        """Pop the worst record in the container."""
        raise NotImplementedError

    def peak(self) -> RecordType:
        """Peak the first record."""
        raise NotImplementedError

    def to_list(self, nbest: int = -1) -> List[RecordType]:
        """Sort the record (of any layout) to be a list and return the best N.

        Parameters
        ----------
        nbest: int
            The best N records to be returned. Default to return all.

        Returns
        -------
        records: List[RecordType]
            The sorted list of records.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def commit(self, table_name: str, nbest: int, log_file_name: Optional[str] = None, **db_kwargs):
        """Commit the records to the database.
        The final list in the DB will have at maximum nbest configs.
        If the workload has old configs, then it might have at most MAX_CONFIG_PER_ITEM configs
        with new and old configs.

        Parameters
        ----------
        table_name: str
            The table name to commit.

        nbest: int
            The maximum number of best records to be committed.

        log_file_name: Optional[str]
            The tuning log name contains this record.

        **db_kwargs
            The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
            or "endpoint_url=http://localhost:8000".
        """
        assert nbest <= MAX_CONFIG_PER_ITEM, '"nbest" exceeds MAX_CONFIG_PER_ITEM ({})'.format(
            MAX_CONFIG_PER_ITEM
        )

        if self.workload_key is None or self.alter_key is None:
            raise RuntimeError("Commmit with workload_key=None or alter_key=None is not allowed")

        # DynamoDB does not accept "None" so we change it to "unknown".
        framework_build_config = self.get_framework_build_config()
        build_config: Union[Dict[str, str], str] = "unknown"
        if framework_build_config is not None:
            build_config = framework_build_config
        serialized_build_config = serialize_framework_build_config(build_config)

        # Configs by other framework build commits that we do not want to touch.
        other_record_items = []

        # Try to query the item to see if it exists already or not.
        key_condition_expr = "Target = :target AND PrimaryRangeKey = :key"
        expr_vals = {":target": {"S": self.target_key}, ":key": {"S": self.workload_key}}
        query_options = {
            "TableName": table_name,
            "KeyConditionExpression": key_condition_expr,
            "ExpressionAttributeValues": expr_vals,
            "ProjectionExpression": "BestConfigs",
        }

        try:
            # Item already exists. Query existing configs and merge to the new oens.
            client = boto3.client("dynamodb", **db_kwargs)
            result = client.query(**query_options)

            # Push existing configs with the same framework built commit to the heap.
            for exist_config in convert_to_list(result["Items"][0]["BestConfigs"]):
                if serialized_build_config != serialize_framework_build_config(
                    exist_config["framework_build_config"]
                ):
                    other_record_items.append(exist_config)
                else:
                    record = self.decode(exist_config["config"])
                    if record is None:
                        continue
                    self.push(record)
        except Exception as err:  # pylint:disable=broad-except
            # Item does not exist. First commit a new one without configs.
            item = {
                "Target": {"S": self.target_key},
                "TargetIDKeys": {"S": self.alter_key},
                "PrimaryRangeKey": {"S": self.workload_key},
            }
            item.update(self.gen_task_item())

            try:
                client = boto3.client("dynamodb", **db_kwargs)
                client.put_item(TableName=table_name, Item=item)
            except Exception as err:  # pylint:disable=broad-except
                raise RuntimeError("Failed to commit: %s" % str(err))

        # Create commit items.
        record_items = []
        for record in self.to_list(nbest):
            record_item = {
                "config": self.encode(record),
                "log_path": log_file_name if log_file_name is not None else " ",
                "framework_build_config": build_config,
            }
            record_item.update(self.gen_record_item(record))
            record_items.append(record_item)

        # Remove the existing old configs if we already have too many. This can not only maintain
        # the DB table size, but also avoid the DynamoDB item size limitation (400Kb) error.
        expected_configs = len(record_items) + len(other_record_items)
        other_record_items = (
            other_record_items[: MAX_CONFIG_PER_ITEM - len(record_items)]
            if expected_configs > MAX_CONFIG_PER_ITEM
            else other_record_items
        )

        # Update the best config to the exist record.
        try:
            update_expr = "SET BestConfigs = :config"
            update_expr_vals = {":config": convert_to_db_list(record_items + other_record_items)}
            client.update_item(
                TableName=table_name,
                Key={"Target": {"S": self.target_key}, "PrimaryRangeKey": {"S": self.workload_key}},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=update_expr_vals,
            )
        except Exception as err:  # pylint:disable=broad-except
            raise RuntimeError("Failed to append the best config: %s" % str(err))

    def query(
        self,
        table_name: str,
        use_alter_key: bool = False,
        **db_kwargs,
    ):
        """Querying the DB and update records.

        Parameters
        ----------
        table_name: str
            The table name to be queried.

        use_alter_key: bool
            Use the alternative key instead of target key to query.

        **db_kwargs
            The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
            or "endpoint_url=http://localhost:8000".

        Returns
        -------
        items: Generator
            The best configs for the given target and primary key.
        """

        # Generate query
        query_options: Dict[str, Any] = {
            "TableName": table_name,
            "ProjectionExpression": "BestConfigs",
        }

        query_key = ""
        if use_alter_key:
            if self.alter_key is None:
                raise RuntimeError("Cannnot query with alter_key is None")
            query_options["IndexName"] = "TargetIDKeysIndex"
            key_condition_expr = "TargetIDKeys = :target"
            expr_vals = {":target": {"S": self.alter_key}}
            query_key = self.alter_key
        else:
            key_condition_expr = "Target = :target"
            expr_vals = {":target": {"S": self.target_key}}
            query_key = self.target_key

        if self.workload_key is not None:
            key_condition_expr += " AND begins_with(PrimaryRangeKey, :key)"
            expr_vals[":key"] = {"S": self.workload_key}
            query_key = self.workload_key

        query_options["KeyConditionExpression"] = key_condition_expr
        query_options["ExpressionAttributeValues"] = expr_vals

        try:
            client = boto3.client("dynamodb", **db_kwargs)
            last_evaluate_key = None
            while True:
                if last_evaluate_key is None:
                    result = client.query(**query_options)
                else:
                    result = client.query(
                        ExclusiveStartKey=result["LastEvaluatedKey"], **query_options
                    )
                for item in result["Items"]:
                    for best_config in convert_to_list(item["BestConfigs"]):
                        self.push(self.decode(best_config["config"]))

                if "LastEvaluatedKey" not in result:
                    break
                last_evaluate_key = result["LastEvaluatedKey"]
        except Exception as err:  # pylint:disable=broad-except
            raise RuntimeError("Failed to query %s from %s: %s" % (query_key, table_name, str(err)))


@yaml_object(YAML())
class TuneResult:
    """The result of a tuning job."""

    def __init__(self):
        self.error_code = TuneErrorCode.NORMAL
        self.error_msgs: List[str] = []
        self.log_file: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self):
        return "%s(error_code=%r,error_msgs=%r,metadata=%r)" % (
            self.__class__.__name__,
            self.error_code,
            self.error_msgs,
            self.metadata,
        )

    @staticmethod
    def create_records_by_workloads(
        log_file_path: str, nbest: int, workload: Optional[Workload] = None
    ) -> Sequence[Records]:
        """Parse the tuning log and create records by workloads.

        Parameters
        ----------
        log_file_path: str
            The log file path.

        nbest: int
            The maximum number of best records to be kept.

        workload: Optional[Workload]
            The target workload. If presented, the returned map will only have one entry.

        Returns
        -------
        records: Sequence[Records]
            A list of created records.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def commit(
        self,
        commit_options: Dict[str, Any],
        workload: Optional[Workload] = None,
        silent: bool = False,
    ):
        """Commit 1) a full tuning log to S3 bucket and 2) the best config to DynamoDB.
        If anything goes wrong, result error code will be updated and the error message
        will be attached.

        Parameters
        ----------
        commit_options: Dict[str, Any]
            Result committing related options.

            ``db``:
                DynamoDB options (`Dict[str, str]`).
            ``table-name``:
                DynamoDB table name (`str`).
            ``nbest``:
                The number of the top configs we will commit to DB (`int`).
            ``commit-workload``:
                Commit workload to the workload table in the DB (`bool`).
            ``commit-log``:
                A S3 folder path to upload a full tuning log. The log file will be removed
                after tuning if this field is unset (`Optional[str]`).

        workload: Optional[Workload]
            The corresponding workload. If not presented, we attempt to generate workloads from
            the tuning logs.

        silent: bool
            Disable logging if True. Default False.
        """
        if self.log_file is None or not os.path.exists(self.log_file):
            raise RuntimeError("log file is missing in the TuneResult")

        # Upload tuning log to S3 bucket.
        s3_log_file: Optional[str] = None
        if commit_options["commit-log"] is not None:
            file_name = os.path.basename(self.log_file)
            bucket_name, folder_path = split_s3_path(commit_options["commit-log"])
            s3_log_file = "s3://{0}/{1}/{2}".format(bucket_name, folder_path, file_name)
            err_msg = upload_s3_file(self.log_file, s3_log_file)
            if not err_msg and not silent:
                log.info("Full tuning log has been uploaded to %s", s3_log_file)
            elif err_msg:
                if not silent:
                    log.warning(err_msg)
                self.error_code = TuneErrorCode.STORAGE_ERROR
                self.error_msgs.append(err_msg)

        # Submit results to DB.
        try:
            self.commit_tuning_log(
                workload=workload,
                log_file_path=self.log_file,
                table_name=commit_options["table-name"],
                nbest=commit_options["nbest"],
                s3_log_file_path=s3_log_file,
                **commit_options["db"],
            )
            if not silent:
                log.info("Results are submitted to the database")
        except Exception as err:  # pylint:disable=broad-except
            self.error_code = TuneErrorCode.STORAGE_ERROR
            msg = "Failed to commit result: {} ".format(str(err))
            self.error_msgs.append(msg)
            if not silent:
                log.warning(msg)

    def commit_tuning_log(
        self,
        workload: Optional[Workload],
        log_file_path: str,
        table_name: str,
        nbest: int = 20,
        s3_log_file_path: Optional[str] = None,
        **db_kwargs,
    ):
        """Commit tuning results to the database.

        Parameters
        ----------
        workload: Optional[Workload]
            The corresponding workload. If not presented, we attempt to generate workloads from
            the tuning logs.

        log_file_path: str
            The path of log file for the task.

        table_name: str
            The table name to commit.

        nbest: int
            Number of the best schedule to be committed.

        s3_log_file_path: Optional[str]
            The full log file path in S3. We will not access S3 in this function but only store
            the path along with the item in DynamoDB for future reference.

        **db_kwargs
            The kwargs of boto3 client. Commonly used: "region_name='us-west-1'"
            or "endpoint_url=http://localhost:8000".
        """
        assert os.path.exists(log_file_path), "Tuning log not found: %s" % log_file_path

        best_records: Sequence[Records] = self.create_records_by_workloads(
            log_file_path, nbest, workload
        )

        # Commit the best record of each task.
        commit_count = 0
        for records in best_records:
            if not records:
                log.warning(
                    "Workload with key %s has no valid record to be committed", records.workload_key
                )
                continue
            records.commit(
                log_file_name=s3_log_file_path, table_name=table_name, nbest=nbest, **db_kwargs
            )
            commit_count += 1

        if not best_records or not commit_count:
            raise RuntimeError("No valid record in the log file %s" % log_file_path)
