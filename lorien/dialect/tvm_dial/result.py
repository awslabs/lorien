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
Tuning records of TVM dialect.
"""
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from ...tune.result import Records
from .util import TVM_BUILD_CONFIG

TVMRecordType = TypeVar("TVMRecordType", bound=Tuple)


class TVMRecords(Records[TVMRecordType]):
    """The container to maintain the records of a tuning task."""

    def get_framework_build_config(self) -> Optional[Dict[str, str]]:
        """Get the framework build configurations that generate these records.
        If None, then the committed records will not have this information.
        """
        return TVM_BUILD_CONFIG

    @staticmethod
    def encode(record: TVMRecordType) -> str:
        """Encode a record to a string."""
        raise NotImplementedError

    @staticmethod
    def decode(record_str: str) -> TVMRecordType:
        """Decode a string to a record."""
        raise NotImplementedError

    def gen_task_item(self) -> Dict[str, Any]:
        """Generate an item that can be committed to the database. Note that since all records
        in this container should be for the same task, they should be in the same task item.
        """
        raise NotImplementedError

    @staticmethod
    def gen_record_item(record: TVMRecordType):
        """Generate an item for a record that can be appended to the task item."""
        raise NotImplementedError

    def push(self, record: TVMRecordType):
        """Push a new record.

        Parameters
        ----------
        record: Any
            The record to be pushed.
        """
        raise NotImplementedError

    def pop(self) -> TVMRecordType:
        """Pop the worst record in the container."""
        raise NotImplementedError

    def peak(self) -> TVMRecordType:
        """Peak the first record."""
        raise NotImplementedError

    def to_list(self, nbest: int = -1) -> List[TVMRecordType]:
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
