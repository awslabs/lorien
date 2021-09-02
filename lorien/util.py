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
Utility functions
"""
import os
import time
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union

import boto3
import ruamel.yaml

from .dialect import AVAILABLE_DIALECTS


def is_dialect_enabled(name: str) -> bool:
    """Check if the dialect is enabled.

    Parameters
    ----------
    name: str
        The dialect name.

    Returns
    -------
    ret: bool
        Whether the dialect is enabled.
    """
    return name in AVAILABLE_DIALECTS


def get_time_str() -> str:
    """Generate a string using the current time.

    Returns
    -------
    ret: str
        A string in <year><month><day>-<hour><min> format.
    """
    curr_time = time.gmtime()
    return "{y:04d}{M:02d}{d:02d}-{h:02d}{m:02d}".format(
        y=curr_time.tm_year,
        M=curr_time.tm_mon,
        d=curr_time.tm_mday,
        h=curr_time.tm_hour,
        m=curr_time.tm_min,
    )


def split_s3_path(full_path: str) -> Tuple[str, str]:
    """Split a S3 path to bucket name and folder (file) path.

    Parameters
    ----------
    full_path: str
        A full S3 path in format <bucket-name>/<folder1>/.../<folderN>/<file-name>

    Returns
    -------
    bucket_n_path: Tuple[str, str]
        A pair of bucket name and folder (file) path. Folder (file) path will be an empty string
        if full_path only contains a bucket.
    """

    if full_path.startswith("s3://"):
        full_path = full_path[5:]

    path = Path(full_path)
    return (path.parts[0], os.path.join(*path.parts[1:]) if len(path.parts) > 1 else "")


def delete_s3_file(s3_path: str) -> str:
    """Use boto3 client to delete a file in S3.

    Parameters
    ----------
    s3_path: str
        The S3 file path. It should start with "s3://".

    Returns
    -------
    msg: str
        The error message if failed, or an empty string.
    """
    bucket_name, file_path = split_s3_path(s3_path)

    try:
        client = boto3.client("s3")
        client.delete_object(Bucket=bucket_name, Key=file_path)
    except Exception as err:  # pylint: disable=broad-except
        return "Failed to delete %s in S3: %s" % (s3_path, str(err))
    return ""


def download_s3_file(s3_path: str, local_path: str, delete: bool = False):
    """Use boto3 client to download a file from S3.

    Parameters
    ----------
    s3_path: str
        The S3 file path. It should start with "s3://".

    local_path: str
        The full file path (including the file name) for the downloaded file.

    delete: bool
        Whether to delete the file on S3 after successfully downloaded.

    Returns
    -------
    msg: str
        The error message if failed, or an empty string.
    """
    bucket_name, file_path = split_s3_path(s3_path)

    try:
        client = boto3.client("s3")
        client.download_file(bucket_name, file_path, local_path)
        if delete:
            delete_s3_file(s3_path)
    except Exception as err:  # pylint: disable=broad-except
        return "Failed to download the file from S3: {}".format(str(err))
    return ""


def upload_s3_file(local_path: str, s3_path: str) -> str:
    """Use boto3 client to upload a file to S3.

    Parameters
    ----------
    local_path: str
        The full file path (including the file name) for the file to be uploaded.

    s3_path: str
        The S3 file path. It should start with "s3://".

    Returns
    -------
    msg: str
        The error message if failed, or an empty string.
    """
    bucket_name, file_path = split_s3_path(s3_path)

    try:
        client = boto3.client("s3")
        client.upload_file(local_path, bucket_name, file_path)
    except Exception as err:  # pylint: disable=broad-except
        return "Failed to upload the file to S3: {}".format(str(err))
    return ""


def deep_tuple_to_list(inp: Union[Tuple, List]) -> List[Any]:
    """Transform a nested tuple to a nested list.

    Parameters
    ----------
    inp: Union[Tuple, List]
        The input object.

    Returns
    -------
    ret: List[Any]
        The nested list.
    """
    return list(map(deep_tuple_to_list, inp)) if isinstance(inp, (list, tuple)) else inp


def dump_to_yaml(obj: Any, single_line: bool = True) -> str:
    """Dump an object to a string in YAML format.

    Parameters
    ----------
    obj: Any
        The YAML object.

    single_line: bool
        Whether to generate a sinlge-line style YAML string.

    Returns
    -------
    ret: str
        The dumped YAML string.
    """
    try:
        dumped = ruamel.yaml.dump(
            obj, default_flow_style=single_line, width=float("inf")  # type:ignore
        )
        assert dumped is not None
        return dumped
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to dump %s to YAML: %s" % (str(obj), str(err)))


def load_from_yaml(yaml_str: str, expected_type: Optional[Any] = None) -> Any:
    """Load and construct an object from YAML string.

    Parameters
    ----------
    yaml_str: str
        The YAML string.

    expected_type: Optional[Any]
        The expected object type from the given YAML string. None indicates any type.

    Returns
    -------
    ret: Any
        The constructed object.
    """
    try:
        ret = ruamel.yaml.load(yaml_str, Loader=ruamel.yaml.Loader)
        if expected_type is not None and not isinstance(ret, expected_type):
            raise RuntimeError("Type mismatched: expected %s but get %s" % (expected_type, ret))
        return ret
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to load %s: %s" % (str(yaml_str), str(err)))


def serialize_framework_build_config(dict_: Union[Dict[str, str], str]) -> Tuple[Any, ...]:
    """Serialize a dict to a hashable tuple.

    Parameters
    ----------
    dict_: Dict[str, str]

    Returns
    -------
    hashable_tuple: Tuple[Any, ...]
        A hashable tuple.
    """
    if isinstance(dict_, dict):
        return tuple(sorted(list(dict_.items())))
    return (dict_,)
