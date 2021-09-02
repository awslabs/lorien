"""
The unit tests for utility functions.
"""
# pylint:disable=missing-docstring, redefined-outer-name, invalid-name
# pylint:disable=unused-argument, unused-import
import os
import tempfile

import boto3
import pytest
from lorien.util import (
    deep_tuple_to_list,
    delete_s3_file,
    download_s3_file,
    dump_to_yaml,
    get_time_str,
    load_from_yaml,
    serialize_framework_build_config,
    split_s3_path,
    upload_s3_file,
)

from .common import mock_s3_client


def test_get_time_str():
    assert get_time_str()


def test_split_s3_path():
    bucket, folder = split_s3_path("s3://bucket_name")
    assert bucket == "bucket_name"
    assert not folder

    bucket, folder = split_s3_path("bucket_name/folder1")
    assert bucket == "bucket_name"
    assert folder == "folder1"

    bucket, folder = split_s3_path("bucket_name/folder1/folder2/")
    assert bucket == "bucket_name"
    assert folder == "folder1/folder2"

    bucket, folder = split_s3_path("bucket_name/folder1/folder2/file.log")
    assert bucket == "bucket_name"
    assert folder == "folder1/folder2/file.log"


def test_manipulate_s3(mock_s3_client):
    with tempfile.NamedTemporaryFile(mode="w", prefix="lorien-test-s3-") as temp_file:
        temp_file.write("aaa\n")
        temp_file.flush()
        ret = upload_s3_file(temp_file.name, "s3://invalid-bucket")
        assert ret.startswith("Failed to upload the file")
        assert upload_s3_file(temp_file.name, "s3://unit-test-bucket/a/b/c/temp") == ""

    with tempfile.NamedTemporaryFile(mode="w", prefix="lorien-test-s3-") as temp_file:
        temp_file_path = temp_file.name

    ret = download_s3_file("s3://invalid-bucket", temp_file_path)
    assert ret.startswith("Failed to download the file")
    assert download_s3_file("s3://unit-test-bucket/a/b/c/temp", temp_file_path, delete=True) == ""
    with open(temp_file_path, "r") as filep:
        context = filep.read()
    os.remove(temp_file_path)
    assert context.find("aaa") != -1

    ret = delete_s3_file("s3://invalid-bucket")
    assert ret.startswith("Failed to delete")


def test_deep_tuple_to_list():
    assert deep_tuple_to_list((1, (2, 3, (4, 5), (6, 7)))) == [1, [2, 3, [4, 5], [6, 7]]]


def test_manipulate_yaml():
    class FakeClass:
        def __init__(self, val):
            self.val = val

    data = [1, {2: 3, 4: 5}]
    assert data == load_from_yaml(dump_to_yaml(data))

    # Local class cannot be loaded
    with pytest.raises(RuntimeError):
        load_from_yaml(dump_to_yaml({"a": 4, "b": FakeClass(5)}))


def test_serialize_framework_build_config():
    assert serialize_framework_build_config({"a": "b", "c": "d"}) == (("a", "b"), ("c", "d"))
    assert serialize_framework_build_config("aa") == ("aa",)
